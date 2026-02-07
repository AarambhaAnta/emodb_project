"""
ECAPA-TDNN training utilities for emotion recognition.
"""
import os
import torch
import numpy as np
import pandas as pd

# Monkey patch for torchaudio compatibility with newer versions
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    def _list_audio_backends():
        """Dummy function for compatibility with older SpeechBrain versions."""
        return ['soundfile']
    torchaudio.list_audio_backends = _list_audio_backends
    if not hasattr(torchaudio, 'get_audio_backend'):
        torchaudio.get_audio_backend = lambda: 'soundfile'

import speechbrain as sb
from tqdm import tqdm

from ..extract_config import get_config, get_path


class EmotionBrain(sb.core.Brain):
    """Custom Brain class for emotion recognition training."""
    
    def __init__(self, *args, **kwargs):
        """Initialize EmotionBrain with error tracking."""
        super().__init__(*args, **kwargs)
        self.error_metrics = None
        self.last_valid_error = None
    
    def on_stage_start(self, stage, epoch=None):
        """Initialize metrics at the start of each stage."""
        print(f"Stage {stage} starting (epoch {epoch})")
        if stage == sb.Stage.VALID:
            self.error_metrics = self.hparams.error_stats()
            print("Initialized error_metrics for validation")
    
    def compute_forward(self, batch, stage):
        """
        Forward pass: load MFCC features and compute embeddings.
        
        Args:
            batch: Batch of data
            stage: Training stage (TRAIN/VALID/TEST)
        
        Returns:
            predictions: Model output
            lens: Sequence lengths
        """
        batch = batch.to(self.device)
        
        feats = []
        lengths = []
        
        # Load MFCC features for each utterance
        for mfcc_path in batch.mfcc:
            if not os.path.exists(mfcc_path):
                raise FileNotFoundError(f"Missing MFCC file: {mfcc_path}")
            
            # Load MFCC array
            arr = np.load(mfcc_path)
            arr = np.squeeze(arr)
            
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D features in {mfcc_path}, got {arr.shape}")
            
            # Ensure shape is (time_frames, features)
            if arr.shape[0] <= 256 and arr.shape[1] > arr.shape[0]:
                arr = arr.T
            
            # Handle NaN/Inf
            if np.isnan(arr).any() or np.isinf(arr).any():
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            
            feat_t = torch.tensor(arr, dtype=torch.float32)
            feats.append(feat_t)
            lengths.append(feat_t.shape[0])
        
        # Pad sequences
        feats_tensor = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        feats_tensor = feats_tensor.to(self.device)
        
        # Compute relative lengths
        lengths = torch.tensor(lengths, dtype=torch.float32, device=self.device)
        max_len = feats_tensor.shape[1]
        lens = lengths / max_len
        lens = torch.clamp(lens, min=1e-6, max=1.0)
        
        # Clamp features
        feats_tensor = torch.clamp(feats_tensor, -50.0, 50.0)
        
        # Normalize features
        feats_tensor = self.modules.mean_var_norm(feats_tensor, lens)
        
        # Extract embeddings
        emb = self.modules.embedding_model(feats_tensor, lens)
        
        # Classify
        out = self.modules.classifier(emb)
        
        if not torch.isfinite(out).all():
            raise ValueError("NaN/Inf in classifier output")
        
        return out, lens
    
    def compute_objectives(self, predictions, batch, stage):
        """
        Compute loss and update metrics.
        
        Args:
            predictions: Model predictions
            batch: Batch of data
            stage: Training stage
        
        Returns:
            loss: Computed loss
        """
        predictions, lens = predictions
        
        emo = batch.label_encoded.data.long()
        
        # Keep as [B, 1] for SpeechBrain metrics
        if emo.dim() == 1:
            emo = emo.unsqueeze(1)
        
        loss = self.hparams.compute_cost(predictions, emo, lens)
        
        if stage != sb.Stage.TRAIN and self.error_metrics is not None:
            self.error_metrics.append(batch.id, predictions, emo, lens)
        
        return loss
    
    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Update metrics at the end of each stage."""
        print(f"Stage {stage} ending (epoch {epoch})")
        if stage == sb.Stage.VALID and self.error_metrics is not None:
            self.last_valid_error = self.error_metrics.summarize("average")
            print(f"Validation error: {self.last_valid_error}")
        elif stage == sb.Stage.VALID:
            print("WARNING: error_metrics is None at end of validation stage!")


def prepare_datasets(hparams, train_csv_path, val_csv_path, tmp_dir):
    """
    Prepare train and validation datasets.
    
    Args:
        hparams: Hyperparameters
        train_csv_path: Path to training CSV
        val_csv_path: Path to validation CSV
        tmp_dir: Temporary directory for label encoder
    
    Returns:
        train_data: Training dataset
        valid_data: Validation dataset
    """
    # Create temporary directory
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Fix CSV column names (SpeechBrain expects 'ID' not 'id')
    temp_train_csv = os.path.join(tmp_dir, "temp_train.csv")
    temp_val_csv = os.path.join(tmp_dir, "temp_val.csv")
    
    for src, dst in [(train_csv_path, temp_train_csv), (val_csv_path, temp_val_csv)]:
        df = pd.read_csv(src)
        if 'id' in df.columns and 'ID' not in df.columns:
            df = df.rename(columns={'id': 'ID'})
        df.to_csv(dst, index=False)
    
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(temp_train_csv)
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(temp_val_csv)
    
    # Verify datasets are not empty
    if len(train_data) == 0:
        raise ValueError(f"Training dataset is empty! Check {train_csv_path}")
    if len(valid_data) == 0:
        raise ValueError(f"Validation dataset is empty! Check {val_csv_path}")
    
    datasets = [train_data, valid_data]
    
    # Create label encoder
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_encoded")
    def label_pipeline(label):
        yield label_encoder.encode_label_torch(label)
    
    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)
    
    # Load or create label encoder
    os.makedirs(tmp_dir, exist_ok=True)
    label_encoder.load_or_create(
        path=os.path.join(tmp_dir, "label_encoder.txt"),
        from_didatasets=[train_data],
        output_key="label",
    )
    
    # Set expected number of emotion categories (7: anger, boredom, disgust, fear, happiness, sadness, neutral)
    label_encoder.expect_len(7)
    
    # Set output keys
    sb.dataio.dataset.set_output_keys(datasets, ["id", "mfcc", "label_encoded"])
    
    return train_data, valid_data


def train_speaker_model(speaker_id, loso_dir, output_dir, hparams, run_opts, tmp_dir="/tmp/emodb_training"):
    """
    Train ECAPA-TDNN model for one speaker using LOSO.
    
    Args:
        speaker_id: Speaker ID (e.g., '03')
        loso_dir: Base LOSO directory
        output_dir: Output directory for models
        hparams: Hyperparameters
        run_opts: Run options
        tmp_dir: Temporary directory
    
    Returns:
        best_valid_error: Best validation error achieved
    """
    speaker_dir = os.path.join(loso_dir, f"speaker_{speaker_id}")
    # dev.csv = 80% training data, train.csv = 20% validation data
    train_csv = os.path.join(speaker_dir, "dev.csv")
    val_csv = os.path.join(speaker_dir, "train.csv")
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing dev.csv for speaker {speaker_id}: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Missing train.csv for speaker {speaker_id}: {val_csv}")
    
    # Prepare datasets
    print(f"Loading datasets for speaker {speaker_id}...")
    train_data, valid_data = prepare_datasets(hparams, train_csv, val_csv, tmp_dir)
    print(f"Train samples: {len(train_data)}, Valid samples: {len(valid_data)}")
    
    # Create output directory for this speaker
    speaker_output_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
    os.makedirs(speaker_output_dir, exist_ok=True)
    
    # Update save folder in hparams for checkpointer
    hparams["save_folder"] = speaker_output_dir
    
    # Initialize Brain
    print(f"Initializing model for speaker {speaker_id}...")
    brain = EmotionBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # Train model
    print(f"Starting training for speaker {speaker_id}...")
    try:
        brain.fit(
            epoch_counter=hparams["epoch_counter"],
            train_set=train_data,
            valid_set=valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    except Exception as e:
        print(f"ERROR during training for speaker {speaker_id}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Get best validation error from last validation stage
    best_valid_error = brain.last_valid_error if brain.last_valid_error is not None else float('inf')
    
    if best_valid_error == float('inf'):
        print(f"WARNING: Speaker {speaker_id} - No valid error recorded (validation may not have run)")
    
    return best_valid_error


def train_all_speakers(loso_dir=None, output_dir=None, hparams_file=None, run_opts=None):
    """
    Train ECAPA-TDNN models for all speakers using LOSO.
    
    Args:
        loso_dir: Base LOSO directory
        output_dir: Output directory for models
        hparams_file: Path to hyperparameters YAML file OR loaded hparams dict
        run_opts: Run options
    
    Returns:
        results: Dictionary with training results for each speaker
    """
    config = None
    if loso_dir is None or output_dir is None or isinstance(hparams_file, dict) or hparams_file is None:
        config = get_config()
    
    if loso_dir is None:
        loso_dir_path = config.get('LOSO', {}).get('OUTPUT_DIR', config.get('PATHS', {}).get('LOSO', 'data/processed/loso'))
        loso_dir = get_path(config, loso_dir_path)
    
    if output_dir is None:
        output_dir_path = config.get('PATHS', {}).get('MODELS', 'output/models')
        output_dir = get_path(config, output_dir_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get hparams file path if a dict was passed
    if isinstance(hparams_file, dict) or hparams_file is None:
        hparams_path = os.path.join(config['BASE_DIR'], 'config', 'ecapa_hparams.yaml')
    else:
        hparams_path = hparams_file
    
    # Get all speaker directories
    speaker_dirs = sorted([d for d in os.listdir(loso_dir) if d.startswith('speaker_')])
    speakers = [d.replace('speaker_', '') for d in speaker_dirs]
    
    results = {}
    
    for speaker_id in tqdm(speakers, desc="Training speakers"):
        print(f"\n{'='*70}")
        print(f"Training speaker {speaker_id}")
        print(f"{'='*70}")
        
        try:
            # Reload hyperparameters for EACH speaker to reset epoch counter
            from hyperpyyaml import load_hyperpyyaml
            print(f"Loading fresh hyperparameters for speaker {speaker_id}...")
            with open(hparams_path) as f:
                speaker_hparams = load_hyperpyyaml(f)
            print(f"Epoch counter initialized: current={speaker_hparams['epoch_counter'].current}, limit={speaker_hparams['epoch_counter'].limit}")
            
            best_error = train_speaker_model(
                speaker_id=speaker_id,
                loso_dir=loso_dir,
                output_dir=output_dir,
                hparams=speaker_hparams,
                run_opts=run_opts
            )
            
            results[speaker_id] = {
                'status': 'success',
                'best_error': best_error
            }
            
            print(f"✓ Speaker {speaker_id} - Best error: {best_error:.4f}")
            
        except Exception as e:
            results[speaker_id] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"✗ Speaker {speaker_id} - Failed: {e}")
    
    return results
