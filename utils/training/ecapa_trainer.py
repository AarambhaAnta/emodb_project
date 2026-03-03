"""ECAPA-TDNN training for emotion recognition (per-speaker LOSO)."""
import os
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torchaudio

# torchaudio compatibility shim for older SpeechBrain versions
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: 'soundfile'

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from ..extract_config import get_config, get_path

logger = logging.getLogger(__name__)

_LABEL_ENCODER_PATH = str(Path(__file__).parent.parent.parent / 'config' / 'label_encoder.txt')


class EmotionBrain(sb.core.Brain):
    """SpeechBrain Brain for ECAPA-TDNN emotion classification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_val_error = float('inf')
        self.epochs_without_improvement = 0
        # Read patience from hparams yaml (key: early_stopping_patience), fallback to 8
        self.early_stopping_patience = getattr(self.hparams, 'early_stopping_patience', 8)

    def on_stage_start(self, stage, epoch=None):
        if stage == sb.Stage.VALID:
            self.error_metrics = self.hparams.error_stats()

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats, lengths = [], []
        for mfcc_path in batch.mfcc:
            arr = np.load(mfcc_path)
            arr = np.squeeze(arr)
            if arr.ndim == 2 and arr.shape[0] <= 256 and arr.shape[1] > arr.shape[0]:
                arr = arr.T  # ensure (time, features)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            t = torch.tensor(arr, dtype=torch.float32)
            feats.append(t)
            lengths.append(t.shape[0])

        feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True).to(self.device)
        lens = torch.tensor(lengths, dtype=torch.float32, device=self.device)
        max_len = feats.shape[1]
        lens = lens/max_len

        lens = torch.clamp(lens, min=1e-6, max=1.0)
        feats = torch.clamp(feats, -50.0, 50.0)

        # Data augmentation: SpecAugment during training only (if enabled)
        use_aug = getattr(self.hparams, 'use_augmentation', True)
        if stage == sb.Stage.TRAIN and use_aug:
            feats = self._apply_spec_augment(feats)

        feats = self.modules.mean_var_norm(feats, lens)
        emb = self.modules.embedding_model(feats, lens)
        out = self.modules.classifier(emb)
        return out, lens

    def _apply_spec_augment(self, feats):
        """Apply time and frequency masking for data augmentation.
        
        Reduced aggressiveness for small datasets: only 30% probability,
        smaller mask sizes.
        """
        batch_size, time_steps, freq_bins = feats.shape
        
        # Time masking: mask up to 5% of time steps (reduced from 10%)
        if time_steps > 20:
            for i in range(batch_size):
                if torch.rand(1).item() > 0.7:  # 30% probability (reduced from 50%)
                    t_mask_width = int(0.05 * time_steps)  # 5% (reduced from 10%)
                    if t_mask_width > 0:
                        t_start = torch.randint(0, max(1, time_steps - t_mask_width), (1,)).item()
                        feats[i, t_start:t_start + t_mask_width, :] = 0
        
        # Frequency masking: mask up to 2-3 frequency bins (reduced from 5)
        if freq_bins > 5:
            for i in range(batch_size):
                if torch.rand(1).item() > 0.7:  # 30% probability (reduced from 50%)
                    f_mask_width = min(3, freq_bins // 12)  # Smaller masks
                    if f_mask_width > 0:
                        f_start = torch.randint(0, max(1, freq_bins - f_mask_width), (1,)).item()
                        feats[i, :, f_start:f_start + f_mask_width] = 0
        
        return feats

    def compute_objectives(self, predictions, batch, stage):
        out, lens = predictions
        emo = batch.label_encoded.data.long()
        if emo.dim() == 1:
            emo = emo.unsqueeze(1)
        loss = self.hparams.compute_cost(out, emo, lens)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, out, emo, lens)
        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            logger.info(
                f"Epoch {epoch} | train_loss: {stage_loss:.4f}"
            )
        elif stage == sb.Stage.VALID:
            self.last_valid_error = self.error_metrics.summarize("average")
            val_acc = 1.0 - self.last_valid_error
            logger.info(
                f"Epoch {epoch} | val_loss: {stage_loss:.4f} | "
                f"val_error: {self.last_valid_error:.4f} | val_acc: {val_acc:.4f} ({val_acc*100:.2f}%)"
            )
            
            # Early stopping logic
            improvement_threshold = 0.005  # Only count as improvement if error reduces by >0.5%
            if self.last_valid_error < (self.best_val_error - improvement_threshold):
                self.best_val_error = self.last_valid_error
                self.epochs_without_improvement = 0
                logger.info(f"✓ New best validation error: {self.best_val_error:.4f}")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement for {self.epochs_without_improvement} epoch(s) (best: {self.best_val_error:.4f})")
            
            # Trigger early stopping if patience exceeded
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                self.hparams.epoch_counter.current = self.hparams.epoch_counter.limit
            
            if self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta={"error": self.last_valid_error},
                    min_keys=["error"],
                )
            
            # Update learning rate if scheduler is ReduceLROnPlateau
            if hasattr(self.hparams, 'lr_annealing'):
                try:
                    old_lr, new_lr = self.hparams.lr_annealing(
                        [self.optimizer], 
                        current_epoch=epoch,
                        current_loss=self.last_valid_error
                    )
                    if old_lr != new_lr:
                        logger.info(f"Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
                    else:
                        logger.info(f"Current learning rate: {new_lr:.2e}")
                except Exception as e:
                    logger.debug(f"LR scheduler update skipped: {e}")


def _prepare_datasets(train_csv, val_csv, tmp_dir):
    """Load train DynamicItemDataset with encoded labels."""
    os.makedirs(tmp_dir, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    if 'id' in train_df.columns and 'ID' not in train_df.columns:
        train_df = train_df.rename(columns={'id': 'ID'})
    
    val_df = pd.read_csv(val_csv)
    if 'id' in val_df.columns and 'ID' not in val_df.columns:
        val_df = val_df.rename(columns={'id': 'ID'})
    
    dst = os.path.join(tmp_dir, 'train.csv')
    train_df.to_csv(dst, index=False)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(dst)

    dst = os.path.join(tmp_dir, 'val.csv')
    val_df.to_csv(dst, index=False)
    val_data = sb.dataio.dataset.DynamicItemDataset.from_csv(dst)

    datasets = [train_data, val_data]

    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_encoded")
    def label_pipeline(label):
        yield label_encoder.encode_label_torch(label)

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    label_encoder.load_or_create(
        path = _LABEL_ENCODER_PATH,
        from_didatasets = [train_data],
        output_key = "label",
    )

    sb.dataio.dataset.set_output_keys(datasets, ["id", "mfcc",  "label_encoded"])

    return train_data, val_data


def train_speaker_model(speaker_id, loso_dir, output_dir, hparams, run_opts,
                        tmp_dir="/tmp/emodb_training"):
    """Train ECAPA-TDNN for one speaker on all non-test data; saves model.pt."""
    speaker_dir = os.path.join(loso_dir, f"speaker_{speaker_id}")
    train_csv = os.path.join(speaker_dir, "train.csv")
    val_csv = os.path.join(speaker_dir, "val.csv")

    train_data, val_data = _prepare_datasets(train_csv, val_csv, tmp_dir)
    logger.info(f"Speaker {speaker_id}: {len(train_data)} train samples")

    out_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
    os.makedirs(out_dir, exist_ok=True)
    hparams["save_folder"] = out_dir
    hparams["checkpointer"] = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=out_dir,
        recoverables={
            "embedding_model": hparams["embedding_model"],
            "classifier":      hparams["classifier"],
            "normalizer":      hparams["mean_var_norm"],
            "counter":         hparams["epoch_counter"],
        },
    )

    brain = EmotionBrain(
        modules=hparams["modules"], 
        opt_class=hparams["opt_class"],
        hparams=hparams, 
        run_opts=run_opts, 
        checkpointer=hparams["checkpointer"],
    )
    val_loader_kwargs = hparams.get("val_dataloader_options", {
        "batch_size": hparams["dataloader_options"]["batch_size"],
        "num_workers": hparams["dataloader_options"].get("num_workers", 0),
        "shuffle": False,
        "drop_last": False,
    })
    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=train_data,
        valid_set=val_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=val_loader_kwargs,
    )

    # CRITICAL FIX: Load the best checkpoint before saving model.pt
    logger.info(f"Speaker {speaker_id}: Loading best checkpoint...")
    brain.checkpointer.recover_if_possible(min_key="error")
    logger.info(f"Speaker {speaker_id}: Best validation error = {brain.best_val_error:.4f}")

    torch.save({
        "embedding_model": hparams["embedding_model"].state_dict(),
        "classifier":      hparams["classifier"].state_dict(),
        "normalizer":      hparams["mean_var_norm"].state_dict(),
        "speaker_id":      speaker_id,
        "best_val_error":  brain.best_val_error,
    }, os.path.join(out_dir, "model.pt"))

    return brain.best_val_error


def train_all_speakers(loso_dir=None, output_dir=None, hparams_file=None, run_opts=None):
    """Train ECAPA-TDNN for all speakers; returns per-speaker results dict."""
    config = get_config()
    loso_dir   = loso_dir   or get_path(config, config['PATHS']['LOSO'])
    output_dir = output_dir or get_path(config, config['PATHS']['MODELS'])
    hparams_path = (hparams_file if isinstance(hparams_file, str)
                    else os.path.join(config['BASE_DIR'], 'config', 'ecapa_hparams.yaml'))
    os.makedirs(output_dir, exist_ok=True)

    speakers = sorted(d.replace('speaker_', '') for d in os.listdir(loso_dir)
                      if d.startswith('speaker_'))
    results = {}
    for spk in tqdm(speakers, desc="Training speakers"):
        try:
            with open(hparams_path) as f:
                hparams = load_hyperpyyaml(f)
            error = train_speaker_model(spk, loso_dir, output_dir, hparams, run_opts)
            results[spk] = {'status': 'success', 'best_error': error}
            logger.info(f"Speaker {spk} done — error={error:.4f}")
        except Exception as e:
            results[spk] = {'status': 'failed', 'error': str(e)}
            logger.error(f"Speaker {spk} failed: {e}")
    return results
