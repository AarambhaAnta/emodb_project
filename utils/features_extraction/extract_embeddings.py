"""
Embedding extraction utilities for PLDA training.

Extracts embeddings from trained ECAPA-TDNN models for use in PLDA training.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import hyperpyyaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Monkey patch for torchaudio compatibility
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    def _list_audio_backends():
        return ['soundfile']
    torchaudio.list_audio_backends = _list_audio_backends
    if not hasattr(torchaudio, 'get_audio_backend'):
        torchaudio.get_audio_backend = lambda: 'soundfile'

import speechbrain as sb


def load_ecapa_model(model_dir, hparams_file=None):
    """
    Load trained ECAPA-TDNN model.
    
    Args:
        model_dir: Directory containing trained model checkpoints
        hparams_file: Path to hyperparameters YAML file
    
    Returns:
        embedding_model: Loaded ECAPA-TDNN embedding model
        mean_var_norm: Mean/variance normalization module
    """
    if hparams_file is None:
        project_root = Path(model_dir).parent.parent.parent
        hparams_file = project_root / 'config' / 'ecapa_hparams.yaml'
    
    print(f"Loading ECAPA-TDNN from {model_dir}...")
    print(f"  Hyperparameters: {hparams_file}")
    
    # Load hyperparameters
    with open(hparams_file, 'r') as f:
        hparams = hyperpyyaml.load_hyperpyyaml(f)
    
    embedding_model = hparams['embedding_model']
    mean_var_norm = hparams['mean_var_norm']
    
    # Load consolidated model if available
    model_path = Path(model_dir) / 'model.pt'
    if model_path.exists():
        print(f"  Loading consolidated model: {model_path.name}")
        payload = torch.load(model_path, map_location='cpu')
        embedding_model.load_state_dict(payload['embedding_model'])
        mean_var_norm.load_state_dict(payload['normalizer'])
    else:
        # Fall back to SpeechBrain checkpoints
        checkpoint_dirs = list(Path(model_dir).glob('CKPT+*'))
        if checkpoint_dirs:
            checkpoint_path = sorted(checkpoint_dirs)[-1]
            print(f"  Loading checkpoint: {checkpoint_path.name}")
            
            emb_ckpt = checkpoint_path / 'embedding_model.ckpt'
            norm_ckpt = checkpoint_path / 'normalizer.ckpt'
            
            if emb_ckpt.exists():
                embedding_model.load_state_dict(torch.load(emb_ckpt, map_location='cpu'))
            else:
                raise FileNotFoundError(f"Missing embedding_model.ckpt in {checkpoint_path}")
            
            if norm_ckpt.exists():
                mean_var_norm.load_state_dict(torch.load(norm_ckpt, map_location='cpu'))
            else:
                raise FileNotFoundError(f"Missing normalizer.ckpt in {checkpoint_path}")
        else:
            print("  Warning: No model.pt or checkpoint found, using initialized model")
    
    embedding_model.eval()
    
    print(f"✓ Model loaded successfully")
    
    return embedding_model, mean_var_norm


def extract_embedding_from_mfcc(mfcc_path, embedding_model, mean_var_norm, device='cpu'):
    """
    Extract embedding from single MFCC file.
    
    Args:
        mfcc_path: Path to MFCC .npy file
        embedding_model: ECAPA-TDNN embedding model
        mean_var_norm: Mean/variance normalization module
        device: Device to run on
    
    Returns:
        embedding: Extracted embedding vector
    """
    # Load MFCC features
    feat = np.load(mfcc_path)
    feat = np.squeeze(feat)
    
    if feat.ndim != 2:
        raise ValueError(f"Expected 2D features in {mfcc_path}, got {feat.shape}")
    
    # Ensure shape is (time_frames, features)
    if feat.shape[0] <= 256 and feat.shape[1] > feat.shape[0]:
        feat = feat.T
    
    # Handle NaN/Inf
    if np.isnan(feat).any() or np.isinf(feat).any():
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Convert to tensor
    feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)  # (1, time, features)
    lens = torch.tensor([1.0], device=device)
    
    # Clamp features
    feat_tensor = torch.clamp(feat_tensor, -50.0, 50.0)
    
    # Normalize features
    feat_tensor = mean_var_norm(feat_tensor, lens)
    
    # Extract embedding
    with torch.no_grad():
        emb = embedding_model(feat_tensor, lens)
    
    return emb.squeeze().cpu().numpy()


def extract_embeddings_from_csv(
    csv_path,
    model_dir,
    output_dir,
    hparams_file=None,
    device='cpu'
):
    """
    Extract embeddings for all samples in a CSV file.
    
    Args:
        csv_path: Path to CSV with MFCC paths and labels
        model_dir: Directory containing trained ECAPA model
        output_dir: Directory to save extracted embeddings
        hparams_file: Path to ECAPA hyperparameters file
        device: Device to run on
    
    Returns:
        output_csv: Path to output CSV with embedding paths
    """
    # Load model
    embedding_model, mean_var_norm = load_ecapa_model(model_dir, hparams_file)
    embedding_model = embedding_model.to(device)
    mean_var_norm = mean_var_norm.to(device)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExtracting embeddings from {len(df)} samples...")
    print(f"  Input CSV: {csv_path}")
    print(f"  Output directory: {output_dir}")
    
    # Extract embeddings
    embedding_paths = []
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        mfcc_path = row['mfcc']
        sample_id = row.get('id', idx)
        
        if not os.path.exists(mfcc_path):
            print(f"  Warning: Missing MFCC file {mfcc_path}, skipping...")
            embedding_paths.append(None)
            failed += 1
            continue
        
        try:
            # Extract embedding
            embedding = extract_embedding_from_mfcc(
                mfcc_path,
                embedding_model,
                mean_var_norm,
                device=device
            )
            
            # Save embedding
            emb_filename = f"{sample_id}.npy"
            emb_path = os.path.join(output_dir, emb_filename)
            np.save(emb_path, embedding)
            
            embedding_paths.append(emb_path)
            successful += 1
            
        except Exception as e:
            print(f"  Warning: Failed to extract embedding for {sample_id}: {e}")
            embedding_paths.append(None)
            failed += 1
    
    # Create output DataFrame
    output_df = df.copy()
    output_df['embedding'] = embedding_paths
    
    # Remove rows with failed extractions
    output_df = output_df[output_df['embedding'].notna()]
    
    # Save output CSV
    output_csv = csv_path.replace('.csv', '_embeddings.csv')
    if output_dir != os.path.dirname(csv_path):
        output_csv = os.path.join(output_dir, os.path.basename(output_csv))
    
    output_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Extraction complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Embedding dimension: {embedding.shape[0]}")
    print(f"  Output CSV: {output_csv}")
    
    return output_csv


def extract_embeddings_for_speaker(
    speaker_id,
    loso_dir,
    model_dir,
    output_dir,
    hparams_file=None,
    device='cpu'
):
    """
    Extract embeddings for a single speaker's train and validation sets.
    
    Args:
        speaker_id: Speaker ID (e.g., '03')
        loso_dir: LOSO base directory
        model_dir: Directory containing trained ECAPA model for this speaker
        output_dir: Output directory for embeddings
        hparams_file: Path to ECAPA hyperparameters
        device: Device to run on
    
    Returns:
        train_emb_csv: Path to training embeddings CSV
        val_emb_csv: Path to validation embeddings CSV
    """
    print(f"\n{'='*70}")
    print(f"Extracting Embeddings for Speaker {speaker_id}")
    print(f"{'='*70}")
    
    speaker_loso_dir = os.path.join(loso_dir, f"speaker_{speaker_id}")
    speaker_model_dir = os.path.join(model_dir, f"speaker_{speaker_id}")
    speaker_output_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
    
    if not os.path.exists(speaker_model_dir):
        raise FileNotFoundError(f"Model directory not found: {speaker_model_dir}")
    
    # Extract training set embeddings
    train_csv = os.path.join(speaker_loso_dir, "dev.csv")
    train_output_dir = os.path.join(speaker_output_dir, "train")
    
    if os.path.exists(train_csv):
        print(f"\nTraining Set:")
        train_emb_csv = extract_embeddings_from_csv(
            train_csv,
            speaker_model_dir,
            train_output_dir,
            hparams_file,
            device
        )
    else:
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    
    # Extract validation set embeddings
    val_csv = os.path.join(speaker_loso_dir, "train.csv")
    val_output_dir = os.path.join(speaker_output_dir, "val")
    
    if os.path.exists(val_csv):
        print(f"\nValidation Set:")
        val_emb_csv = extract_embeddings_from_csv(
            val_csv,
            speaker_model_dir,
            val_output_dir,
            hparams_file,
            device
        )
    else:
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")
    
    print(f"\n{'='*70}")
    print(f"✓ Speaker {speaker_id} embeddings extracted")
    print(f"{'='*70}")
    
    return train_emb_csv, val_emb_csv


def main():
    parser = argparse.ArgumentParser(
        description='Extract embeddings from ECAPA-TDNN models for PLDA training'
    )
    
    parser.add_argument(
        '--speaker',
        type=str,
        help='Extract for specific speaker (e.g., 03, 08, 15)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Extract for all speakers'
    )
    parser.add_argument(
        '--loso-dir',
        type=str,
        default='data/processed/loso',
        help='LOSO base directory'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='output/models',
        help='Directory containing trained ECAPA models'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/embeddings',
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--hparams',
        type=str,
        default='config/ecapa_hparams.yaml',
        help='ECAPA hyperparameters file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on'
    )
    
    args = parser.parse_args()
    
    if args.speaker:
        # Extract for single speaker
        extract_embeddings_for_speaker(
            speaker_id=args.speaker,
            loso_dir=args.loso_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            hparams_file=args.hparams,
            device=args.device
        )
    elif args.all:
        # Extract for all speakers
        speakers = sorted([
            d.replace('speaker_', '')
            for d in os.listdir(args.loso_dir)
            if d.startswith('speaker_')
        ])
        
        for speaker_id in speakers:
            try:
                extract_embeddings_for_speaker(
                    speaker_id=speaker_id,
                    loso_dir=args.loso_dir,
                    model_dir=args.model_dir,
                    output_dir=args.output_dir,
                    hparams_file=args.hparams,
                    device=args.device
                )
            except Exception as e:
                print(f"\n✗ Failed to extract embeddings for speaker {speaker_id}: {e}")
                continue
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
