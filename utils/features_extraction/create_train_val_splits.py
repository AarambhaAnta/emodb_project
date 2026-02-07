"""
Create stratified train/validation splits for LOSO cross-validation.

Splits training data 80/20 by emotion for each speaker to enable validation
during training while maintaining speaker independence.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ..extract_config import get_config, get_path


def create_train_val_splits(loso_dir=None, train_ratio=None, random_state=None, config=None):
    """
    Create stratified 80/20 train/validation splits for each speaker.
    
    For each speaker's training data (from other speakers):
    - Split train_ratio (default 80%) for training / (100-train_ratio)% for validation
    - Stratified by emotion label to maintain class distribution
    - Creates separate MFCC file directories for organization
    
    This enables validation during training without using test data.
    
    Args:
        loso_dir (str, optional): Base LOSO directory.
            If None, uses config path or 'data/processed/loso'.
        train_ratio (float, optional): Training data ratio (default 0.8 = 80%).
            If None, uses config or 0.8.
        random_state (int, optional): Random seed for reproducibility.
            If None, uses config or 42.
        config (dict, optional): Configuration dictionary. If None, loads from default.
    
    Returns:
        dict: Dictionary with speaker_id and split statistics:
            - train_80_csv: Path to 80% training CSV (renamed to dev.csv)
            - val_20_csv: Path to 20% validation CSV (renamed to train.csv)
            - train_samples: Number of training samples
            - val_samples: Number of validation samples
            - train_emotions: Distribution of emotions in training set
            - val_emotions: Distribution of emotions in validation set
            
    File Structure Modified:
        loso/speaker_XX/
        ├── other/          # Original training data from LOSO (all other speakers)
        │   ├── *.npy
        ├── dev/            # 80% training data (split from other/)
        │   ├── *.npy
        ├── train/          # 20% validation data (split from other/)
        │   ├── *.npy
        ├── test/           # Original test data (this speaker)
        │   ├── *.npy
        ├── other.csv       # Original training metadata (not modified)
        ├── dev.csv         # 80% training metadata
        ├── train.csv       # 20% validation metadata (for SpeechBrain compatibility)
        └── test.csv        # Original test metadata
        
    Note:
        The naming convention (dev.csv for training, train.csv for validation)
        follows SpeechBrain conventions where 'train' is used for validation.
        
    Example:
        >>> splits_info = create_train_val_splits()
        >>> print(f"Speaker 03 train: {splits_info['03']['train_samples']}")
        >>> print(f"Speaker 03 val: {splits_info['03']['val_samples']}")
        >>> print(f"Train emotions: {splits_info['03']['train_emotions']}")
    """
    # Load config if not provided
    if config is None:
        config = get_config()
    
    # Get parameters from config or use defaults
    if loso_dir is None:
        loso_dir_path = config.get('LOSO', {}).get('OUTPUT_DIR', 'data/processed/loso')
        loso_dir = get_path(config, loso_dir_path)
    
    if train_ratio is None:
        train_ratio = config.get('LOSO', {}).get('TRAIN_RATIO', 0.8)
    
    if random_state is None:
        random_state = config.get('LOSO', {}).get('RANDOM_STATE', 42)
    
    # Get all speaker directories
    speaker_dirs = sorted([d for d in os.listdir(loso_dir) if d.startswith('speaker_')])
    
    if len(speaker_dirs) == 0:
        print(f"Warning: No speaker directories found in {loso_dir}")
        print("Please run create_loso_splits() first.")
        return {}
    
    splits_info = {}
    
    for speaker_dir in tqdm(speaker_dirs, desc="Creating train/val splits"):
        speaker_path = os.path.join(loso_dir, speaker_dir)
        train_csv_path = os.path.join(speaker_path, 'other.csv')
        
        if not os.path.exists(train_csv_path):
            print(f"Warning: {train_csv_path} not found, skipping...")
            continue
        
        # Load training data (from other speakers)
        train_df = pd.read_csv(train_csv_path)
        
        # Stratified split by emotion label
        try:
            train_80, val_20 = train_test_split(
                train_df,
                train_size=train_ratio,
                stratify=train_df['label'],
                random_state=random_state
            )
        except ValueError as e:
            print(f"Warning: Could not stratify {speaker_dir}: {e}")
            # Fall back to non-stratified split
            train_80, val_20 = train_test_split(
                train_df,
                train_size=train_ratio,
                random_state=random_state
            )
        
        # Create directories (dev=80% train, train=20% val for SpeechBrain)
        train_80_dir = os.path.join(speaker_path, 'dev')
        val_20_dir = os.path.join(speaker_path, 'train')
        os.makedirs(train_80_dir, exist_ok=True)
        os.makedirs(val_20_dir, exist_ok=True)
        
        # Copy MFCC files and update paths for train_80 (dev)
        train_80_rows = []
        for idx, row in train_80.iterrows():
            src_mfcc = row['mfcc']
            mfcc_data = np.load(src_mfcc)
            
            mfcc_filename = os.path.basename(src_mfcc)
            dst_mfcc = os.path.join(train_80_dir, mfcc_filename)
            np.save(dst_mfcc, mfcc_data)
            
            new_row = row.copy()
            new_row['mfcc'] = dst_mfcc
            train_80_rows.append(new_row)
        
        train_80_df = pd.DataFrame(train_80_rows)
        train_80_csv = os.path.join(speaker_path, 'dev.csv')
        train_80_df.to_csv(train_80_csv, index=False)
        
        # Copy MFCC files and update paths for val_20 (train)
        val_20_rows = []
        for idx, row in val_20.iterrows():
            src_mfcc = row['mfcc']
            mfcc_data = np.load(src_mfcc)
            
            mfcc_filename = os.path.basename(src_mfcc)
            dst_mfcc = os.path.join(val_20_dir, mfcc_filename)
            np.save(dst_mfcc, mfcc_data)
            
            new_row = row.copy()
            new_row['mfcc'] = dst_mfcc
            val_20_rows.append(new_row)
        
        val_20_df = pd.DataFrame(val_20_rows)
        val_20_csv = os.path.join(speaker_path, 'train.csv')
        val_20_df.to_csv(val_20_csv, index=False)
        
        # Store split information
        speaker_id = speaker_dir.replace('speaker_', '')
        splits_info[speaker_id] = {
            'train_80_csv': train_80_csv,
            'val_20_csv': val_20_csv,
            'train_samples': len(train_80_df),
            'val_samples': len(val_20_df),
            'train_emotions': train_80_df['label'].value_counts().to_dict(),
            'val_emotions': val_20_df['label'].value_counts().to_dict()
        }
    
    print(f"\nCreated train/val splits for {len(splits_info)} speakers")
    print(f"Train ratio: {train_ratio*100:.0f}%, Val ratio: {(1-train_ratio)*100:.0f}%")
    
    return splits_info
