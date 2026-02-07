"""
Create Leave-One-Speaker-Out (LOSO) cross-validation splits for MFCC features.

This module creates speaker-independent train/test splits for emotion recognition.
For each speaker, all other speakers' data becomes training data, and the target
speaker's data becomes test data.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..extract_config import get_config, get_path


def create_loso_splits(input_csv=None, output_dir=None, config=None):
    """
    Create LOSO train/test splits with MFCC features.
    
    For each speaker:
    - Train set: MFCC features from ALL OTHER speakers
    - Test set: MFCC features from THIS speaker
    
    This enables speaker-independent emotion recognition evaluation.
    
    Args:
        input_csv (str, optional): Path to input CSV with MFCC metadata.
            If None, uses config path or 'data/csv/emodb_mfcc_features.csv'.
        output_dir (str, optional): Base directory for LOSO splits.
            If None, uses config path or 'data/processed/loso'.
        config (dict, optional): Configuration dictionary. If None, loads from default.
    
    Returns:
        dict: Dictionary with speaker_id as key and paths as values:
            - train_csv: Path to training CSV
            - test_csv: Path to test CSV
            - train_samples: Number of training samples
            - test_samples: Number of test samples
            
    File Structure Created:
        loso/
        ├── speaker_03/
        │   ├── other/           # MFCC files from other speakers (training)
        │   │   ├── *.npy
        │   ├── test/            # MFCC files from this speaker
        │   │   ├── *.npy
        │   ├── train.csv        # Training metadata
        │   └── test.csv         # Test metadata
        ├── speaker_08/
        │   └── ...
        
    Example:
        >>> splits_info = create_loso_splits()
        >>> print(f"Created splits for {len(splits_info)} speakers")
        >>> print(f"Speaker 03: {splits_info['03']['train_samples']} train, "
        ...       f"{splits_info['03']['test_samples']} test")
    """
    # Load config if not provided
    if config is None:
        config = get_config()
    
    # Get paths from config
    if input_csv is None:
        input_csv_path = config.get('MFCC', {}).get('OUTPUT_CSV', 
                                                    'data/csv/emodb_mfcc_features.csv')
        input_csv = get_path(config, input_csv_path)
    
    if output_dir is None:
        output_dir_path = config.get('LOSO', {}).get('OUTPUT_DIR', 'data/processed/loso')
        output_dir = get_path(config, output_dir_path)
    
    # Load MFCC metadata
    df = pd.read_csv(input_csv)
    speakers = sorted(df['spk_id'].unique())
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    splits_info = {}
    
    for speaker_id in tqdm(speakers, desc="Creating LOSO splits"):
        # Format speaker ID with leading zero
        speaker_id_str = str(speaker_id).zfill(2)
        
        # Create directories for this speaker
        speaker_dir = os.path.join(output_dir, f"speaker_{speaker_id_str}")
        train_dir = os.path.join(speaker_dir, "other")
        test_dir = os.path.join(speaker_dir, "test")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Split data: this speaker vs. all others
        test_df = df[df['spk_id'] == speaker_id].copy()
        train_df = df[df['spk_id'] != speaker_id].copy()
        
        # Copy MFCC files for TRAIN set (other speakers)
        train_rows = []
        for idx, row in train_df.iterrows():
            src_mfcc = row['mfcc']
            mfcc_data = np.load(src_mfcc)
            
            # Save to speaker's train directory
            mfcc_filename = os.path.basename(src_mfcc)
            dst_mfcc = os.path.join(train_dir, mfcc_filename)
            np.save(dst_mfcc, mfcc_data)
            
            # Update MFCC path
            new_row = row.copy()
            new_row['mfcc'] = dst_mfcc
            train_rows.append(new_row)
        
        # Save training CSV
        train_csv = pd.DataFrame(train_rows)
        train_csv_path = os.path.join(speaker_dir, "train.csv")
        train_csv.to_csv(train_csv_path, index=False)
        
        # Copy MFCC files for TEST set (this speaker)
        test_rows = []
        for idx, row in test_df.iterrows():
            src_mfcc = row['mfcc']
            mfcc_data = np.load(src_mfcc)
            
            # Save to speaker's test directory
            mfcc_filename = os.path.basename(src_mfcc)
            dst_mfcc = os.path.join(test_dir, mfcc_filename)
            np.save(dst_mfcc, mfcc_data)
            
            # Update MFCC path
            new_row = row.copy()
            new_row['mfcc'] = dst_mfcc
            test_rows.append(new_row)
        
        # Save test CSV
        test_csv = pd.DataFrame(test_rows)
        test_csv_path = os.path.join(speaker_dir, "test.csv")
        test_csv.to_csv(test_csv_path, index=False)
        
        # Store split information
        splits_info[speaker_id_str] = {
            'train_csv': train_csv_path,
            'test_csv': test_csv_path,
            'train_samples': len(train_csv),
            'test_samples': len(test_csv)
        }
    
    print(f"\nCreated LOSO splits for {len(splits_info)} speakers in: {output_dir}")
    print(f"Total speakers: {len(speakers)}")
    
    return splits_info
