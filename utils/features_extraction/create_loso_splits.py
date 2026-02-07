"""
Create Leave-One-Speaker-Out (LOSO) cross-validation splits for MFCC features.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.extract_config import get_config


def create_loso_splits(input_csv=None, output_dir=None):
    """
    Create LOSO train/test splits with MFCC features.
    
    For each speaker:
    - Train set: MFCC from OTHER speakers
    - Test set: MFCC from THIS speaker
    
    Args:
        input_csv (str, optional): Path to input CSV with MFCC metadata
        output_dir (str, optional): Base directory for LOSO splits
    
    Returns:
        dict: Dictionary with speaker_id as key and paths as values
    """
    config_path = '/Users/adityakumar/Developer/Projects/emodb_project/config/emodb_config.yaml'
    config = get_config(config_path)
    
    if input_csv is None:
        input_csv = os.path.join(config['BASE_DIR'], config['MFCC']['OUTPUT_CSV'])
    
    if output_dir is None:
        output_dir = os.path.join(config['BASE_DIR'], 'data/processed/loso')
    
    df = pd.read_csv(input_csv)
    speakers = sorted(df['spk_id'].unique())
    
    os.makedirs(output_dir, exist_ok=True)
    
    splits_info = {}
    
    for speaker_id in tqdm(speakers, desc="Creating LOSO splits"):
        speaker_id_str = str(speaker_id).zfill(2)
        speaker_dir = os.path.join(output_dir, f"speaker_{speaker_id_str}")
        train_dir = os.path.join(speaker_dir, "other")
        test_dir = os.path.join(speaker_dir, "test")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Split data
        test_df = df[df['spk_id'] == speaker_id].copy()
        train_df = df[df['spk_id'] != speaker_id].copy()
        
        # Copy MFCC files for TRAIN set
        train_rows = []
        for idx, row in train_df.iterrows():
            src_mfcc = row['mfcc']
            mfcc_data = np.load(src_mfcc)
            
            mfcc_filename = os.path.basename(src_mfcc)
            dst_mfcc = os.path.join(train_dir, mfcc_filename)
            np.save(dst_mfcc, mfcc_data)
            
            new_row = row.copy()
            new_row['mfcc'] = dst_mfcc
            train_rows.append(new_row)
        
        train_csv = pd.DataFrame(train_rows)
        train_csv_path = os.path.join(speaker_dir, "train.csv")
        train_csv.to_csv(train_csv_path, index=False)
        
        # Copy MFCC files for TEST set
        test_rows = []
        for idx, row in test_df.iterrows():
            src_mfcc = row['mfcc']
            mfcc_data = np.load(src_mfcc)
            
            mfcc_filename = os.path.basename(src_mfcc)
            dst_mfcc = os.path.join(test_dir, mfcc_filename)
            np.save(dst_mfcc, mfcc_data)
            
            new_row = row.copy()
            new_row['mfcc'] = dst_mfcc
            test_rows.append(new_row)
        
        test_csv = pd.DataFrame(test_rows)
        test_csv_path = os.path.join(speaker_dir, "test.csv")
        test_csv.to_csv(test_csv_path, index=False)
        
        splits_info[speaker_id_str] = {
            'train_csv': train_csv_path,
            'test_csv': test_csv_path,
            'train_samples': len(train_csv),
            'test_samples': len(test_csv)
        }
    
    return splits_info
