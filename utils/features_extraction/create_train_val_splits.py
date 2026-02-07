"""
Create stratified train/validation splits for LOSO cross-validation.
Splits training data 80/20 by emotion for each speaker.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def create_train_val_splits(loso_dir=None, train_ratio=0.8, random_state=42):
    """
    Create stratified 80/20 train/validation splits for each speaker.
    
    For each speaker's training data:
    - Split 80% train / 20% validation
    - Stratified by emotion label
    - Maintain MFCC files organization
    
    Args:
        loso_dir (str, optional): Base LOSO directory
        train_ratio (float): Training data ratio (default 0.8)
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary with speaker_id and split statistics
    """
    if loso_dir is None:
        loso_dir = '/Users/adityakumar/Developer/Projects/emodb_project/data/processed/loso'
    
    # Get all speaker directories
    speaker_dirs = sorted([d for d in os.listdir(loso_dir) if d.startswith('speaker_')])
    
    splits_info = {}
    
    for speaker_dir in tqdm(speaker_dirs, desc="Creating train/val splits"):
        speaker_path = os.path.join(loso_dir, speaker_dir)
        train_csv_path = os.path.join(speaker_path, 'train.csv')
        
        if not os.path.exists(train_csv_path):
            continue
        
        # Load training data
        train_df = pd.read_csv(train_csv_path)
        
        # Stratified split by emotion label
        train_80, val_20 = train_test_split(
            train_df,
            train_size=train_ratio,
            stratify=train_df['label'],
            random_state=random_state
        )
        
        # Create directories
        train_80_dir = os.path.join(speaker_path, 'dev')
        val_20_dir = os.path.join(speaker_path, 'train')
        os.makedirs(train_80_dir, exist_ok=True)
        os.makedirs(val_20_dir, exist_ok=True)
        
        # Copy MFCC files and update paths for train_80
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
        
        # Copy MFCC files and update paths for val_20
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
        
        # Store split info
        speaker_id = speaker_dir.replace('speaker_', '')
        splits_info[speaker_id] = {
            'train_80_csv': train_80_csv,
            'val_20_csv': val_20_csv,
            'train_samples': len(train_80_df),
            'val_samples': len(val_20_df),
            'train_emotions': train_80_df['label'].value_counts().to_dict(),
            'val_emotions': val_20_df['label'].value_counts().to_dict()
        }
    
    return splits_info
