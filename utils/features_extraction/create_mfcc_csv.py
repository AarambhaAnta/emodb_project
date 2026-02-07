#!/usr/bin/env env python3
"""
Create MFCC features CSV for LOSO stage from existing .npy files.

This script creates a CSV file compatible with the LOSO stage by:
1. Reading metadata from segmented_metadata.csv
2. Finding corresponding .npy MFCC feature files
3. Creating a CSV with all required columns: id, duration, wav, mfcc, mfcc_shape, start, stop, spk_id, label

Usage:
    python utils/features_extraction/create_mfcc_csv.py \
        --metadata data/csv/segmented_metadata.csv \
        --features-dir data/processed/features/mfcc \
        --output data/csv/emodb_mfcc_features.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from utils.extract_config import get_config
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from utils.extract_config import get_config


def create_mfcc_csv(metadata_csv=None, features_dir=None, output_csv=None, config=None):
    """
    Create MFCC features CSV from metadata and .npy files.
    
    Args:
        metadata_csv: Path to segmented_metadata.csv (optional, uses config default)
        features_dir: Directory containing .npy MFCC files (optional, uses config default)
        output_csv: Output CSV path (optional, uses config default)
        config: Configuration dictionary (optional, loads from config file)
        
    Returns:
        DataFrame with MFCC features information
    """
    # Load config if not provided
    if config is None:
        config = get_config()
    
    # Set default paths from config
    if metadata_csv is None:
        metadata_csv = os.path.join(
            config['BASE_DIR'],
            config['PATHS']['CSV'],
            'segmented_metadata.csv'
        )
    
    if features_dir is None:
        # Use MFCC OUTPUT_DIR which includes 'mfcc' subdirectory
        features_dir = os.path.join(
            config['BASE_DIR'],
            config['MFCC']['OUTPUT_DIR']
        )
    
    if output_csv is None:
        output_csv = os.path.join(
            config['BASE_DIR'],
            config['PATHS']['CSV'],
            'emodb_mfcc_features.csv'
        )
    
    print(f"Creating MFCC features CSV...")
    print(f"Metadata: {metadata_csv}")
    print(f"Features directory: {features_dir}")
    print(f"Output: {output_csv}")
    print("=" * 70)
    
    # Read metadata
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    
    df = pd.read_csv(metadata_csv)
    print(f"✓ Loaded {len(df)} samples from metadata")
    
    # Check features directory exists
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # Prepare columns
    mfcc_paths = []
    mfcc_shapes = []
    missing_files = []
    
    # Process each sample
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sample_id = row['id']
        
        # Construct .npy file path
        npy_file = os.path.join(features_dir, f"{sample_id}.npy")
        
        if os.path.exists(npy_file):
            # Load to get shape
            mfcc = np.load(npy_file)
            mfcc_paths.append(os.path.abspath(npy_file))
            mfcc_shapes.append(f"{mfcc.shape[0]}x{mfcc.shape[1]}")
        else:
            missing_files.append(sample_id)
            mfcc_paths.append("")
            mfcc_shapes.append("")
    
    # Add MFCC columns
    df['mfcc'] = mfcc_paths
    df['mfcc_shape'] = mfcc_shapes
    
    # Reorder columns to match expected format
    columns_order = ['id', 'duration', 'wav', 'mfcc', 'mfcc_shape', 'start', 'stop', 'spk_id', 'label']
    df = df[columns_order]
    
    # Remove rows with missing files if any
    if missing_files:
        print(f"\n⚠ Warning: {len(missing_files)} samples missing .npy files:")
        for sample_id in missing_files[:10]:
            print(f"  - {sample_id}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        
        df = df[df['mfcc'] != ""]
        print(f"✓ Removed {len(missing_files)} samples with missing files")
    
    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print("=" * 70)
    print(f"✓ Created CSV with {len(df)} samples")
    print(f"✓ Saved to: {output_csv}")
    print(f"\nColumns: {', '.join(df.columns)}")
    print(f"Sample counts by emotion:")
    for label, count in df['label'].value_counts().sort_index().items():
        print(f"  Label {label}: {count} samples")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Create MFCC features CSV for LOSO stage"
    )
    parser.add_argument(
        '--metadata',
        type=str,
        help='Path to segmented metadata CSV (default: from config)'
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        help='Directory containing .npy MFCC files (default: from config, usually data/processed/features/mfcc)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV path (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Create CSV
    df = create_mfcc_csv(
        metadata_csv=args.metadata,
        features_dir=args.features_dir,
        output_csv=args.output
    )
    
    print(f"\n✓ Done! You can now use this CSV with:")
    print(f"  ./run.sh loso")


if __name__ == "__main__":
    main()
