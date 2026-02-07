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
        --features-dir data/processed/features \
        --output data/csv/emodb_mfcc_features.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os


def create_mfcc_csv(metadata_csv, features_dir, output_csv):
    """
    Create MFCC features CSV from metadata and .npy files.
    
    Args:
        metadata_csv: Path to segmented_metadata.csv
        features_dir: Directory containing .npy MFCC files
        output_csv: Output CSV path
        
    Returns:
        DataFrame with MFCC features information
    """
    print(f"Creating MFCC features CSV...")
    print(f"Metadata: {metadata_csv}")
    print(f"Features directory: {features_dir}")
    print(f"Output: {output_csv}")
    print("=" * 70)
    
    # Read metadata
    df = pd.read_csv(metadata_csv)
    print(f"✓ Loaded {len(df)} samples from metadata")
    
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
        default='data/csv/segmented_metadata.csv',
        help='Path to segmented metadata CSV'
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default='data/processed/features',
        help='Directory containing .npy MFCC files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/csv/emodb_mfcc_features.csv',
        help='Output CSV path'
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
