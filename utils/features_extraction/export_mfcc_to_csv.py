"""
Export MFCC features to CSV format for MATLAB or other tools.

This module provides utilities to convert MFCC .npy files into CSV format
for compatibility with MATLAB and other analysis tools.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def export_mfcc_to_csv(
    input_csv,
    output_dir=None,
    format_type='flat',
    include_metadata=True,
    config=None
):
    """
    Export MFCC features from .npy files to CSV format.
    
    Args:
        input_csv (str): Path to CSV with MFCC file paths (e.g., emodb_mfcc_features.csv)
        output_dir (str, optional): Directory to save CSV files. If None, uses input CSV directory
        format_type (str): Output format:
            - 'flat': One row per sample, columns are mfcc_1, mfcc_2, ..., mfcc_40 (averaged over time)
            - 'stats': One row per sample with statistics (mean, std, min, max) for each coefficient
            - 'frames': One row per frame with all coefficients (creates larger CSV)
            - 'separate': One CSV file per sample (for MATLAB cell arrays)
        include_metadata (bool): Whether to include id, duration, wav, spk_id, label columns
        config (dict, optional): Configuration dictionary
    
    Returns:
        str: Path to output CSV file(s)
    
    Example:
        >>> # Export as averaged features (flat format)
        >>> output = export_mfcc_to_csv(
        ...     'data/csv/emodb_mfcc_features.csv',
        ...     format_type='flat'
        ... )
        
        >>> # Export with statistics (mean, std, min, max)
        >>> output = export_mfcc_to_csv(
        ...     'data/csv/emodb_mfcc_features.csv',
        ...     format_type='stats'
        ... )
    """
    # Load input CSV
    df = pd.read_csv(input_csv)
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    if format_type == 'flat':
        return _export_flat_format(df, output_dir, include_metadata)
    elif format_type == 'stats':
        return _export_stats_format(df, output_dir, include_metadata)
    elif format_type == 'frames':
        return _export_frames_format(df, output_dir, include_metadata)
    elif format_type == 'separate':
        return _export_separate_format(df, output_dir, include_metadata)
    else:
        raise ValueError(f"Unknown format_type: {format_type}. Choose from: flat, stats, frames, separate")


def _export_flat_format(df, output_dir, include_metadata):
    """Export MFCC as one row per sample with averaged coefficients."""
    rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exporting MFCC (flat)"):
        mfcc_path = row['mfcc']
        mfcc_data = np.load(mfcc_path)  # Shape: (n_mfcc, n_frames)
        
        # Average over time (frames)
        mfcc_mean = np.mean(mfcc_data, axis=1)  # Shape: (n_mfcc,)
        
        # Create row dict
        row_dict = {}
        if include_metadata:
            row_dict['id'] = row['id']
            row_dict['duration'] = row['duration']
            row_dict['wav'] = row['wav']
            row_dict['spk_id'] = row['spk_id']
            row_dict['label'] = row['label']
        
        # Add MFCC coefficients
        for i, val in enumerate(mfcc_mean):
            row_dict[f'mfcc_{i+1}'] = val
        
        rows.append(row_dict)
    
    # Create DataFrame
    result_df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'mfcc_features_flat.csv')
    result_df.to_csv(output_path, index=False)
    
    print(f"✓ Exported {len(result_df)} samples to: {output_path}")
    print(f"  Format: flat (averaged over time)")
    print(f"  Columns: {len(result_df.columns)}")
    
    return output_path


def _export_stats_format(df, output_dir, include_metadata):
    """Export MFCC with statistics (mean, std, min, max) for each coefficient."""
    rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exporting MFCC (stats)"):
        mfcc_path = row['mfcc']
        mfcc_data = np.load(mfcc_path)  # Shape: (n_mfcc, n_frames)
        
        # Compute statistics over time
        mfcc_mean = np.mean(mfcc_data, axis=1)
        mfcc_std = np.std(mfcc_data, axis=1)
        mfcc_min = np.min(mfcc_data, axis=1)
        mfcc_max = np.max(mfcc_data, axis=1)
        
        # Create row dict
        row_dict = {}
        if include_metadata:
            row_dict['id'] = row['id']
            row_dict['duration'] = row['duration']
            row_dict['wav'] = row['wav']
            row_dict['spk_id'] = row['spk_id']
            row_dict['label'] = row['label']
        
        # Add MFCC statistics
        for i in range(len(mfcc_mean)):
            row_dict[f'mfcc_{i+1}_mean'] = mfcc_mean[i]
            row_dict[f'mfcc_{i+1}_std'] = mfcc_std[i]
            row_dict[f'mfcc_{i+1}_min'] = mfcc_min[i]
            row_dict[f'mfcc_{i+1}_max'] = mfcc_max[i]
        
        rows.append(row_dict)
    
    # Create DataFrame
    result_df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'mfcc_features_stats.csv')
    result_df.to_csv(output_path, index=False)
    
    print(f"✓ Exported {len(result_df)} samples to: {output_path}")
    print(f"  Format: stats (mean, std, min, max)")
    print(f"  Columns: {len(result_df.columns)}")
    
    return output_path


def _export_frames_format(df, output_dir, include_metadata):
    """Export MFCC with one row per frame (large CSV)."""
    rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exporting MFCC (frames)"):
        mfcc_path = row['mfcc']
        mfcc_data = np.load(mfcc_path)  # Shape: (n_mfcc, n_frames)
        
        # Transpose to (n_frames, n_mfcc)
        mfcc_transposed = mfcc_data.T
        
        # Create one row per frame
        for frame_idx, frame_data in enumerate(mfcc_transposed):
            row_dict = {}
            if include_metadata:
                row_dict['id'] = row['id']
                row_dict['frame'] = frame_idx
                row_dict['duration'] = row['duration']
                row_dict['wav'] = row['wav']
                row_dict['spk_id'] = row['spk_id']
                row_dict['label'] = row['label']
            
            # Add MFCC coefficients for this frame
            for i, val in enumerate(frame_data):
                row_dict[f'mfcc_{i+1}'] = val
            
            rows.append(row_dict)
    
    # Create DataFrame
    result_df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'mfcc_features_frames.csv')
    result_df.to_csv(output_path, index=False)
    
    print(f"✓ Exported {len(result_df)} frames to: {output_path}")
    print(f"  Format: frames (one row per frame)")
    print(f"  Columns: {len(result_df.columns)}")
    print(f"  Warning: Large file size!")
    
    return output_path


def _export_separate_format(df, output_dir, include_metadata):
    """Export each MFCC sample as a separate CSV file."""
    output_subdir = os.path.join(output_dir, 'mfcc_separate')
    os.makedirs(output_subdir, exist_ok=True)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exporting MFCC (separate)"):
        sample_id = row['id']
        mfcc_path = row['mfcc']
        mfcc_data = np.load(mfcc_path)  # Shape: (n_mfcc, n_frames)
        
        # Transpose to (n_frames, n_mfcc) for easier MATLAB reading
        mfcc_transposed = mfcc_data.T
        
        # Create DataFrame with column names
        columns = [f'mfcc_{i+1}' for i in range(mfcc_transposed.shape[1])]
        mfcc_df = pd.DataFrame(mfcc_transposed, columns=columns)
        
        # Save to separate CSV
        output_path = os.path.join(output_subdir, f'{sample_id}.csv')
        mfcc_df.to_csv(output_path, index=False)
    
    # Create metadata CSV
    if include_metadata:
        metadata_cols = ['id', 'duration', 'wav', 'spk_id', 'label']
        metadata_df = df[metadata_cols].copy()
        metadata_df['mfcc_csv'] = metadata_df['id'].apply(
            lambda x: os.path.join(output_subdir, f'{x}.csv')
        )
        metadata_path = os.path.join(output_dir, 'mfcc_metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        print(f"✓ Metadata saved to: {metadata_path}")
    
    print(f"✓ Exported {len(df)} samples to: {output_subdir}")
    print(f"  Format: separate (one CSV per sample)")
    print(f"  Each file: (n_frames, n_mfcc)")
    
    return output_subdir


def export_for_matlab(input_csv, output_dir=None):
    """
    Convenience function to export MFCC in MATLAB-friendly format.
    
    Exports both statistics format and separate files for maximum flexibility.
    
    Args:
        input_csv (str): Path to emodb_mfcc_features.csv
        output_dir (str, optional): Output directory
    
    Returns:
        dict: Paths to exported files
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_csv), 'matlab_export')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Exporting MFCC features for MATLAB...")
    print("=" * 70)
    
    # Export statistics format (good for machine learning)
    stats_path = _export_stats_format(
        pd.read_csv(input_csv),
        output_dir,
        include_metadata=True
    )
    
    print()
    
    # Export separate format (good for detailed analysis)
    separate_path = _export_separate_format(
        pd.read_csv(input_csv),
        output_dir,
        include_metadata=True
    )
    
    print()
    print("=" * 70)
    print("✓ MATLAB export complete!")
    print(f"  Statistics CSV: {stats_path}")
    print(f"  Separate files: {separate_path}")
    
    return {
        'stats': stats_path,
        'separate': separate_path,
        'metadata': os.path.join(output_dir, 'mfcc_metadata.csv')
    }


if __name__ == '__main__':
    """Command-line interface for MFCC export."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export MFCC features to CSV format')
    parser.add_argument('--input', required=True, help='Input CSV with MFCC paths')
    parser.add_argument('--output-dir', help='Output directory (default: same as input)')
    parser.add_argument('--format', choices=['flat', 'stats', 'frames', 'separate', 'matlab'],
                       default='stats', help='Output format (default: stats)')
    parser.add_argument('--no-metadata', action='store_true', 
                       help='Exclude metadata columns')
    
    args = parser.parse_args()
    
    if args.format == 'matlab':
        export_for_matlab(args.input, args.output_dir)
    else:
        export_mfcc_to_csv(
            args.input,
            output_dir=args.output_dir,
            format_type=args.format,
            include_metadata=not args.no_metadata
        )
