"""Create LOSO train/validation/test splits from an MFCC features CSV.

For each speaker S:
  - test  : all rows where spk_id == S   (held-out speaker)
  - train : 80% of rows where spk_id != S (from remaining 9 speakers)
  - val   : 20% of rows where spk_id != S (from remaining 9 speakers)

No files are copied — CSVs reference the original .npy paths.

Output structure:
    output_dir/
    ├── speaker_03/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── speaker_08/
    │   └── ...
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_splits(mfcc_csv, output_dir, seed=42):
    """Build per-speaker LOSO splits from the MFCC features CSV.

    Args:
        mfcc_csv (str): CSV from extract_mfcc() with columns ID, mfcc, spk_id, label.
        output_dir (str): Root directory to write per-speaker split CSVs.
        seed (int): Random seed for train/val split reproducibility.

    Returns:
        dict: {speaker_id: {'train': path, 'val': path, 'test': path, ...}}
    """
    df = pd.read_csv(mfcc_csv)
    speakers = sorted(df['spk_id'].unique(), key=lambda x: str(x).zfill(2))
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for spk in tqdm(speakers, desc="Creating splits"):
        spk_str = str(spk).zfill(2)
        spk_dir = os.path.join(output_dir, f"speaker_{spk_str}")
        os.makedirs(spk_dir, exist_ok=True)

        test_df  = df[df['spk_id'] == spk].copy()
        train_val_df = df[df['spk_id'] != spk].copy()

        # Stratified 80/20 split of the remaining speakers
        train_idx, val_idx = train_test_split(
            train_val_df.index,
            test_size=0.2,
            random_state=seed,
            stratify=train_val_df['label'],
        )
        train_df = train_val_df.loc[train_idx]
        val_df   = train_val_df.loc[val_idx]

        train_path = os.path.join(spk_dir, 'train.csv')
        val_path = os.path.join(spk_dir, 'val.csv')
        test_path  = os.path.join(spk_dir, 'test.csv')

        train_df.reset_index(drop=True).to_csv(train_path, index=False)
        val_df.reset_index(drop=True).to_csv(val_path, index=False)
        test_df.reset_index(drop=True).to_csv(test_path, index=False)

        results[spk_str] = {
            'train': train_path,
            'val': val_path,
            'test':  test_path,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples':  len(test_df),
        }

    print(f"  Splits created for {len(results)} speakers in {output_dir}")
    return results
