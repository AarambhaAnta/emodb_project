"""Create LOSO splits from an MFCC features CSV.

Split strategy:
  For each test speaker S (10 folds total):
    - test  : all segments of speaker S        (1 speaker, fully held-out)
    - train : all segments from the other 9 speakers
    - val   : identical to train               (same 9-speaker pool)

Train and val are the same data. The model trains and monitors loss on the
full 9-speaker pool; the test speaker is never seen during training.

Output structure:
    output_dir/
    ├── speaker_03/
    │   ├── train.csv       # all 9 non-test speakers
    │   ├── val.csv         # identical to train.csv
    │   └── test.csv        # speaker 03 only
    ├── speaker_08/
    │   └── ...
"""
import os

import pandas as pd
from tqdm import tqdm


def create_splits(mfcc_csv, output_dir, seed=42):
    """Build LOSO splits: 9-speaker pool used as both train and val, 1 speaker test.

    Args:
        mfcc_csv (str): CSV from extract_mfcc() with columns ID, mfcc, spk_id, label.
        output_dir (str): Root directory to write per-speaker split CSVs.
        seed (int): Kept for API compatibility; splits are deterministic.

    Returns:
        dict: {speaker_id: {'train': path, 'val': path, 'test': path, ...}}
    """
    df = pd.read_csv(mfcc_csv)

    # Normalise spk_id to zero-padded strings so '3' and '03' both become '03'
    df['spk_id'] = df['spk_id'].apply(lambda x: str(x).zfill(2))

    speakers = sorted(df['spk_id'].unique())
    if len(speakers) < 2:
        raise ValueError(f"Need at least 2 speakers for LOSO (got {len(speakers)})")

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for spk in tqdm(speakers, desc="Creating LOSO splits"):
        spk_str = str(spk).zfill(2)
        spk_dir = os.path.join(output_dir, f"speaker_{spk_str}")
        os.makedirs(spk_dir, exist_ok=True)

        test_df = df[df['spk_id'] == spk].copy()
        pool_df = df[df['spk_id'] != spk].copy()

        # Train and val are the same: the full 9-speaker pool
        train_df = pool_df.reset_index(drop=True)
        val_df   = pool_df.reset_index(drop=True)

        train_path = os.path.join(spk_dir, 'train.csv')
        val_path   = os.path.join(spk_dir, 'val.csv')
        test_path  = os.path.join(spk_dir, 'test.csv')

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path,     index=False)
        test_df.reset_index(drop=True).to_csv(test_path, index=False)

        pool_speakers = sorted(pool_df['spk_id'].unique().tolist())
        print(
            f"  speaker_{spk_str}: "
            f"test=[{spk_str}]  pool={pool_speakers} "
            f"| {len(train_df)} train+val / {len(test_df)} test"
        )

        results[spk_str] = {
            'train':         train_path,
            'val':           val_path,
            'test':          test_path,
            'pool_speakers': pool_speakers,
            'test_speaker':  spk_str,
            'train_samples': len(train_df),
            'val_samples':   len(val_df),
            'test_samples':  len(test_df),
        }

    print(f"  Splits created for {len(results)} speakers in {output_dir}")
    return results
