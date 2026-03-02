"""Extract MFCC features from segmented audio and create a features CSV.

Parameters (matching existing LOSO data):
  sr=16000, n_mfcc=40, n_fft=396, hop_length=80,
  max_length=32000 samples (2 s), center=False
Output shape per file: (396, 40)  — (time_frames, n_mfcc)
Output CSV columns: ID, mfcc, spk_id, label, duration
"""
import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# MFCC defaults — keep in sync with config/ecapa_hparams.yaml
N_MFCC = 40
N_FFT = 396
HOP_LENGTH = 80
SR = 16000
MAX_SAMPLES = 32000  # pad / truncate to 2 s


def _extract(y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Return (time_frames, n_mfcc) array with exactly 396 frames."""
    # Pad or truncate
    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)))
    else:
        y = y[:MAX_SAMPLES]

    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
        hop_length=hop_length, htk=True, norm='ortho',
        center=False,   # ensures exactly 396 time frames
    )
    return mfccs.T  # (396, 40)


def extract_mfcc(input_csv, output_dir, csv_path):
    """Extract MFCC features for every row in input_csv.

    Args:
        input_csv (str): CSV from segment_audio() with columns ID, wav, spk_id, label.
        output_dir (str): Directory to write .npy feature files.
        csv_path (str): Path for the output features CSV.

    Returns:
        pd.DataFrame: Metadata including mfcc file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    df = pd.read_csv(input_csv)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCCs"):
        file_id = row['ID']
        wav_path = row['wav']

        y, _ = librosa.load(wav_path, sr=SR)
        mfccs = _extract(y)

        npy_path = os.path.join(output_dir, f"{file_id}.npy")
        np.save(npy_path, mfccs)

        rows.append({
            'ID': file_id,
            'mfcc': npy_path,
            'spk_id': row['spk_id'],
            'label': row['label'],
            'duration': row.get('duration', round(len(y) / SR, 4)),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_path, index=False)
    print(f"  {len(out_df)} MFCC files saved → {csv_path}")
    return out_df
