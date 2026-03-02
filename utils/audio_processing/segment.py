"""Segment EmoDb audio files and create metadata CSV.

EmoDb filename format: {speaker_id}{sentence}{emotion}{version}.wav
  e.g. 03a01Fa.wav → speaker=03, emotion=F(0), label=0

Emotion codes: W=2, L=4, E=5, A=3, F=0, T=6, N=1
Segmentation rules: <2s→1seg, 2-4s→2segs, 4-6s→3segs, ≥6s→4segs
Output CSV columns: ID, duration, wav, spk_id, label
"""
import os
import librosa
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

EMOTION_MAP = {'W': 2, 'L': 4, 'E': 5, 'A': 3, 'F': 0, 'T': 6, 'N': 1}
SR = 16000


def _n_segments(duration_s):
    if duration_s >= 6: return 4
    if duration_s >= 4: return 3
    if duration_s >= 2: return 2
    return 1


def segment_audio(raw_dir, output_dir, csv_path):
    """Segment all WAV files in raw_dir and save a metadata CSV.

    Args:
        raw_dir (str): Directory containing raw EmoDb .wav files.
        output_dir (str): Directory to write segmented .wav files.
        csv_path (str): Path for the output CSV file.

    Returns:
        pd.DataFrame: Metadata for all created segments.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    rows = []
    wav_files = sorted(Path(raw_dir).glob('*.wav'))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {raw_dir}")

    for wav_file in tqdm(wav_files, desc="Segmenting audio"):
        stem = wav_file.stem
        spk_id = stem[:2]
        emotion_char = stem[-2].upper()
        label = EMOTION_MAP.get(emotion_char, -1)
        if label == -1:
            continue  # skip files with unrecognised emotion codes

        y, _ = librosa.load(str(wav_file), sr=SR)
        duration = len(y) / SR
        n = _n_segments(duration)
        seg_len = len(y) // n

        for i in range(n):
            start = i * seg_len
            stop = len(y) if i == n - 1 else (i + 1) * seg_len
            seg_id = f"{stem}_{i + 1}"
            out_path = os.path.join(output_dir, f"{seg_id}.wav")
            sf.write(out_path, y[start:stop], SR)
            rows.append({
                'ID': seg_id,
                'duration': round((stop - start) / SR, 4),
                'wav': out_path,
                'spk_id': spk_id,
                'label': label,
            })

    df = pd.DataFrame(rows).sort_values('ID').reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"  {len(df)} segments saved → {csv_path}")
    return df
