"""
MFCC feature extraction module for EmoDb audio files.
Extracts MFCC features with 40 dimensions x time_frames matching MATLAB implementation.
"""
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.extract_config import get_config


def extract_mfcc_features(wav_path, sr=16000, n_mfcc=40, n_fft=400, hop_length=80, 
                          max_length=32000, normalize=True):
    """
    Extract MFCC features from audio file matching MATLAB implementation.
    
    Args:
        wav_path (str): Path to the audio file
        sr (int): Sampling rate (default 16000)
        n_mfcc (int): Number of MFCC coefficients (default 40)
        n_fft (int): FFT window size - 25ms at 16kHz = 400 samples
        hop_length (int): Hop length - 5ms at 16kHz = 80 samples
        max_length (int): Maximum audio length in samples (default 32000)
        normalize (bool): Normalize audio signal (default True)
    
    Returns:
        np.ndarray: MFCC features with shape (40, time_frames)
    """
    # Load audio
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    
    # Normalize
    if normalize and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    # Pad or truncate to fixed length
    if len(y) > max_length:
        y = y[:max_length]
    else:
        padding = max_length - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=40,
        fmin=0.0,
        fmax=sr/2.0,
        htk=True,
        norm='ortho'
    )
    
    return mfccs


def extract_mfcc_from_dataset(input_csv=None, output_csv=None, output_dir=None):
    """
    Extract MFCC features from all audio files in the dataset.
    
    Args:
        input_csv (str, optional): Path to input CSV with audio metadata
        output_csv (str, optional): Path to output CSV with MFCC features
        output_dir (str, optional): Directory to save .npy files
    
    Returns:
        pd.DataFrame: DataFrame with MFCC feature paths and metadata
    """
    config_path = '/Users/adityakumar/Developer/Projects/emodb_project/config/emodb_config.yaml'
    config = get_config(config_path)
    
    if input_csv is None:
        input_csv = os.path.join(config['BASE_DIR'], config['CSV_DIR'], 'segmented_metadata.csv')
    if output_csv is None:
        output_csv = os.path.join(config['BASE_DIR'], config['MFCC']['OUTPUT_CSV'])
    if output_dir is None:
        output_dir = os.path.join(config['BASE_DIR'], config['MFCC']['OUTPUT_DIR'])
    
    sr = config['SAMPLING_RATE']
    n_mfcc = 40
    n_fft = int(25e-3 * sr)
    hop_length = int(5e-3 * sr)
    max_length = 32000
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    
    features_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCC"):
        file_id = row['id']
        wav_path = row['wav']
        
        mfcc_features = extract_mfcc_features(
            wav_path, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, 
            hop_length=hop_length, max_length=max_length
        )
        
        npy_filename = f"{file_id}.npy"
        npy_path = os.path.join(output_dir, npy_filename)
        np.save(npy_path, mfcc_features)
        
        entry = {
            'id': file_id,
            'duration': row['duration'],
            'wav': wav_path,
            'mfcc': npy_path,
            'mfcc_shape': f"{mfcc_features.shape[0]}x{mfcc_features.shape[1]}",
            'start': row['start'],
            'stop': row['stop'],
            'spk_id': row['spk_id'],
            'label': row['label']
        }
        features_data.append(entry)
    
    features_df = pd.DataFrame(features_data)
    features_df.to_csv(output_csv, index=False)
    
    return features_df
