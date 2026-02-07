"""
MFCC feature extraction module for EmoDb audio files.

Extracts MFCC features with 40 dimensions × time_frames matching MATLAB implementation.
"""
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from ..extract_config import get_config, get_path


def extract_mfcc_features(wav_path, config=None, sr=None, n_mfcc=None, n_fft=None,  
                          hop_length=None, n_mels=None, max_length=None, 
                          normalize=None, fmin=None, fmax=None, htk=None, norm=None):
    """
    Extract MFCC features from audio file matching MATLAB implementation.
    
    Args:
        wav_path (str): Path to the audio file
        config (dict, optional): Configuration dictionary. If None, loads from default config.
        sr (int, optional): Sampling rate. If None, uses config or 16000.
        n_mfcc (int, optional): Number of MFCC coefficients. If None, uses config or 40.
        n_fft (int, optional): FFT window size. If None, uses config or 400 (25ms at 16kHz).
        hop_length (int, optional): Hop length. If None, uses config or 80 (5ms at 16kHz).
        n_mels (int, optional): Number of mel bands. If None, uses config or 40.
        max_length (int, optional): Maximum audio length in samples. If None, uses config or 32000.
        normalize (bool, optional): Normalize audio signal. If None, uses config or True.
        fmin (float, optional): Minimum frequency. If None, uses config or 0.0.
        fmax (float, optional): Maximum frequency. If None, uses config or sr/2.
        htk (bool, optional): Use HTK formula. If None, uses config or True.
        norm (str, optional): Normalization method. If None, uses config or 'ortho'.
    
    Returns:
        np.ndarray: MFCC features with shape (n_mfcc, time_frames)
        
    Example:
        >>> mfcc = extract_mfcc_features('audio.wav')
        >>> print(mfcc.shape)
        (40, 401)
    """
    # Load config if not provided
    if config is None:
        config = get_config()
    
    # Get audio and MFCC parameters from config if not provided
    audio_config = config.get('AUDIO', {})
    mfcc_config = config.get('MFCC', {})
    
    sr = sr or audio_config.get('SAMPLING_RATE', 16000)
    n_mfcc = n_mfcc or mfcc_config.get('N_MFCC', 40)
    n_fft = n_fft or mfcc_config.get('N_FFT', 400)
    hop_length = hop_length or mfcc_config.get('HOP_LENGTH', 80)
    n_mels = n_mels or mfcc_config.get('N_MELS', 40)
    max_length = max_length or mfcc_config.get('MAX_LENGTH', 32000)
    normalize = normalize if normalize is not None else mfcc_config.get('NORMALIZE', True)
    fmin = fmin if fmin is not None else mfcc_config.get('FMIN', 0.0)
    fmax = fmax or mfcc_config.get('FMAX', sr / 2.0)
    htk = htk if htk is not None else mfcc_config.get('HTK', True)
    norm = norm or mfcc_config.get('NORM', 'ortho')
    
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
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk,
        norm=norm
    )
    
    return mfccs


def extract_mfcc_from_dataset(input_csv=None, output_csv=None, output_dir=None, config=None):
    """
    Extract MFCC features from all audio files in the dataset.
    
    Reads metadata CSV, extracts MFCC features for each audio file,
    and saves features as .npy files with updated metadata CSV.
    
    Args:
        input_csv (str, optional): Path to input CSV with audio metadata.
            If None, uses config path or 'data/csv/segmented_metadata.csv'.
        output_csv (str, optional): Path to output CSV with MFCC features.
            If None, uses config path or 'data/csv/emodb_mfcc_features.csv'.
        output_dir (str, optional): Directory to save .npy files.
            If None, uses config path or 'data/processed/features/mfcc'.
        config (dict, optional): Configuration dictionary. If None, loads from default.
    
    Returns:
        pd.DataFrame: DataFrame with MFCC feature paths and metadata with columns:
            - id: File ID
            - duration: Audio duration
            - wav: Path to WAV file
            - mfcc: Path to MFCC .npy file
            - mfcc_shape: Shape of MFCC array (e.g., '40x401')
            - start, stop: Sample indices
            - spk_id: Speaker ID
            - label: Emotion label
            
    Example:
        >>> df = extract_mfcc_from_dataset()
        >>> print(f"Extracted {len(df)} MFCC features")
    """
    # Load config if not provided
    if config is None:
        config = get_config()
    
    # Get paths from config or use defaults
    base_dir = config.get('BASE_DIR', os.getcwd())
    
    if input_csv is None:
        csv_dir = config.get('PATHS', {}).get('CSV', 'data/csv')
        input_csv = get_path(config, csv_dir, 'segmented_metadata.csv')
    
    if output_csv is None:
        output_csv_path = config.get('MFCC', {}).get('OUTPUT_CSV', 
                                                     'data/csv/emodb_mfcc_features.csv')
        output_csv = get_path(config, output_csv_path)
    
    if output_dir is None:
        output_dir_path = config.get('MFCC', {}).get('OUTPUT_DIR', 
                                                     'data/processed/features/mfcc')
        output_dir = get_path(config, output_dir_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure output CSV directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(input_csv)
    
    features_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCC"):
        file_id = row['id']
        wav_path = row['wav']
        
        # Extract MFCC features
        mfcc_features = extract_mfcc_features(wav_path, config=config)
        
        # Save as .npy file
        npy_filename = f"{file_id}.npy"
        npy_path = os.path.join(output_dir, npy_filename)
        np.save(npy_path, mfcc_features)
        
        # Create entry with MFCC info
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
    
    # Create DataFrame and save
    features_df = pd.DataFrame(features_data)
    features_df.to_csv(output_csv, index=False)
    
    print(f"\nExtracted {len(features_df)} MFCC features")
    print(f"Features saved to: {output_dir}")
    print(f"CSV saved to: {output_csv}")
    
    return features_df
