"""
Metadata extraction and audio segmentation utilities for EmoDb dataset.

This module provides functions for:
- Parsing EmoDb filenames to extract speaker and emotion information
- Extracting audio metadata (duration, samples, etc.)
- Segmenting audio files into equal parts based on duration
"""
import os
import librosa
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from ..extract_config import get_config


def parse_filename(filename, emotion_mapping=None):
    """
    Parse EmoDb filename to extract speaker ID and emotion label.
    
    EmoDb filename format: [speaker_id][sentence][emotion][version].wav
    Example: 03a01Fa.wav
        - Speaker: 03
        - Sentence: a01
        - Emotion: F (Freude/happiness)
        - Version: a
    
    Args:
        filename (str): Audio filename (with or without .wav extension)
        emotion_mapping (dict, optional): Emotion code to numeric mapping.
            If None, loads from config.
    
    Returns:
        tuple: (file_id, speaker_id, emotion_label)
            - file_id (str): Filename without extension
            - speaker_id (str): Two-digit speaker ID (e.g., '03')
            - emotion_label (int): Numeric emotion label
    
    Example:
        >>> parse_filename('03a01Fa.wav')
        ('03a01Fa', '03', 0)  # 0 = happiness
    """
    if emotion_mapping is None:
        config = get_config()
        emotion_mapping = config.get('EMOTION_MAPPING', {})
    
    # Remove .wav extension if present
    file_id = filename.replace('.wav', '')
    
    # Extract speaker ID (first 2 characters)
    speaker_id = file_id[:2]
    
    # Extract emotion code (second to last character)
    emotion_code = file_id[-2].upper()
    
    # Map emotion code to numeric label
    emotion_label = emotion_mapping.get(emotion_code, -1)
    
    return file_id, speaker_id, emotion_label


def extract_metadata(wav_path, sr=None):
    """
    Extract audio metadata from WAV file.
    
    Args:
        wav_path (str): Path to WAV file
        sr (int, optional): Target sampling rate. If None, loads from config.
    
    Returns:
        tuple: (duration, start_sample, stop_sample)
            - duration (float): Audio duration in seconds
            - start_sample (int): Always 0
            - stop_sample (int): Total number of samples
            
    Raises:
        Exception: If audio file cannot be loaded
    """
    if sr is None:
        config = get_config()
        sr = config.get('AUDIO', {}).get('SAMPLING_RATE', 16000)
    
    try:
        # Load audio file
        y, _ = librosa.load(wav_path, sr=sr)
        
        # Compute duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Start at 0, stop at total samples
        start_sample = 0
        stop_sample = int(duration * sr)
        
        return duration, start_sample, stop_sample
        
    except Exception as e:
        print(f"Error processing {wav_path}: {str(e)}")
        return None, None, None


def get_metadata(wav_path, sr=None, emotion_mapping=None):
    """
    Extract complete metadata for a WAV file.
    
    Combines filename parsing and audio metadata extraction.
    
    Args:
        wav_path (str): Path to WAV file
        sr (int, optional): Target sampling rate. If None, loads from config.
        emotion_mapping (dict, optional): Emotion mapping. If None, loads from config.
    
    Returns:
        tuple: (file_id, duration, wav_path, start_sample, stop_sample, 
                speaker_id, emotion_label)
    """
    filename = os.path.basename(wav_path)
    
    # Parse filename
    file_id, speaker_id, emotion_label = parse_filename(filename, emotion_mapping)
    
    # Extract audio metadata
    duration, start_sample, stop_sample = extract_metadata(wav_path, sr)
    
    return file_id, duration, wav_path, start_sample, stop_sample, speaker_id, emotion_label


def extract_metadata_from_folder(folder_path, sr=None, emotion_mapping=None):
    """
    Extract metadata from all WAV files in a folder.
    
    Args:
        folder_path (str): Path to folder containing WAV files
        sr (int, optional): Target sampling rate. If None, loads from config.
        emotion_mapping (dict, optional): Emotion mapping. If None, loads from config.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - id: File ID without extension
            - duration: Audio duration in seconds
            - wav: Full path to WAV file
            - start: Start sample (always 0)
            - stop: Stop sample
            - spk_id: Speaker ID
            - label: Numeric emotion label
    """
    folder = Path(folder_path)
    wav_files = list(folder.glob('*.wav'))
    
    if len(wav_files) == 0:
        print(f"Warning: No WAV files found in {folder_path}")
        return pd.DataFrame()
    
    metadata_list = []
    
    for wav_file in wav_files:
        file_id, duration, path, start, stop, speaker_id, emotion_label = \
            get_metadata(str(wav_file), sr, emotion_mapping)
        
        metadata_list.append({
            'id': file_id,
            'duration': duration,
            'wav': path,
            'start': start,
            'stop': stop,
            'spk_id': speaker_id,
            'label': emotion_label
        })
    
    df = pd.DataFrame(metadata_list)
    return df


# -------------------------------------------------
# Audio Segmentation Functions
# -------------------------------------------------


def get_num_segments(duration, config=None):
    """
    Determine number of segments based on audio duration.
    
    Default segmentation rules:
    - duration >= 6s: 4 segments
    - duration >= 4s: 3 segments
    - duration >= 2s: 2 segments
    - duration < 2s: 1 segment (no segmentation)
    
    Args:
        duration (float): Audio duration in seconds
        config (dict, optional): Configuration dictionary with SEGMENTATION settings.
            If None, uses default thresholds.
    
    Returns:
        int: Number of segments (1-4)
    """
    if config is None:
        config = get_config()
    
    seg_config = config.get('SEGMENTATION', {})
    threshold_2 = seg_config.get('MIN_DURATION_2_SEGMENTS', 2)
    threshold_3 = seg_config.get('MIN_DURATION_3_SEGMENTS', 4)
    threshold_4 = seg_config.get('MIN_DURATION_4_SEGMENTS', 6)
    
    if duration >= threshold_4:
        return 4
    elif duration >= threshold_3:
        return 3
    elif duration >= threshold_2:
        return 2
    else:
        return 1


def get_segment(wav_path, output_dir, file_id, num_segments, sr=None):
    """
    Segment audio file into equal parts and save to output directory.
    
    Args:
        wav_path (str): Path to input WAV file
        output_dir (str): Directory to save segmented files
        file_id (str): Base file ID for naming segments
        num_segments (int): Number of segments to create
        sr (int, optional): Target sampling rate. If None, loads from config.
    
    Returns:
        list: List of dicts with segment information:
            - id: Segment ID (file_id_1, file_id_2, etc.)
            - duration: Segment duration in seconds
            - wav: Path to segment file
            - start: Start sample (always 0 for segments)
            - stop: Stop sample (length of segment)
    """
    if sr is None:
        config = get_config()
        sr = config.get('AUDIO', {}).get('SAMPLING_RATE', 16000)
    
    # Load audio
    y, _ = librosa.load(wav_path, sr=sr)
    total_samples = len(y)
    total_duration = total_samples / sr
    
    segment_info = []
    
    if num_segments == 1:
        # No segmentation needed, just copy with new naming
        output_path = os.path.join(output_dir, f"{file_id}_1.wav")
        sf.write(output_path, y, sr)
        
        segment_info.append({
            'id': f"{file_id}_1",
            'duration': total_duration,
            'wav': output_path,
            'start': 0,
            'stop': total_samples,
        })
    else:
        # Segment into equal parts
        samples_per_segment = total_samples // num_segments
        
        for i in range(num_segments):
            start_sample = i * samples_per_segment
            
            # Last segment takes remaining samples
            if i == num_segments - 1:
                stop_sample = total_samples
            else:
                stop_sample = (i + 1) * samples_per_segment
            
            # Extract segment
            segment = y[start_sample:stop_sample]
            segment_duration = len(segment) / sr
            
            # Save segment
            segment_filename = f"{file_id}_{i + 1}.wav"
            output_path = os.path.join(output_dir, segment_filename)
            sf.write(output_path, segment, sr)
            
            segment_info.append({
                'id': f"{file_id}_{i + 1}",
                'duration': segment_duration,
                'wav': output_path,
                'start': 0,
                'stop': len(segment),
            })
    
    return segment_info


def extract_segment_from_folder(output_dir, input_csv, sr=None):
    """
    Segment all audio files listed in input CSV.
    
    Reads metadata CSV, segments each audio file based on duration,
    and creates a new CSV with segment metadata.
    
    Args:
        output_dir (str): Directory to save segmented audio files
        input_csv (str): Path to input CSV with columns: id, duration, wav, 
                        spk_id, label
        sr (int, optional): Target sampling rate. If None, loads from config.
    
    Returns:
        pd.DataFrame: DataFrame with segmented audio metadata:
            - id: Segment ID
            - duration: Segment duration
            - wav: Path to segment file
            - start: Start sample (0)
            - stop: Stop sample
            - spk_id: Speaker ID
            - label: Emotion label
    """
    if sr is None:
        config = get_config()
        sr = config.get('AUDIO', {}).get('SAMPLING_RATE', 16000)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(input_csv)
    
    segmented_metadata = []
    
    # Process each file
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting audio files"):
        file_id = row['id']
        duration = row['duration']
        wav_path = row['wav']
        speaker_id = row['spk_id']
        label = row['label']
        
        # Determine number of segments
        num_segments = get_num_segments(duration)
        
        # Get segments
        segments = get_segment(wav_path, output_dir, file_id, num_segments, sr)
        
        # Add speaker and emotion label to each segment
        for segment in segments:
            segment['spk_id'] = speaker_id
            segment['label'] = label
            segmented_metadata.append(segment)
    
    # Create DataFrame from segmented metadata
    segmented_df = pd.DataFrame(segmented_metadata)
    
    # Sort by id
    segmented_df = segmented_df.sort_values('id').reset_index(drop=True)
    
    return segmented_df
