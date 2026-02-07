from cProfile import label
from encodings.punycode import T
import os
import librosa
import pandas as pd
from pathlib import Path
import soundfile as sf
import tqdm

# -------------------------------------------------
# Step 1: Metadata extraction from raw file.
# -------------------------------------------------

# sampling rate
SR = 16000
# Emotion mapping for EmoDb to numeric labels
EMOTION_MAPPING = {
    'W': 2,  # Wut (anger)
    'L': 4,  # Langeweile (boredom)
    'E': 5,  # Ekel (disgust)
    'A': 3,  # Angst (fear/anxiety)
    'F': 0,  # Freude (happiness)
    'T': 6,  # Trauer (sadness)
    'N': 1   # Neutral
}

def parse_filename(filename):
    """
    Parse EmoDb filename to extract speaker and emotion.
    Format: [speaker_id][sentence][emotion][version].wav = 0)
    """
    # Remove .wav extension
    name = filename.replace('.wav','')

    # Extract speaker ID (first 2 digits) - already has leading zero
    speaker = name[:2]

    # Extract emotion (second to last character)
    emotion_code = name[-2].upper()
    
    # Extract emotion (second to last character)
    emotion_code = name[-2]
    emotion_label = EMOTION_MAPPING.get(emotion_code, -1)

    return name, speaker, emotion_label

def extract_metadata(wav_path, sr=SR):
    """
    Extract audio features from wav file.
    Returns duration, start time (0), and stop time (duration)
    """
    try:
        # Load audio file
        # ! check with prof sr = None
        y, _ = librosa.load(wav_path, sr=sr)

        # Compute duration
        duration = librosa.get_duration(y=y, sr=sr)

        # set start to 0 and stop to duration * 16000
        start = 0
        stop = duration * sr

        return duration, start, stop
    except Exception as e:
        print(f"Error processing {wav_path}: {str(e)}")
        return None, None, None
    
def get_metadata(wav_path, sr=SR):
    filename = os.path.basename(wav_path)

    name, speaker , emotion_label = parse_filename(filename=filename)
    duration, start, stop = extract_metadata(wav_path=wav_path, sr=sr)

    return name, duration, wav_path, start, stop, speaker, emotion_label

def extract_metadata_from_folder(folder_path, sr = SR):
    """
    Extract metadata from all .wav files in a folder.
    
    Args:
        folder_path: Path to folder containing .wav files
        
    Returns:
        pandas DataFrame with columns: name, duration, path, start, stop, speaker, emotion
    """
    folder = Path(folder_path)
    wav_files = list(folder.glob('*.wav'))
    
    metadata_list = []
    
    for wav_file in wav_files:
        name, duration, path, start, stop, speaker, emotion_label = get_metadata(str(wav_file), sr=sr)
        metadata_list.append({
            'id': name,
            'duration': duration,
            'wav': path,
            'start': start,
            'stop': stop,
            'spk_id': speaker,
            'label': emotion_label
        })
    
    df = pd.DataFrame(metadata_list)
    return df

# -------------------------------------------------
# Step 2: Segment the raw file in parts
# -------------------------------------------------

def get_num_segments(duration):
    """
    Determine number of segments based on duration.
    - duration >= 6: 4 parts
    - duration >= 4: 3 parts
    - duration >= 2: 2 parts
    - duration < 2: 1 part (no segmentation)
    """
    if duration >=6:
        return 4
    elif duration>=4:
        return 3
    elif duration>=2:
        return 2
    else:
        return 1
    
def get_segment(wav_path, output_dir, file_id, num_segments, sr=SR):
    """
    Segment audio file into equal parts and save to output directory.
    Returns list of segment info (id, duration, wav, start, stop)
    """
    # Load audio
    y, _ = librosa.load(path=wav_path, sr=sr)
    total_samples = len(y)
    total_duration = total_samples/sr;

    segment_info = []

    if num_segments == 1:
        # No segmentation needed, just copy
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
        # segment into equal parts
        samples_per_segment = total_samples//num_segments

        for i in range(num_segments):
            start_sample = i*samples_per_segment

            # Last segment takes remaining samples
            if i==num_segments-1:
                stop_sample = total_samples
            else:
                stop_sample = (i+1)*samples_per_segment

            # Extract segment
            segment = y[start_sample:stop_sample]
            segment_duration = len(segment)/sr

            # save segment
            segment_filename = f"{file_id}_{i+1}.wav"
            output_path = os.path.join(output_dir, segment_filename)
            sf.write(output_path, segment, sr)

            segment_info.append({
                'id': f"{file_id}_{i+1}",
                'duration': segment_duration,
                'wav': output_path,
                'start': 0,
                'stop': len(segment),
            })

    return segment_info

def extract_segment_from_folder(output_dir, intput_csv, sr=SR):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    df = pd.read_csv(intput_csv)

    segmented_metadata = []

    # Process each file
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Segmenting audio files"):
        file_id = row['id']
        duration = row['duration']
        wav_path = row['wav']
        speaker = row['spk_id']
        label = row['label']
        
        # Determine number of segments
        num_segments = get_num_segments(duration)
        
        # Get segments
        segments = get_segment(wav_path, output_dir, file_id, num_segments, sr=sr)
        
        # Add speaker and emotion label to each segment
        for segment in segments:
            segment['spk_id'] = speaker
            segment['label'] = label
            segmented_metadata.append(segment)
    
    # Create DataFrame from segmented metadata
    segmented_df = pd.DataFrame(segmented_metadata)
    
    # Sort by id
    segmented_df = segmented_df.sort_values('id').reset_index(drop=True)
    
    return segmented_df
