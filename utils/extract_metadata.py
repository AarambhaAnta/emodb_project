import os
import librosa
import yaml

def load_config(config):
    """Receive and store configuration"""
    global _config
    _config = config
    return _config

_config = None

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
    
    # Get Emotin Mapping Path
    emotion_mapping_path = os.path.join(_config['BASE_DIR'], _config['EMOTION_MAPPING_PATH'])
    with open(emotion_mapping_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Get Emotion and Emotion label
    emotion = config_data['emotion_mapping'].get(emotion_code, 'unknown')
    emotion_label = config_data['emotion_to_id'].get(emotion, 'unknown')

    return speaker, emotion_label

def extract_metadata(wav_path):
    try:
        # Load audio file
        y, sr = librosa.load(wav_path, sr=None)

        # Compute duration
        duration = librosa.get_duration(y=y, sr=sr)

        # set start to 0 and stop to duration * 16000
        start = 0
        stop = duration * 16000

        return duration, start, stop
    except Exception as e:
        print(f"Error processing {wav_path}: {str(e)}")
        return None, None, None
    
def get_metadata(wav_path):
    filename = os.path.basename(wav_path)

    speaker , emotion_label = parse_filename(filename=filename)
    duration, start, stop = extract_metadata(wav_path=wav_path)

    return filename, duration, start, stop, speaker, emotion_label
