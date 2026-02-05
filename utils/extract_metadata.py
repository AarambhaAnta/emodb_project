import os
import librosa

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

def extract_metadata(wav_path):
    """
    Extract audio features from wav file.
    Returns duration, start time (0), and stop time (duration)
    """
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

    name, speaker , emotion_label = parse_filename(filename=filename)
    duration, start, stop = extract_metadata(wav_path=wav_path)

    return name, duration, wav_path, start, stop, speaker, emotion_label
