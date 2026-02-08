"""Testing utilities for emotion recognition."""

from .emotion_testing import (
    EmotionCentroidTester,
    average_emotion_embeddings_for_all,
    average_emotion_embeddings_for_speaker,
    average_test_embeddings_for_all,
    average_test_embeddings_for_speaker,
    test_speaker_emotions,
    test_all_speakers
)
from .plda_scoring import (
    score_all_speakers_plda,
    score_speaker_plda
)

__all__ = [
    "EmotionCentroidTester",
    "average_emotion_embeddings_for_all",
    "average_emotion_embeddings_for_speaker",
    "average_test_embeddings_for_all",
    "average_test_embeddings_for_speaker",
    "score_all_speakers_plda",
    "score_speaker_plda",
    "test_speaker_emotions",
    "test_all_speakers"
]
