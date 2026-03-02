"""Training utilities for emotion recognition."""

from .ecapa_trainer import EmotionBrain, train_speaker_model, train_all_speakers
from .plda_trainer import PLDAModel, train_speaker_plda, train_all_speakers_plda

__all__ = [
    'EmotionBrain',
    'train_speaker_model',
    'train_all_speakers',
    'PLDAModel',
    'train_speaker_plda',
    'train_all_speakers_plda',
]
