"""Training utilities for emotion recognition."""

from .ecapa_trainer import (
    EmotionBrain,
    prepare_datasets,
    train_speaker_model,
    train_all_speakers
)

__all__ = [
    'EmotionBrain',
    'prepare_datasets',
    'train_speaker_model',
    'train_all_speakers'
]
