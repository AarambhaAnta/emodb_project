"""Training utilities for emotion recognition."""

from .ecapa_trainer import (
    EmotionBrain,
    prepare_datasets,
    train_speaker_model,
    train_all_speakers
)

from .lda_trainer import (
    LDAModel,
    load_features_from_csv,
    train_speaker_lda,
    train_all_speakers_lda
)

from .plda_trainer import (
    PLDAModel,
    train_speaker_plda,
    train_all_speakers_plda
)

__all__ = [
    # ECAPA-TDNN
    'EmotionBrain',
    'prepare_datasets',
    'train_speaker_model',
    'train_all_speakers',
    # LDA
    'LDAModel',
    'load_features_from_csv',
    'train_speaker_lda',
    'train_all_speakers_lda',
    # PLDA
    'PLDAModel',
    'train_speaker_plda',
    'train_all_speakers_plda'
]
