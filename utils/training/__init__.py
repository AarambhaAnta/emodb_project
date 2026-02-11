"""Training utilities for emotion recognition."""

from .ecapa_trainer import (
    EmotionBrain,
    prepare_datasets,
    prepare_train_dataset,
    train_speaker_model,
    train_all_speakers,
    train_speaker_model_other,
    train_all_speakers_other,
    train_speaker_model_firstpart,
    train_all_speakers_firstpart,
    train_speaker_model_other_firstpart,
    train_all_speakers_other_firstpart
)

# from .lda_trainer import (
#     LDAModel,
#     load_features_from_csv,
#     train_speaker_lda,
#     train_all_speakers_lda
# )

from .plda_trainer import (
    PLDAModel,
    train_speaker_plda,
    train_all_speakers_plda,
    train_speaker_plda_firstpart,
    train_all_speakers_plda_firstpart
)

__all__ = [
    # ECAPA-TDNN
    'EmotionBrain',
    'prepare_datasets',
    'prepare_train_dataset',
    'train_speaker_model',
    'train_all_speakers',
    'train_speaker_model_other',
    'train_all_speakers_other',
    'train_speaker_model_firstpart',
    'train_all_speakers_firstpart',
    'train_speaker_model_other_firstpart',
    'train_all_speakers_other_firstpart',
    # LDA
    # 'LDAModel',
    # 'load_features_from_csv',
    # 'train_speaker_lda',
    # 'train_all_speakers_lda',
    # PLDA
    'PLDAModel',
    'train_speaker_plda',
    'train_all_speakers_plda',
    'train_speaker_plda_firstpart',
    'train_all_speakers_plda_firstpart'
]
