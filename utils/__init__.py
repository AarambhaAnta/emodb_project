"""
EmoDb utilities package for emotion analysis.

This package provides utilities for:
- Audio processing and metadata extraction
- Feature extraction (MFCC)
- LOSO cross-validation splits
- Model training with ECAPA-TDNN

Main entry points:
- get_config(): Load configuration
- Submodules: audio_processing, features_extraction, training
"""

from .extract_config import get_config

# Expose submodules for organized access
from . import audio_processing
from . import features_extraction  
from . import training
from . import testing

__all__ = [
    'get_config',
    'audio_processing',
    'features_extraction',
    'training',
    'testing'
]

__version__ = '0.1.0'
