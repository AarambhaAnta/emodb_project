"""
EmoDb utilities package for emotion analysis.
"""

from .extract_config import get_config
from .extract_metadata import parse_filename, extract_metadata, EMOTION_MAPPING
from .extract_features import extract_features

__all__ = [
    'get_config',
    'parse_filename',
    'extract_metadata',
    'extract_features',
    'EMOTION_MAPPING',
]

__version__ = '0.1.0'
