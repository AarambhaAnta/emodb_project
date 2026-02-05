"""
EmoDb utilities package for emotion analysis.
"""

from .extract_config import get_config
from .extract_metadata import parse_filename, extract_metadata, get_metadata,extract_metadata_from_folder, EMOTION_MAPPING
from .create_csv import create_csv

__all__ = [
    'get_config',
    'parse_filename',
    'extract_metadata',
    'get_metadata',
    'extract_metadata_from_folder'
    'EMOTION_MAPPING',
    'create_csv'
]

__version__ = '0.1.0'
