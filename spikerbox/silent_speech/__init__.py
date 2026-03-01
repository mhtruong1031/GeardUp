"""
Silent Speech Decoder: subvocal EMG → small vocabulary (Yes, No, Help, Water).
"""

from .config import (
    VOCAB,
    NUM_CLASSES,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    SAMPLE_RATE,
    WINDOW_SEC,
    WINDOW_LEN,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
)
from .model import SilentSpeechCNN

__all__ = [
    "VOCAB",
    "NUM_CLASSES",
    "LABEL_TO_IDX",
    "IDX_TO_LABEL",
    "SAMPLE_RATE",
    "WINDOW_SEC",
    "WINDOW_LEN",
    "DEFAULT_DATA_DIR",
    "DEFAULT_MODEL_PATH",
    "SilentSpeechCNN",
]
