"""
Configuration for Silent Speech Decoder: vocabulary, timing, and paths.
"""

import os

# Vocabulary for subvocal commands (4 classes).
VOCAB = ["yes", "no", "help", "water", "fuck"]
NUM_CLASSES = len(VOCAB)

# Label string -> index.
LABEL_TO_IDX = {w: i for i, w in enumerate(VOCAB)}
IDX_TO_LABEL = {i: w for i, w in enumerate(VOCAB)}

# Sampling: Human SpikerBox 2-channel mode.
SAMPLE_RATE = 5000.0
SILENT_SPEECH_CHANNELS = 2

# Recording window for one utterance (seconds).
WINDOW_SEC = 1.5
WINDOW_LEN = int(SAMPLE_RATE * WINDOW_SEC)  # samples per channel

# Default paths (relative to project root or cwd).
DEFAULT_DATA_DIR = "data/silent_speech"
DEFAULT_MODEL_DIR = "models/silent_speech"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "silent_speech_cnn.pt")

# Fallback DC offset if no calibration is used (e.g. mid-scale 14-bit).
EMG_OFFSET = 8250

# Calibration: 60s rest session to compute per-channel baseline (mean).
CALIBRATION_DURATION_SEC = 60.0
CALIBRATION_FILENAME = "calibration.npz"


def load_calibration(dir_or_path: str):
    """
    Load baseline mean from a calibration file. Returns array of shape (SILENT_SPEECH_CHANNELS,)
    to subtract per channel, or None if not found. Pass a directory or full path to calibration.npz.
    """
    import numpy as np
    path = dir_or_path if dir_or_path.endswith(".npz") else os.path.join(dir_or_path, CALIBRATION_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        mean = data["mean"]
        return np.asarray(mean, dtype=np.float64)
    except Exception:
        return None


# Preprocessing (EMG bandpass).
BANDPASS_LOW_HZ = 50.0
BANDPASS_HIGH_HZ = 500.0

# FFT decomposition: 3 standard signals (frequency bands) for model input.
# Bands: low 50–200 Hz, mid 200–350 Hz, high 350–500 Hz (within bandpass range).
FFT_NUM_BANDS = 3
FFT_BAND_EDGES_HZ = [50.0, 200.0, 350.0, 500.0]

# Preset actions: (channel_0based, num_activations, "tts_phrase")
# e.g. 3 activations on channel 1 (index 0) -> "yes"; 3 on channel 2 (index 1) -> "np"
PRESET_ACTIONS: list[tuple[int, int, str]] = [
    (0, 3, "yes"),
    (1, 3, "np"),
]
# Activation detection: signal (after baseline subtract) must exceed this to count as one activation.
ACTIVATION_THRESHOLD = 400.0
# Min seconds between two activations on same channel (refractory) so one twitch = 1 count.
ACTIVATION_REFRACTORY_SEC = 0.35
# Activations must occur within this window (seconds) to trigger the action; then counter resets.
PRESET_WINDOW_SEC = 4.0
