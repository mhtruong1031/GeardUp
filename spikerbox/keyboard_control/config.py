"""
Configuration for keyboard control: channels, threshold, key mappings.
"""

import os

NUM_CHANNELS = 2
# Fixed thresholds on abs(signal - baseline). Activate when level >= THRESHOLD_OFFSET.
THRESHOLD_OFFSET = 300.0
# Deactivate when level < DEACTIVATE_OFFSET (hysteresis; use <= THRESHOLD_OFFSET to avoid jitter).
DEACTIVATE_OFFSET = 40.0
# In-app calibration: record this many seconds at startup to compute baseline (if not loading from file).
CALIBRATION_DURATION_SEC = 15.0
# Number of consecutive below-threshold samples required before releasing the key.
DEACTIVATE_BUFFER_SAMPLES = 2

# Channel index -> key to hold while active.
CHANNEL_KEYS = {
    0: "d",
    1: "a",
}

# Sample rate (Human SpikerBox).
SAMPLE_RATE = 5000.0
# Live plot: seconds of history to show per channel.
PLOT_HISTORY_SEC = 1.0

# Calibration (reuse silent_speech format).
DEFAULT_CALIBRATION_DIR = "data/silent_speech"
CALIBRATION_FILENAME = "calibration.npz"
EMG_OFFSET = 8250


def load_calibration(dir_or_path: str):
    """Load baseline mean from calibration.npz; returns shape (NUM_CHANNELS,) or None."""
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
