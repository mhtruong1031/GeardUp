"""
Configuration for keyboard control: channels, threshold, key mappings.
"""

import os

NUM_CHANNELS = 2
# Thresholds scaled to baseline: activate when level >= baseline * (1 + THRESHOLD_SCALE).
THRESHOLD_SCALE = 0.4  # e.g. 0.2 = 20% above baseline
# Deactivate when level < baseline * (1 + DEACTIVATE_SCALE) (hysteresis).
DEACTIVATE_SCALE = 0.1  # e.g. 0.05 = 5% above baseline
# Baseline: EMA of levels when "at rest" (percentile-based rest detection).
INITIAL_WINDOW_CHUNKS = 100  # when no calibration: use first N chunk levels to set initial baseline (median)
MA_ALPHA = 0.05  # EMA of baseline; smaller = smoother
# Baseline movement scale: use MA_ALPHA * REST_ALPHA_SCALE when at rest, MA_ALPHA * ACTIVATION_ALPHA_SCALE when above threshold
REST_ALPHA_SCALE = 1.25  # baseline moves 1.25x as much when at rest
ACTIVATION_ALPHA_SCALE = 0.5  # baseline moves half as much when above threshold (activating)
# Bandpass filter (Hz). Applied to raw stream before level and baseline.
BANDPASS_LOW_HZ = 50.0
BANDPASS_HIGH_HZ = 400.0

# Calibration: explicit "hold still" phase to compute per-channel baseline and noise_std.
CALIBRATION_DURATION_SEC = 5.0
CALIBRATION_FILENAME = "keyboard_control_calibration.npz"
DEFAULT_CALIBRATION_DIR = "data/keyboard_control"

# Level metric per chunk: "max", "rms", or "p95" (95th percentile of |filtered|); p95 reduces spike sensitivity.
LEVEL_METRIC = "p95"
# Rest detection: sliding window of recent levels; "at rest" when current level <= REST_PERCENTILE of window.
REST_WINDOW_SEC = 2.0
REST_WINDOW_CHUNKS = 50  # number of recent level samples per channel for percentile (rest) detection
REST_PERCENTILE = 0.7
# When calibration has noise_std: activation threshold = baseline + max(scale*baseline, NOISE_STD_K * noise_std).
NOISE_STD_K = 3.0

# Channel index -> key when only that channel is active.
CHANNEL_KEYS = {
    0: "d",
    1: "a",
}
# Key to press when both channels are active at the same time (no fighting).
BOTH_KEY = "w"

# Sample rate (Human SpikerBox).
SAMPLE_RATE = 5000.0
# Live plot: seconds of history to show per channel.
PLOT_HISTORY_SEC = 1.0

# SpikerBox 14-bit ADC range.
ADC_MIN, ADC_MAX = 0, 16383


def compute_level(filtered: "np.ndarray", metric: str) -> float:
    """Compute single scalar level from bandpass-filtered chunk. metric in ('max', 'rms', 'p95')."""
    import numpy as np
    abs_f = np.asarray(filtered, dtype=np.float64).ravel()
    if metric == "max":
        return float(np.max(abs_f))
    if metric == "rms":
        return float(np.sqrt(np.mean(abs_f ** 2)))
    if metric == "p95":
        return float(np.percentile(abs_f, 95))
    return float(np.max(abs_f))


def load_calibration(dir_or_path: str):
    """
    Load baseline and optional noise_std from a calibration file.
    Returns dict with keys "baseline" (array), "noise_std" (array or None), or None if not found.
    Pass a directory or full path to keyboard_control_calibration.npz.
    """
    path = dir_or_path if dir_or_path.endswith(".npz") else os.path.join(dir_or_path, CALIBRATION_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        import numpy as np
        data = np.load(path, allow_pickle=True)
        baseline = np.asarray(data["baseline"], dtype=np.float64)
        noise_std = np.asarray(data["noise_std"], dtype=np.float64) if "noise_std" in data else None
        return {"baseline": baseline, "noise_std": noise_std}
    except Exception:
        return None
