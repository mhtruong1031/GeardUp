"""
Calibration for keyboard control: record rest-only EMG, compute per-channel baseline and
noise_std (same bandpass and level metric as run). Run before keyboard_control for robust
activation detection.

Usage:
  python -m spikerbox.keyboard_control.calibrate --port /dev/cu.usbmodem101 --out data/keyboard_control
"""

import argparse
import os
import sys

import numpy as np
from scipy import signal as scipy_signal

from spikerbox import SpikerBox

from .config import (
    NUM_CHANNELS,
    SAMPLE_RATE,
    CALIBRATION_DURATION_SEC,
    CALIBRATION_FILENAME,
    LEVEL_METRIC,
    BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ,
    compute_level,
)


def _make_bandpass():
    nyq = SAMPLE_RATE / 2.0
    low = max(BANDPASS_LOW_HZ / nyq, 0.001)
    high = min(BANDPASS_HIGH_HZ / nyq, 0.999)
    b, a = scipy_signal.butter(4, [low, high], btype="band")
    return b, a


def _bandpass_chunk(x: np.ndarray, b: np.ndarray, a: np.ndarray, zi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal bandpass one chunk; returns (filtered, new_zi)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    out, zf = scipy_signal.lfilter(b, a, x, zi=zi)
    return out, zf


def run_calibration(
    port: str = "/dev/cu.usbmodem101",
    out_dir: str = "data/keyboard_control",
    duration_sec: float = CALIBRATION_DURATION_SEC,
    buffer_size: int = 2000,
) -> dict:
    """
    Record rest-only EMG for duration_sec; compute per-channel baseline (median of levels)
    and noise_std (IQR/1.35). Returns dict with baseline, noise_std arrays.
    """
    b, a = _make_bandpass()
    zi_list = [scipy_signal.lfilter_zi(b, a) * 0.0 for _ in range(NUM_CHANNELS)]
    level_lists = [[] for _ in range(NUM_CHANNELS)]
    target_samples = int(duration_sec * SAMPLE_RATE)
    total_samples = 0

    with SpikerBox(
        port=port,
        input_buffer_size=buffer_size,
        num_channels=NUM_CHANNELS,
    ) as box:
        while total_samples < target_samples:
            data = box.run()
            if not isinstance(data, tuple) or len(data) != NUM_CHANNELS:
                continue
            for ch in range(NUM_CHANNELS):
                raw = np.asarray(data[ch], dtype=np.float64)
                filtered, zi_list[ch] = _bandpass_chunk(raw, b, a, zi_list[ch])
                level = compute_level(filtered, LEVEL_METRIC)
                level_lists[ch].append(level)
            total_samples += len(data[0])

    baseline = np.zeros(NUM_CHANNELS, dtype=np.float64)
    noise_std = np.zeros(NUM_CHANNELS, dtype=np.float64)
    for ch in range(NUM_CHANNELS):
        arr = np.array(level_lists[ch], dtype=np.float64)
        baseline[ch] = float(np.median(arr))
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        noise_std[ch] = iqr / 1.35 if iqr > 0 else float(np.std(arr)) if len(arr) > 1 else 0.0

    return {"baseline": baseline, "noise_std": noise_std}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate keyboard control: hold still to record baseline and noise_std per channel."
    )
    parser.add_argument("--port", default="/dev/cu.usbmodem101", help="Serial port for SpikerBox")
    parser.add_argument("--out", default="data/keyboard_control", help="Directory to save calibration.npz")
    parser.add_argument("--duration", type=float, default=CALIBRATION_DURATION_SEC, help="Calibration duration in seconds")
    parser.add_argument("--buffer-size", type=int, default=2000, help="Serial read buffer size")
    args = parser.parse_args()

    print(f"Calibration: hold still for {args.duration:.0f} s. Port: {args.port}")
    print(f"Level metric: {LEVEL_METRIC}. Saving to {args.out}/{CALIBRATION_FILENAME}\n")

    try:
        result = run_calibration(
            port=args.port,
            out_dir=args.out,
            duration_sec=args.duration,
            buffer_size=args.buffer_size,
        )
    except KeyboardInterrupt:
        print("\nCalibration cancelled.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, CALIBRATION_FILENAME)
    np.savez(
        path,
        baseline=result["baseline"],
        noise_std=result["noise_std"],
        sample_rate=SAMPLE_RATE,
        duration_sec=args.duration,
    )
    print(f"Saved: {path}")
    for ch in range(NUM_CHANNELS):
        print(f"  Ch{ch}: baseline={result['baseline'][ch]:.1f}, noise_std={result['noise_std'][ch]:.1f}")


if __name__ == "__main__":
    main()
