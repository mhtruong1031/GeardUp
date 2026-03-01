"""
60-second calibration: user does nothing (rest) while we record and compute per-channel baseline mean.
Run before collection and inference so the same baseline is subtracted everywhere.

Usage:
  python -m spikerbox.silent_speech.calibrate --port /dev/cu.usbmodem101 --out data/silent_speech
"""

import argparse
import os
import sys
import time

import numpy as np
import serial.tools.list_ports

from spikerbox import SpikerBox

from .config import (
    SAMPLE_RATE,
    SILENT_SPEECH_CHANNELS,
    CALIBRATION_DURATION_SEC,
    CALIBRATION_FILENAME,
)


def list_ports() -> None:
    """Print available serial ports."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        return
    for p in ports:
        print(f"  {p.device}\t{p.description or '(no description)'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 60s calibration (do nothing) to compute EMG baseline for collection and inference."
    )
    parser.add_argument("--port", default="/dev/cu.usbmodem101", help="Serial port for SpikerBox")
    parser.add_argument("--out", default="data/silent_speech", help="Directory to save calibration.npz")
    parser.add_argument("--duration", type=float, default=CALIBRATION_DURATION_SEC, help="Calibration duration in seconds")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    parser.add_argument("--buffer-size", type=int, default=2000, help="Serial read buffer size (smaller = finer chunks)")
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        return

    duration_sec = args.duration
    target_samples = int(duration_sec * SAMPLE_RATE)
    print(f"Calibration: stay still and do nothing for {duration_sec:.0f} seconds.")
    print(f"Port: {args.port}. Saving to {args.out}/{CALIBRATION_FILENAME}\n")

    all_ch0 = []
    all_ch1 = []
    total_samples = 0

    try:
        with SpikerBox(
            port=args.port,
            input_buffer_size=args.buffer_size,
            num_channels=SILENT_SPEECH_CHANNELS,
        ) as box:
            start = None
            while total_samples < target_samples:
                data = box.run()
                if not isinstance(data, tuple) or len(data) != SILENT_SPEECH_CHANNELS:
                    continue
                ch0, ch1 = np.asarray(data[0], dtype=np.float64), np.asarray(data[1], dtype=np.float64)
                all_ch0.append(ch0)
                all_ch1.append(ch1)
                total_samples += len(ch0)
                if start is None:
                    start = time.perf_counter()
                elapsed = (time.perf_counter() - start) if start else 0
                if total_samples % 5000 < len(ch0):
                    print(f"  {elapsed:.1f}s — {total_samples} / {target_samples} samples")
            mean_ch0 = np.concatenate(all_ch0).mean()
            mean_ch1 = np.concatenate(all_ch1).mean()
            mean = np.array([mean_ch0, mean_ch1], dtype=np.float64)
    except KeyboardInterrupt:
        print("\nCalibration cancelled.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, CALIBRATION_FILENAME)
    np.savez(path, mean=mean, sample_rate=SAMPLE_RATE, duration_sec=duration_sec)
    print(f"\nSaved: {path}")
    print(f"Baseline mean (subtract per channel): ch0={mean[0]:.1f}, ch1={mean[1]:.1f}")


if __name__ == "__main__":
    main()
