"""
Preset-action inference: count EMG activations per channel and trigger TTS (e.g. 3 on ch1 -> "yes", 3 on ch2 -> "np").
No CNN model; uses threshold + refractory to count distinct activations.

Usage:
  python -m spikerbox.silent_speech.preset_inference --port /dev/cu.usbmodem101
"""

import argparse
import os
import time

import numpy as np

from spikerbox import SpikerBox

from .config import (
    SILENT_SPEECH_CHANNELS,
    DEFAULT_DATA_DIR,
    EMG_OFFSET,
    load_calibration,
    PRESET_ACTIONS,
    ACTIVATION_THRESHOLD,
    ACTIVATION_REFRACTORY_SEC,
    PRESET_WINDOW_SEC,
)


def run_preset_inference(
    port: str = "/dev/cu.usbmodem101",
    buffer_size: int = 2000,
    use_tts: bool = True,
    calibration_path: str | None = None,
    threshold: float | None = None,
    refractory_sec: float | None = None,
    window_sec: float | None = None,
) -> None:
    """
    Stream EMG from SpikerBox. For each channel, count activations (signal above baseline + threshold)
    with refractory. When a channel reaches the count for a preset action, speak the phrase (TTS) and reset.
    """
    thresh = threshold if threshold is not None else ACTIVATION_THRESHOLD
    refractory = refractory_sec if refractory_sec is not None else ACTIVATION_REFRACTORY_SEC
    window = window_sec if window_sec is not None else PRESET_WINDOW_SEC

    cal = None
    if calibration_path:
        cal = load_calibration(calibration_path)
    if cal is None and os.path.isdir(DEFAULT_DATA_DIR):
        cal = load_calibration(DEFAULT_DATA_DIR)
    baseline = cal if cal is not None else np.array([EMG_OFFSET, EMG_OFFSET], dtype=np.float64)

    # Per-channel: activation count in current window; last activation time for refractory
    activation_count = [0] * SILENT_SPEECH_CHANNELS
    last_activation_time = [-1.0] * SILENT_SPEECH_CHANNELS
    window_start_time = [time.perf_counter()] * SILENT_SPEECH_CHANNELS

    if use_tts:
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
        except Exception as e:
            print(f"TTS unavailable: {e}. Continuing without TTS.")
            use_tts = False
            tts_engine = None
    else:
        tts_engine = None

    print(f"Preset inference: port={port}, threshold={thresh}, refractory={refractory}s, window={window}s")
    print("Actions:", PRESET_ACTIONS)
    print("Ctrl+C to stop.\n")

    with SpikerBox(port=port, input_buffer_size=buffer_size, num_channels=SILENT_SPEECH_CHANNELS) as box:
        while True:
            data = box.run()
            now = time.perf_counter()

            if not isinstance(data, tuple) or len(data) != SILENT_SPEECH_CHANNELS:
                continue

            # data is (ch0_array, ch1_array); subtract baseline and check for activations
            for ch in range(SILENT_SPEECH_CHANNELS):
                arr = np.asarray(data[ch], dtype=np.float64) - baseline[ch]
                # Activation = any sample in this chunk exceeds threshold (use abs so both directions count)
                peak = np.abs(arr).max()
                if peak >= thresh:
                    if now - last_activation_time[ch] >= refractory:
                        last_activation_time[ch] = now
                        activation_count[ch] += 1
                # Reset window if we've exceeded the time window
                if now - window_start_time[ch] > window:
                    activation_count[ch] = 0
                    window_start_time[ch] = now

            # Check preset actions: if channel reached required count, trigger and reset
            for channel_idx, required_count, phrase in PRESET_ACTIONS:
                if channel_idx >= SILENT_SPEECH_CHANNELS:
                    continue
                if activation_count[channel_idx] >= required_count:
                    print(phrase)
                    if use_tts and tts_engine:
                        tts_engine.say(phrase)
                        tts_engine.runAndWait()
                    activation_count[channel_idx] = 0
                    window_start_time[channel_idx] = time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preset actions: N activations on channel -> TTS phrase (no CNN)."
    )
    parser.add_argument("--port", default="/dev/cu.usbmodem101", help="Serial port")
    parser.add_argument("--buffer-size", type=int, default=2000, help="Serial read buffer size")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech; only print phrase")
    parser.add_argument("--calibration", default=None, help="Path to calibration dir or calibration.npz")
    parser.add_argument("--threshold", type=float, default=None, help=f"Activation amplitude threshold (default: {ACTIVATION_THRESHOLD})")
    parser.add_argument("--refractory", type=float, default=None, help=f"Min seconds between activations (default: {ACTIVATION_REFRACTORY_SEC})")
    parser.add_argument("--window", type=float, default=None, help=f"Seconds to count activations before reset (default: {PRESET_WINDOW_SEC})")
    args = parser.parse_args()

    try:
        run_preset_inference(
            port=args.port,
            buffer_size=args.buffer_size,
            use_tts=not args.no_tts,
            calibration_path=args.calibration,
            threshold=args.threshold,
            refractory_sec=args.refractory,
            window_sec=args.window,
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
