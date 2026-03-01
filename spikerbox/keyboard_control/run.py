"""
Keyboard control: stream EMG from SpikerBox; when channel N is above a fixed threshold
(baseline from calibration + offset), hold the configured key; on deactivation release it.
Starts with a calibration period to record baseline. Channel 1 (index 0) -> 'd', channel 2 -> 'a'.

Usage:
  python -m spikerbox.keyboard_control.run --port /dev/cu.usbmodem101
"""

import argparse
import collections
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from pynput.keyboard import Controller as KeyController

from spikerbox import SpikerBox

from .config import (
    NUM_CHANNELS,
    THRESHOLD_OFFSET,
    DEACTIVATE_OFFSET,
    DEACTIVATE_BUFFER_SAMPLES,
    CHANNEL_KEYS,
    DEFAULT_CALIBRATION_DIR,
    CALIBRATION_DURATION_SEC,
    EMG_OFFSET,
    SAMPLE_RATE,
    PLOT_HISTORY_SEC,
    load_calibration,
)


def run_calibration(box: SpikerBox, duration_sec: float) -> np.ndarray:
    """Record for duration_sec and return per-channel mean baseline (shape NUM_CHANNELS)."""
    target_samples = int(duration_sec * SAMPLE_RATE)
    all_ch = [[] for _ in range(NUM_CHANNELS)]
    total = 0
    start = time.perf_counter()
    print(f"Calibrating for {duration_sec:.0f} s — stay still (no muscle tension)...")
    while total < target_samples:
        data = box.run()
        if not isinstance(data, tuple) or len(data) != NUM_CHANNELS:
            continue
        for ch in range(NUM_CHANNELS):
            all_ch[ch].append(np.asarray(data[ch], dtype=np.float64))
        total += len(data[0])
        elapsed = time.perf_counter() - start
        if total % 5000 < len(data[0]):
            print(f"  {elapsed:.1f} s — {total} / {target_samples} samples")
    baseline = np.array([np.concatenate(all_ch[ch]).mean() for ch in range(NUM_CHANNELS)], dtype=np.float64)
    print(f"Baseline (mean per ch): {baseline}")
    return baseline


def run(
    port: str = "/dev/cu.usbmodem101",
    buffer_size: int = 2000,
    calibration_path: str | None = None,
    calibration_duration_sec: float | None = None,
    threshold_offset: float | None = None,
    plot_live: bool = True,
) -> None:
    """
    Run calibration (or load from file), then stream EMG. When abs(signal - baseline) > threshold_offset,
    channel is active and holds its key; when below deactivate_offset, release (with buffer).
    """
    thresh = threshold_offset if threshold_offset is not None else THRESHOLD_OFFSET
    cal_sec = calibration_duration_sec if calibration_duration_sec is not None else CALIBRATION_DURATION_SEC

    baseline = None
    if calibration_path:
        baseline = load_calibration(calibration_path)
    if baseline is None and os.path.isdir(DEFAULT_CALIBRATION_DIR):
        baseline = load_calibration(DEFAULT_CALIBRATION_DIR)
    # If no file calibration, we'll run in-app calibration below (inside SpikerBox context)

    deactivate_count = [0] * NUM_CHANNELS  # consecutive below-threshold chunks before release
    key_controller = KeyController()

    # Live plot: ring buffer of recent abs(signal) per channel
    plot_len = int(SAMPLE_RATE * PLOT_HISTORY_SEC)
    plot_bufs = [collections.deque(maxlen=plot_len) for _ in range(NUM_CHANNELS)]
    fig, axes = None, None
    line_handles = None
    thresh_lines = None
    if plot_live:
        fig, axes = plt.subplots(NUM_CHANNELS, 1, sharex=True, figsize=(8, 4))
        if NUM_CHANNELS == 1:
            axes = [axes]
        fig.suptitle("EMG (abs, baseline-subtracted) — Ch0: 'd', Ch1: 'a'")
        line_handles = [ax.plot([], [], "b-", lw=0.8)[0] for ax in axes]
        thresh_lines = []
        for ch, ax in enumerate(axes):
            ax.set_ylabel(f"Ch{ch} ({CHANNEL_KEYS.get(ch, '?')})")
            l_high, = ax.plot([], [], "r--", lw=0.8, label="activate")
            l_low, = ax.plot([], [], "orange", linestyle=":", lw=0.8, label="deactivate")
            thresh_lines.append((l_high, l_low))
        axes[-1].set_xlabel("Time (s)")
        plt.ion()
        plt.show(block=False)

    print(f"Keyboard control: port={port}, threshold={thresh}, deactivate<{DEACTIVATE_OFFSET}")
    print("Channel 1 (index 0) -> hold 'd', Channel 2 (index 1) -> hold 'a'. Deactivate releases key.")
    if plot_live:
        print("Live plot enabled.")
    print("Ctrl+C to stop.\n")

    try:
        with SpikerBox(port=port, input_buffer_size=buffer_size, num_channels=NUM_CHANNELS) as box:
            if baseline is None:
                baseline = run_calibration(box, cal_sec)
            else:
                baseline = np.asarray(baseline, dtype=np.float64)
                if baseline.shape[0] != NUM_CHANNELS:
                    baseline = np.array([baseline[0]] * NUM_CHANNELS, dtype=np.float64)

            while True:
                data = box.run()
                if not isinstance(data, tuple) or len(data) != NUM_CHANNELS:
                    continue

                # Fixed thresholds (no MA): activate above thresh, deactivate below DEACTIVATE_OFFSET
                levels = []
                for ch in range(NUM_CHANNELS):
                    arr = np.asarray(data[ch], dtype=np.float64) - baseline[ch]
                    level = np.abs(arr).max()
                    levels.append(level)
                    if plot_live and plot_bufs[ch] is not None:
                        for v in np.abs(arr).tolist():
                            plot_bufs[ch].append(v)
                threshold_highs = [thresh] * NUM_CHANNELS
                threshold_lows = [DEACTIVATE_OFFSET] * NUM_CHANNELS

                # Only the channel with the bigger level (above its threshold) can be active
                candidates = [
                    ch for ch in range(NUM_CHANNELS)
                    if levels[ch] >= threshold_highs[ch] and CHANNEL_KEYS.get(ch) is not None
                ]
                winner = None
                if candidates:
                    winner = max(candidates, key=lambda ch: levels[ch])

                for ch in range(NUM_CHANNELS):
                    key = CHANNEL_KEYS.get(ch)
                    if key is None:
                        continue
                    th_high = threshold_highs[ch]
                    th_low = threshold_lows[ch]
                    level = levels[ch]

                    if ch == winner:
                        deactivate_count[ch] = 0
                        if not active[ch]:
                            active[ch] = True
                            key_controller.press(key)
                    else:
                        # This channel is not the winner: release if we were holding, or run deactivate buffer
                        if active[ch]:
                            if winner is not None:
                                # Other channel won, release immediately
                                active[ch] = False
                                deactivate_count[ch] = 0
                                key_controller.release(key)
                            elif level < th_low:
                                deactivate_count[ch] += 1
                                if deactivate_count[ch] >= DEACTIVATE_BUFFER_SAMPLES:
                                    active[ch] = False
                                    deactivate_count[ch] = 0
                                    key_controller.release(key)
                        else:
                            deactivate_count[ch] = 0

                if plot_live and fig is not None and plt.fignum_exists(fig.number):
                    n = len(plot_bufs[0])
                    if n > 0:
                        t = np.arange(n) / SAMPLE_RATE
                        t = t - t[-1]  # time 0 = now
                        for ch in range(NUM_CHANNELS):
                            y = np.array(plot_bufs[ch])
                            line_handles[ch].set_data(t, y)
                            l_high, l_low = thresh_lines[ch]
                            l_high.set_data([t[0], t[-1]], [thresh, thresh])
                            l_low.set_data([t[0], t[-1]], [DEACTIVATE_OFFSET, DEACTIVATE_OFFSET])
                            axes[ch].relim()
                            axes[ch].autoscale_view(scalex=False)
                            if active[ch]:
                                axes[ch].set_facecolor("#ffe0e0")
                            else:
                                axes[ch].set_facecolor("white")
                        fig.canvas.draw_idle()
                        plt.pause(0.001)
    finally:
        for ch in range(NUM_CHANNELS):
            if active[ch]:
                key = CHANNEL_KEYS.get(ch)
                if key is not None:
                    key_controller.release(key)
        if plot_live and fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EMG keyboard control: calibration then channel 1 holds 'd', channel 2 holds 'a' (fixed threshold)."
    )
    parser.add_argument("--port", default="/dev/cu.usbmodem101", help="Serial port")
    parser.add_argument("--buffer-size", type=int, default=2000, help="Serial read buffer size")
    parser.add_argument("--calibration", default=None, help="Path to calibration dir or .npz to skip in-app calibration")
    parser.add_argument("--calibration-duration", type=float, default=None, help=f"Seconds to calibrate if no file (default: {CALIBRATION_DURATION_SEC})")
    parser.add_argument("--threshold", type=float, default=None, help=f"Activation threshold on abs(signal-baseline) (default: {THRESHOLD_OFFSET})")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plot")
    args = parser.parse_args()

    try:
        run(
            port=args.port,
            buffer_size=args.buffer_size,
            calibration_path=args.calibration,
            calibration_duration_sec=args.calibration_duration,
            threshold_offset=args.threshold,
            plot_live=not args.no_plot,
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
