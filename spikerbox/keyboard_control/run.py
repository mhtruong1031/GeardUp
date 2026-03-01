"""
Keyboard control: stream EMG from SpikerBox; bandpass filter, then per-channel baseline
(loaded from calibration or initial window) with percentile-based rest detection and EMA
adaptation. Channel 0 -> 'd', channel 1 -> 'a'.

Usage:
  python -m spikerbox.keyboard_control.run --port /dev/cu.usbmodem101
  python -m spikerbox.keyboard_control.calibrate --port /dev/cu.usbmodem101 --out data/keyboard_control  # run first for best results
"""

import argparse
import collections

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from pynput.keyboard import Controller as KeyController

from spikerbox import SpikerBox

from .config import (
    NUM_CHANNELS,
    THRESHOLD_SCALE,
    DEACTIVATE_SCALE,
    INITIAL_WINDOW_CHUNKS,
    MA_ALPHA,
    REST_ALPHA_SCALE,
    ACTIVATION_ALPHA_SCALE,
    BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ,
    CHANNEL_KEYS,
    BOTH_KEY,
    SAMPLE_RATE,
    PLOT_HISTORY_SEC,
    LEVEL_METRIC,
    REST_WINDOW_CHUNKS,
    REST_PERCENTILE,
    NOISE_STD_K,
    DEFAULT_CALIBRATION_DIR,
    CALIBRATION_FILENAME,
    CALIBRATION_DURATION_SEC,
    load_calibration,
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


def _threshold_high(baseline_ch: float, noise_std_ch: float | None, thresh_scale: float) -> float:
    """Activation threshold: baseline + scale or baseline + k*noise_std when calibration has noise_std."""
    if noise_std_ch is not None and noise_std_ch > 0:
        return baseline_ch + max(thresh_scale * baseline_ch, NOISE_STD_K * noise_std_ch)
    return baseline_ch * (1.0 + thresh_scale)


def run(
    port: str = "/dev/cu.usbmodem101",
    buffer_size: int = 2000,
    threshold_offset: float | None = None,
    ma_alpha: float | None = None,
    calibration_path: str | None = None,
    plot_live: bool = True,
) -> None:
    """
    Stream EMG -> bandpass -> level (configurable: max/rms/p95). Baseline from calibration
    or initial window; "at rest" = level <= REST_PERCENTILE of recent levels; baseline EMA
    updated only when at rest. Activate when level >= threshold_high; deactivate with hysteresis.
    """
    thresh_scale = threshold_offset if threshold_offset is not None else THRESHOLD_SCALE
    deact_scale = DEACTIVATE_SCALE
    alpha = ma_alpha if ma_alpha is not None else MA_ALPHA

    # Load calibration if path given or default path exists
    cal = None
    if calibration_path:
        cal = load_calibration(calibration_path)
    if cal is None:
        cal = load_calibration(DEFAULT_CALIBRATION_DIR)
    noise_std = cal["noise_std"] if cal and cal.get("noise_std") is not None else None
    if cal:
        baseline = [float(cal["baseline"][ch]) for ch in range(NUM_CHANNELS)]
        baseline_initialized = [True] * NUM_CHANNELS
    else:
        baseline = [0.0] * NUM_CHANNELS  # set from initial window
        baseline_initialized = [False] * NUM_CHANNELS

    b, a = _make_bandpass()
    zi_list = [scipy_signal.lfilter_zi(b, a) * 0.0 for _ in range(NUM_CHANNELS)]
    # Per-channel: recent levels for percentile-based rest detection (non-circular)
    rest_level_deques = [collections.deque(maxlen=REST_WINDOW_CHUNKS) for _ in range(NUM_CHANNELS)]
    initial_levels = [[] for _ in range(NUM_CHANNELS)]  # for initial baseline when no calibration
    active = [False] * NUM_CHANNELS  # for plot: which channels are above threshold
    current_key_pressed: str | None = None  # one of 'd', 'a', 'w', or None
    key_controller = KeyController()

    plot_len = int(SAMPLE_RATE * PLOT_HISTORY_SEC)
    plot_bufs = [collections.deque(maxlen=plot_len) for _ in range(NUM_CHANNELS)]
    fig, axes = None, None
    line_handles = None
    thresh_lines = None
    if plot_live:
        fig, axes = plt.subplots(NUM_CHANNELS, 1, sharex=True, figsize=(8, 4))
        if NUM_CHANNELS == 1:
            axes = [axes]
        fig.suptitle("EMG bandpass abs — Ch0: 'd', Ch1: 'a', both: 'w' (baseline + percentile rest)")
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

    print(f"Keyboard control: port={port}, bandpass {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz, level={LEVEL_METRIC}")
    if cal:
        print(f"  Calibration loaded: baseline per channel, rest window={REST_WINDOW_CHUNKS} chunks, rest percentile={REST_PERCENTILE}")
    else:
        print(f"  No calibration: initial baseline from first {INITIAL_WINDOW_CHUNKS} chunks; run calibrate for best results.")
    print("Ch0 -> 'd', Ch1 -> 'a', both -> 'w'. Ctrl+C to stop.\n")

    try:
        with SpikerBox(port=port, input_buffer_size=buffer_size, num_channels=NUM_CHANNELS) as box:
            while True:
                data = box.run()
                if not isinstance(data, tuple) or len(data) != NUM_CHANNELS:
                    continue

                levels = []
                for ch in range(NUM_CHANNELS):
                    raw = np.asarray(data[ch], dtype=np.float64)
                    filtered, zi_list[ch] = _bandpass_chunk(raw, b, a, zi_list[ch])
                    level = compute_level(filtered, LEVEL_METRIC)
                    levels.append(level)
                    if plot_live:
                        for v in np.abs(filtered).tolist():
                            plot_bufs[ch].append(v)
                    rest_level_deques[ch].append(level)
                    # Initial baseline from first N chunks when no calibration
                    if not baseline_initialized[ch]:
                        initial_levels[ch].append(level)
                        if len(initial_levels[ch]) >= INITIAL_WINDOW_CHUNKS:
                            baseline[ch] = float(np.median(initial_levels[ch]))
                            baseline_initialized[ch] = True

                # Compute thresholds so we can choose baseline update rate: rest (1.25x) vs activation (0.5x)
                threshold_highs = [
                    _threshold_high(baseline[ch], noise_std[ch] if noise_std is not None else None, thresh_scale)
                    for ch in range(NUM_CHANNELS)
                ]
                for ch in range(NUM_CHANNELS):
                    if not baseline_initialized[ch]:
                        continue
                    level = levels[ch]
                    at_rest = False
                    if len(rest_level_deques[ch]) >= 2:
                        p = np.percentile(list(rest_level_deques[ch]), REST_PERCENTILE * 100)
                        at_rest = level <= p
                    if level >= threshold_highs[ch]:
                        # Activating: baseline moves half as much
                        eff_alpha = min(alpha * ACTIVATION_ALPHA_SCALE, 1.0)
                        baseline[ch] = eff_alpha * level + (1.0 - eff_alpha) * baseline[ch]
                    elif at_rest:
                        # At rest: baseline moves 1.25x as much
                        eff_alpha = min(alpha * REST_ALPHA_SCALE, 1.0)
                        baseline[ch] = eff_alpha * level + (1.0 - eff_alpha) * baseline[ch]

                threshold_lows = [baseline[ch] * (1.0 + deact_scale) for ch in range(NUM_CHANNELS)]

                candidates = [
                    ch for ch in range(NUM_CHANNELS)
                    if baseline_initialized[ch] and levels[ch] >= threshold_highs[ch] and CHANNEL_KEYS.get(ch) is not None
                ]
                # Both active -> 'w'; only ch0 -> 'd'; only ch1 -> 'a'; none -> None (no fighting)
                if len(candidates) >= 2:
                    desired_key = BOTH_KEY
                elif len(candidates) == 1:
                    desired_key = CHANNEL_KEYS[candidates[0]]
                else:
                    desired_key = None
                for ch in range(NUM_CHANNELS):
                    active[ch] = ch in candidates
                if desired_key != current_key_pressed:
                    if current_key_pressed is not None:
                        key_controller.release(current_key_pressed)
                    current_key_pressed = desired_key
                    if current_key_pressed is not None:
                        key_controller.press(current_key_pressed)

                if plot_live and fig is not None and plt.fignum_exists(fig.number):
                    n = len(plot_bufs[0])
                    if n > 0:
                        t = np.arange(n) / SAMPLE_RATE
                        t = t - t[-1]
                        for ch in range(NUM_CHANNELS):
                            y = np.array(plot_bufs[ch])
                            line_handles[ch].set_data(t, y)
                            l_high, l_low = thresh_lines[ch]
                            th_high = _threshold_high(baseline[ch], noise_std[ch] if noise_std is not None else None, thresh_scale)
                            l_high.set_data([t[0], t[-1]], [th_high, th_high])
                            l_low.set_data([t[0], t[-1]], [threshold_lows[ch], threshold_lows[ch]])
                            axes[ch].relim()
                            axes[ch].autoscale_view(scalex=False)
                            axes[ch].set_facecolor("#ffe0e0" if active[ch] else "white")
                        fig.canvas.draw_idle()
                        plt.pause(0.001)
    finally:
        if current_key_pressed is not None:
            key_controller.release(current_key_pressed)
        if plot_live and fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EMG keyboard control: Ch0->'d', Ch1->'a', both->'w'. Bandpass + per-channel baseline (calibration or initial window)."
    )
    parser.add_argument("--port", default="/dev/cu.usbmodem101", help="Serial port")
    parser.add_argument("--buffer-size", type=int, default=2000, help="Serial read buffer size")
    parser.add_argument("--threshold", type=float, default=None, help=f"Scale above baseline for activation (default: {THRESHOLD_SCALE})")
    parser.add_argument("--ma-alpha", type=float, default=None, help=f"EMA alpha for baseline when at rest (default: {MA_ALPHA})")
    parser.add_argument("--calibration", default=None, help="Path to calibration dir or .npz (default: " + DEFAULT_CALIBRATION_DIR + ")")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration first (hold still), save to --calibration or default dir, then run")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plot")
    args = parser.parse_args()

    calibration_path = args.calibration
    if args.calibrate:
        import os
        from .calibrate import run_calibration
        out_dir = calibration_path or DEFAULT_CALIBRATION_DIR
        if out_dir.endswith(".npz"):
            out_dir = os.path.dirname(out_dir) or "."
        print("Running calibration (hold still)...")
        result = run_calibration(port=args.port, out_dir=out_dir, buffer_size=args.buffer_size, duration_sec=CALIBRATION_DURATION_SEC)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, CALIBRATION_FILENAME)
        np.savez(path, baseline=result["baseline"], noise_std=result["noise_std"], sample_rate=SAMPLE_RATE, duration_sec=CALIBRATION_DURATION_SEC)
        print(f"Calibration saved to {path}. Starting keyboard control.\n")
        calibration_path = out_dir

    try:
        run(
            port=args.port,
            buffer_size=args.buffer_size,
            threshold_offset=args.threshold,
            ma_alpha=args.ma_alpha,
            calibration_path=calibration_path,
            plot_live=not args.no_plot,
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
