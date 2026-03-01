#!/usr/bin/env python3
"""
Runtime MCP: collect analog signals from MCU pins, print data to stdout,
and save CSV (format: time x sensors) only on keyboard interrupt (Ctrl+C).

Scripts assume they are run from inside runtime/ and that pipelines/ is
resolved as if it lived inside runtime/ (parent arduino-q is added to path).

Usage (from inside arduino-q/runtime/):
  python MCP.py
  # Training (default): collect until Ctrl+C, save CSV. Optional: -o other.csv
  python MCP.py --mode runtime
  # Runtime: 100 Hz loop, EMG->steering, EEG->throttle/brake->speed wheel. Ctrl+C to stop.

Bridge: uses arduino.app_utils.Bridge when on device, else SocketBridge to arduino-router.
MCU: expects readAnalogChannels() returning "a0,a1,a2,a3" (pins A0–A3 per MCU.cpp).
"""

# --- Training mode config (edit these instead of CLI) ---
COLLECT_INTERVAL_SEC = 0.01  # 100 Hz
OUTPUT_CSV = "training_data.csv"
PREPROCESS = True
PRINT_SAMPLES = True
SOCKET_PATH = "/var/run/arduino-router.sock"

# --- Runtime mode: "training" | "runtime" ---
RUN_MODE = "training"

# --- Runtime-only config (used when RUN_MODE == "runtime") ---
RUNTIME_INTERVAL_SEC = 0.01  # 100 Hz
# EMG normalization: running window for (EMG1 - EMG2) to get range, then normalize to [-1, 1]
EMG_NORM_WINDOW = 100
# EEG amp mode: moving average window size (samples)
EEG_AMP_WINDOW = 50
# Velocity state
VELOCITY_MAX = 1.0
ACCEL_THROTTLE = 2.0
ACCEL_BRAKE = -4.0
FRICTION = 0.02  # per tick
EEG_DEADZONE = 0.05  # below this deviation from avg = neutral

import argparse
import csv
import os
import sys
import time
from collections import deque

# Run-as-if from runtime: add parent (arduino-q) so pipelines and runtime import
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import numpy as np

from pipelines.preprocessing import PreprocessingPipeline


class VelocityState:
    """Track velocity and acceleration for throttle/brake; output normalized speed 0..1."""

    def __init__(self, v_max=1.0, accel_throttle=2.0, accel_brake=-4.0, friction=0.02):
        self.v_max = v_max
        self.accel_throttle = accel_throttle
        self.accel_brake = accel_brake
        self.friction = friction
        self.velocity = 0.0
        self.acceleration = 0.0

    def update(self, throttle: bool, brake: bool, dt: float):
        if throttle:
            self.acceleration = self.accel_throttle
        elif brake:
            self.acceleration = self.accel_brake
        else:
            self.acceleration = 0.0
        self.velocity += self.acceleration * dt
        self.velocity -= self.friction * self.velocity
        if self.velocity < 0:
            self.velocity = 0.0
        if self.velocity > self.v_max:
            self.velocity = self.v_max

    def get_speed_normalized(self):
        return self.velocity / self.v_max if self.v_max > 0 else 0.0


def get_bridge(socket_path="/var/run/arduino-router.sock"):
    from runtime.bridge_client import get_bridge as _get_bridge
    return _get_bridge(socket_path)


def collect_raw_samples(bridge, interval_sec, num_sensors=4, print_samples=True):
    """
    Call MCU readAnalogChannels() every interval_sec until KeyboardInterrupt.
    Returns (timestamps, data) where data is (n_samples, num_sensors) and timestamps (n_samples,).
    """
    timestamps = []
    rows = []
    try:
        while True:
            t = time.monotonic()
            try:
                raw = bridge.call("readAnalogChannels")
            except Exception as e:
                print(f"Bridge call failed: {e}", file=sys.stderr)
                break
            if raw is None:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            raw = str(raw).strip()
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < num_sensors:
                continue
            values = [float(parts[i]) for i in range(num_sensors)]
            timestamps.append(t)
            rows.append(values)
            if print_samples:
                print(f"  {t:.3f}  " + "  ".join(f"{v:.1f}" for v in values), flush=True)
            time.sleep(max(0.0, interval_sec - (time.monotonic() - t)))
    except KeyboardInterrupt:
        pass
    if not rows:
        return np.array([]), np.array([])
    return np.array(timestamps), np.array(rows, dtype=np.float64)


def run_pipeline_and_save_csv(
    timestamps,
    raw_data,
    output_path,
    pipeline=None,
    sensor_names=None,
):
    """
    Build sensors x time array, run pipeline.preprocess(), save CSV as time x sensors.
    raw_data: (n_samples, n_sensors); timestamps: (n_samples,).
    """
    if pipeline is None:
        pipeline = PreprocessingPipeline()
    if sensor_names is None:
        n_sensors = raw_data.shape[1]
        sensor_names = [f"sensor_{i}" for i in range(n_sensors)]

    # Pipeline expects sensors x time
    data_st = raw_data.T  # (n_sensors, n_samples)
    processed = pipeline.preprocess(data_st)
    if processed is None:
        processed = data_st
    if not isinstance(processed, np.ndarray):
        processed = np.asarray(processed)

    # Ensure 2D: (sensors, time)
    if processed.ndim == 1:
        processed = processed.reshape(1, -1)
    n_sensors_out, n_time = processed.shape
    if len(timestamps) != n_time:
        # Use 0..n_time-1 as time if length mismatch
        timestamps = np.arange(n_time, dtype=np.float64)

    # CSV: time x sensors — header time, sensor_0, ...
    out_names = sensor_names[:n_sensors_out] if len(sensor_names) >= n_sensors_out else [f"sensor_{i}" for i in range(n_sensors_out)]
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + out_names)
        for i in range(n_time):
            row = [float(timestamps[i])] + [float(processed[s, i]) for s in range(n_sensors_out)]
            w.writerow(row)
    return output_path


def _parse_sample(raw, num_sensors=4):
    """Parse 'a0,a1,a2,a3' from bridge into list of floats. Returns None if invalid."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = str(raw).strip()
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < num_sensors:
        return None
    return [float(parts[i]) for i in range(num_sensors)]


def _parse_recent_window(raw, num_sensors=4):
    """Parse getRecentWindow() string: 'a0_0,a1_0,a2_0,a3_0,a0_1,...' -> (n_samples, 4) ndarray."""
    if not raw or not str(raw).strip():
        return np.array([]).reshape(0, num_sensors)
    raw = str(raw).strip()
    parts = [float(p.strip()) for p in raw.split(",")]
    n = len(parts) // num_sensors
    return np.array(parts[: n * num_sensors], dtype=np.float64).reshape(n, num_sensors)


def runtime_loop(bridge):
    """
    100 Hz loop: read one sample, EMG -> setSteering, EEG -> throttle/brake -> velocity -> setSpeedWheel.
    Uses getConfig() for eeg_mode and ring_buffer_size; amp mode uses moving avg, ML mode uses model stub.
    """
    cfg = bridge.call("getConfig")
    if cfg is None:
        cfg = "0,300"
    if isinstance(cfg, bytes):
        cfg = cfg.decode("utf-8", errors="replace")
    parts = str(cfg).strip().split(",")
    eeg_mode = int(parts[0]) if len(parts) > 0 else 0  # 0 = amp, 1 = ML
    ring_size = int(parts[1]) if len(parts) > 1 else 300

    interval_sec = RUNTIME_INTERVAL_SEC
    num_sensors = 4
    # EMG: running min/max of (EMG1 - EMG2) for normalization
    emg_diff_history = deque(maxlen=EMG_NORM_WINDOW)
    # EEG amp: running values for moving average (use raw A0,A1 magnitude or sum)
    eeg_amp_history = deque(maxlen=EEG_AMP_WINDOW)
    velocity_state = VelocityState(
        v_max=VELOCITY_MAX,
        accel_throttle=ACCEL_THROTTLE,
        accel_brake=ACCEL_BRAKE,
        friction=FRICTION,
    )
    # ML stub: model loaded later
    eeg_model = None  # placeholder for torch model

    try:
        while True:
            t0 = time.monotonic()
            try:
                raw = bridge.call("readAnalogChannels")
            except Exception as e:
                print(f"Bridge call failed: {e}", file=sys.stderr)
                break
            sample = _parse_sample(raw, num_sensors)
            if sample is None:
                time.sleep(max(0.0, interval_sec - (time.monotonic() - t0)))
                continue

            # Channels: 0,1 = EEG (A0,A1), 2,3 = EMG (A2,A3)
            emg1, emg2 = sample[2], sample[3]
            eeg0, eeg1_ch = sample[0], sample[1]
            emg_diff = emg1 - emg2
            emg_diff_history.append(emg_diff)
            if len(emg_diff_history) >= 2:
                lo, hi = min(emg_diff_history), max(emg_diff_history)
                span = hi - lo
                if span > 0:
                    normalized = 2.0 * (emg_diff - lo) / span - 1.0
                else:
                    normalized = 0.0
                bridge.call("setSteering", str(normalized))

            # EEG -> throttle/brake
            eeg_amp = (abs(eeg0) + abs(eeg1_ch)) / 2.0
            eeg_amp_history.append(eeg_amp)
            if eeg_mode == 0:
                # Amp mode: above avg = throttle, below = brake
                if len(eeg_amp_history) >= EEG_AMP_WINDOW // 2:
                    avg = sum(eeg_amp_history) / len(eeg_amp_history)
                    dev = eeg_amp - avg
                    if dev > EEG_DEADZONE * (avg + 1e-6):
                        throttle, brake = True, False
                    elif dev < -EEG_DEADZONE * (avg + 1e-6):
                        throttle, brake = False, True
                    else:
                        throttle, brake = False, False
                else:
                    throttle, brake = False, False
            else:
                # ML mode: get recent window, run model (stub)
                try:
                    win_raw = bridge.call("getRecentWindow")
                    window = _parse_recent_window(win_raw, num_sensors)
                    if eeg_model is not None and len(window) > 0:
                        # inference placeholder: model(window) -> 0 or 1 (throttle) or 2 (brake)
                        # pred = eeg_model(...); throttle = (pred == 1); brake = (pred == 2)
                        pass
                    throttle, brake = False, False
                except Exception:
                    throttle, brake = False, False

            velocity_state.update(throttle, brake, interval_sec)
            speed_norm = velocity_state.get_speed_normalized()
            bridge.call("setSpeedWheel", str(speed_norm))

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval_sec - elapsed))
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="Runtime: training (collect CSV) or runtime (100 Hz control loop)")
    parser.add_argument("--output", "-o", default=None, help="Override output CSV path in training mode")
    parser.add_argument("--mode", default=None, choices=("training", "runtime"), help="Override RUN_MODE")
    args = parser.parse_args()

    mode = args.mode if args.mode is not None else RUN_MODE
    bridge = get_bridge(SOCKET_PATH)

    if mode == "runtime":
        print("Runtime mode: 100 Hz loop (EMG->steering, EEG->throttle/brake->speed wheel). Ctrl+C to stop.", flush=True)
        runtime_loop(bridge)
        return 0

    output_path = args.output if args.output is not None else OUTPUT_CSV
    print("Collecting until Ctrl+C... (data printed below; CSV saved on interrupt)", flush=True)
    timestamps, raw_data = collect_raw_samples(
        bridge, COLLECT_INTERVAL_SEC, print_samples=PRINT_SAMPLES
    )
    if raw_data.size == 0:
        print("No samples collected.", file=sys.stderr)
        return 1
    print(f"\nCollected {len(timestamps)} samples. Saving CSV...", flush=True)

    pipeline = None if not PREPROCESS else PreprocessingPipeline()
    run_pipeline_and_save_csv(timestamps, raw_data, output_path, pipeline=pipeline)
    print(f"Wrote {output_path} (time x sensors).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
