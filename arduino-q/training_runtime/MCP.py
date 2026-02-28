#!/usr/bin/env python3
"""
Training runtime MCP: collect analog signals from MCU pins, run preprocessing pipeline,
and save result to CSV (format: time x sensors).

Scripts assume they are run from inside training_runtime/ and that pipelines/ is
resolved as if it lived inside training_runtime/ (parent arduino-q is added to path).

Usage (from inside arduino-q/training_runtime/):
  python MCP.py --duration 10 --output training_data.csv
  # or from arduino-q/:
  python -m training_runtime.MCP --duration 10 --output training_data.csv

Bridge: uses arduino.app_utils.Bridge when on device, else SocketBridge to arduino-router.
MCU: expects readAnalogChannels() returning "a0,a1,a2,a3" (pins A0–A3 per MCU.cpp).
"""

import argparse
import csv
import os
import sys
import time

# Run-as-if from training_runtime: add parent (arduino-q) so pipelines and training_runtime import
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import numpy as np

from pipelines.preprocessing import PreprocessingPipeline


def get_bridge(socket_path="/var/run/arduino-router.sock"):
    from training_runtime.bridge_client import get_bridge as _get_bridge
    return _get_bridge(socket_path)


def collect_raw_samples(bridge, duration_sec, interval_sec, num_sensors=4):
    """
    Call MCU readAnalogChannels() every interval_sec for duration_sec.
    Returns (timestamps, data) where data is (n_samples, num_sensors) and timestamps (n_samples,).
    """
    timestamps = []
    rows = []
    t_end = time.monotonic() + duration_sec
    while time.monotonic() < t_end:
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
        time.sleep(max(0.0, interval_sec - (time.monotonic() - t)))
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


def main():
    parser = argparse.ArgumentParser(description="Training runtime: collect analog data, preprocess, save CSV (time x sensors)")
    parser.add_argument("--duration", type=float, default=10.0, help="Collection duration in seconds")
    parser.add_argument("--interval", type=float, default=0.01, help="Seconds between samples")
    parser.add_argument("--output", "-o", default="training_data.csv", help="Output CSV path")
    parser.add_argument("--socket", default="/var/run/arduino-router.sock", help="Bridge Unix socket path (if using SocketBridge)")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip pipeline; save raw data only")
    args = parser.parse_args()

    bridge = get_bridge(args.socket)
    print(f"Collecting for {args.duration}s at {args.interval}s interval...", flush=True)
    timestamps, raw_data = collect_raw_samples(bridge, args.duration, args.interval)
    if raw_data.size == 0:
        print("No samples collected.", file=sys.stderr)
        return 1
    print(f"Collected {len(timestamps)} samples.", flush=True)

    pipeline = None if args.no_preprocess else PreprocessingPipeline()
    run_pipeline_and_save_csv(timestamps, raw_data, args.output, pipeline=pipeline)
    print(f"Wrote {args.output} (time x sensors).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
