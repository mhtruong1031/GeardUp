#!/usr/bin/env python3
"""
Test script: print streamed sample data (A0-A3) and streamed pin outputs.

Pin outputs (mirror MCU.cpp):
  Steering (SimpleFOC): D8 Enable (0/1), D9 Phase A, D5 Phase B, D6 Phase C (PWM 0-4095).
  Speed wheel (single tire, BLDC): D3 PWM (0-4095) from internally tracked velocity.
  Plus the high-level commands: steering_norm (-1..1), speed_norm (0..1).

Use --mock to run without hardware (fake analog data). Without --mock, uses
the same bridge as MCP.py (arduino-router socket or app_utils).

Usage (from arduino-q/runtime/):
  python test_stream_print.py              # real bridge
  python test_stream_print.py --mock       # fake data, no hardware
  python test_stream_print.py --mock -n 50 # mock, stop after 50 samples
"""

import argparse
import math
import os
import sys
import time
from collections import deque

# Constants (match MCP.py runtime config)
RUNTIME_INTERVAL_SEC = 0.01
EMG_NORM_WINDOW = 100
EEG_AMP_WINDOW = 50
VELOCITY_MAX = 1.0
ACCEL_THROTTLE = 2.0
ACCEL_BRAKE = -4.0
FRICTION = 0.02
EEG_DEADZONE = 0.05
SOCKET_PATH = "/var/run/arduino-router.sock"


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
    """Parse getRecentWindow() string into list of sample lists. Returns [] if empty."""
    if not raw or not str(raw).strip():
        return []
    raw = str(raw).strip()
    parts = [float(p.strip()) for p in raw.split(",")]
    n = len(parts) // num_sensors
    return [parts[i * num_sensors : (i + 1) * num_sensors] for i in range(n)]


def get_bridge(socket_path=SOCKET_PATH):
    """Use bridge_client so we don't pull in MCP's numpy/pipelines."""
    from bridge_client import get_bridge as _get_bridge
    return _get_bridge(socket_path)


# Pin numbers (match MCU.cpp)
MOTOR1_ENABLE = 8
MOTOR1_PWM_A = 9
MOTOR1_PWM_B = 5
MOTOR1_PWM_C = 6
SPEED_WHEEL_PWM = 3
PWM_MAX = 4095


def steering_to_pins(normalized_steering: float):
    """
    Same logic as MCU setSteering(): normalized (-1..1) -> Enable, Phase A, B, C.
    Returns (enable 0|1, pwm_a, pwm_b, pwm_c) in 0..4095.
    """
    val = max(-1.0, min(1.0, float(normalized_steering)))
    mag = abs(val)
    pwm_val = int(mag * PWM_MAX)
    pwm_val = min(pwm_val, PWM_MAX)
    enable = 1 if pwm_val > 0 else 0
    if val >= 0:
        return enable, pwm_val, 0, 0
    return enable, 0, pwm_val, 0


def speed_to_pin(normalized_speed: float):
    """Same logic as MCU setSpeedWheel(): normalized (0..1) -> PWM 0..4095."""
    val = max(0.0, min(1.0, float(normalized_speed)))
    return min(int(val * PWM_MAX), PWM_MAX)


class MockBridge:
    """Fake bridge: returns streamed sample data, prints and records pin outputs."""

    def __init__(self, seed=0):
        self._t0 = time.monotonic()
        self._seed = seed
        self.steering_history = []
        self.speed_wheel_history = []

    def call(self, method, *args):
        if method == "readAnalogChannels":
            t = time.monotonic() - self._t0
            # Fake A0–A3: 0–4095 range, gentle variation
            a0 = int(2048 + 800 * math.sin(0.5 * t + self._seed))
            a1 = int(2048 + 800 * math.sin(0.5 * t + 0.3 + self._seed))
            a2 = int(2048 + 600 * math.sin(1.2 * t + self._seed))
            a3 = int(2048 + 600 * math.sin(1.2 * t + 0.7 + self._seed))
            a0 = max(0, min(4095, a0))
            a1 = max(0, min(4095, a1))
            a2 = max(0, min(4095, a2))
            a3 = max(0, min(4095, a3))
            return f"{a0},{a1},{a2},{a3}"
        if method == "getConfig":
            return "0,300"
        if method == "getRecentWindow":
            return ""
        if method == "setSteering":
            val = str(args[0]) if args else "0"
            self.steering_history.append(float(val))
            return None
        if method == "setSpeedWheel":
            val = str(args[0]) if args else "0"
            self.speed_wheel_history.append(float(val))
            return None
        return None


class PrintingBridge:
    """Wraps a bridge; optionally captures last steering/speed for pin print (see run_stream_test)."""

    def __init__(self, bridge):
        self._bridge = bridge
        self._last_steering_norm = None
        self._last_speed_norm = None

    def call(self, method, *args):
        result = self._bridge.call(method, *args)
        if method == "setSteering" and args:
            self._last_steering_norm = float(args[0])
        elif method == "setSpeedWheel" and args:
            self._last_speed_norm = float(args[0])
        return result


def _print_pin_outputs(steering_norm, speed_norm, print_pins: bool):
    """Print one line with all 6-7 pin outputs (SimpleFOC steering + speed wheel)."""
    if not print_pins:
        return
    s = steering_norm if steering_norm is not None else 0.0
    v = speed_norm if speed_norm is not None else 0.0
    enable, pwm_a, pwm_b, pwm_c = steering_to_pins(s)
    speed_pwm = speed_to_pin(v)
    print(
        f"  [PIN] steering_norm={s:.3f}  "
        f"D8_Enable={enable}  D9_A={pwm_a}  D5_B={pwm_b}  D6_C={pwm_c}  |  "
        f"speed_norm={v:.3f}  D3_SpeedWheel={speed_pwm}",
        flush=True,
    )


def run_stream_test(
    bridge,
    interval_sec=RUNTIME_INTERVAL_SEC,
    print_samples=True,
    print_pins=True,
    max_samples=None,
):
    """
    Run one 100 Hz loop: read sample (print it), EMG->steering, EEG->throttle/brake->speed,
    call setSteering/setSpeedWheel (printed by PrintingBridge). Stop on Ctrl+C or after max_samples.
    """
    wrapped = PrintingBridge(bridge)
    cfg = wrapped.call("getConfig")
    if cfg is None:
        cfg = "0,300"
    if isinstance(cfg, bytes):
        cfg = cfg.decode("utf-8", errors="replace")
    parts = str(cfg).strip().split(",")
    eeg_mode = int(parts[0]) if len(parts) > 0 else 0
    num_sensors = 4

    emg_diff_history = deque(maxlen=EMG_NORM_WINDOW)
    eeg_amp_history = deque(maxlen=EEG_AMP_WINDOW)
    velocity_state = VelocityState(
        v_max=VELOCITY_MAX,
        accel_throttle=ACCEL_THROTTLE,
        accel_brake=ACCEL_BRAKE,
        friction=FRICTION,
    )
    eeg_model = None
    n = 0

    try:
        while True:
            if max_samples is not None and n >= max_samples:
                break
            t0 = time.monotonic()
            try:
                raw = wrapped.call("readAnalogChannels")
            except Exception as e:
                print(f"Bridge call failed: {e}", file=sys.stderr)
                break
            sample = _parse_sample(raw, num_sensors)
            if sample is None:
                time.sleep(max(0.0, interval_sec - (time.monotonic() - t0)))
                continue

            if print_samples:
                print(
                    f"[SAMPLE] t={t0:.3f}  A0={sample[0]:.1f}  A1={sample[1]:.1f}  A2={sample[2]:.1f}  A3={sample[3]:.1f}",
                    flush=True,
                )

            emg1, emg2 = sample[2], sample[3]
            eeg0, eeg1_ch = sample[0], sample[1]
            emg_diff = emg1 - emg2
            emg_diff_history.append(emg_diff)
            steering_norm = 0.0  # default when we don't call setSteering yet
            if len(emg_diff_history) >= 2:
                lo, hi = min(emg_diff_history), max(emg_diff_history)
                span = hi - lo
                if span > 0:
                    steering_norm = 2.0 * (emg_diff - lo) / span - 1.0
                else:
                    steering_norm = 0.0
                wrapped.call("setSteering", str(steering_norm))

            eeg_amp = (abs(eeg0) + abs(eeg1_ch)) / 2.0
            eeg_amp_history.append(eeg_amp)
            if eeg_mode == 0:
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
                try:
                    win_raw = wrapped.call("getRecentWindow")
                    window = _parse_recent_window(win_raw, num_sensors)
                    if eeg_model is not None and len(window) > 0:
                        pass
                    throttle, brake = False, False
                except Exception:
                    throttle, brake = False, False

            velocity_state.update(throttle, brake, interval_sec)
            speed_norm = velocity_state.get_speed_normalized()
            wrapped.call("setSpeedWheel", str(speed_norm))

            _print_pin_outputs(steering_norm, speed_norm, print_pins)

            n += 1
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval_sec - elapsed))
    except KeyboardInterrupt:
        pass

    return n


def main():
    parser = argparse.ArgumentParser(
        description="Print streamed sample data and pin outputs (setSteering, setSpeedWheel)."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use fake analog data (no hardware).",
    )
    parser.add_argument(
        "-n",
        "--max-samples",
        type=int,
        default=None,
        help="Stop after N samples (default: run until Ctrl+C).",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Do not print streamed sample lines.",
    )
    parser.add_argument(
        "--no-pins",
        action="store_true",
        help="Do not print setSteering/setSpeedWheel lines.",
    )
    parser.add_argument(
        "--socket",
        default=SOCKET_PATH,
        help="Unix socket path for bridge (default: from MCP).",
    )
    args = parser.parse_args()

    if args.mock:
        bridge = MockBridge()
        print("Using mock bridge (fake A0–A3). Ctrl+C to stop.", flush=True)
    else:
        try:
            bridge = get_bridge(args.socket)
            print("Using real bridge. Ctrl+C to stop.", flush=True)
        except Exception as e:
            print(f"Cannot connect to bridge: {e}", file=sys.stderr)
            print("Use --mock to run without hardware.", file=sys.stderr)
            return 1

    print("Streamed sample data and pin outputs:", flush=True)
    count = run_stream_test(
        bridge,
        print_samples=not args.no_samples,
        print_pins=not args.no_pins,
        max_samples=args.max_samples,
    )
    print(f"\nDone. Processed {count} samples.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
