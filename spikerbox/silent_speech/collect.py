"""
Labelled data collection for Silent Speech Decoder.

Hold down ` (backtick) while think-speaking the prompted word; release to end recording.
Saves variable-length segments (emg shape (2, T), label, sample_rate, actual_len).
Training pads/trims to a fixed window in the dataset; the FCN localizes the word within it.

Usage:
  python -m spikerbox.silent_speech.collect --port /dev/cu.usbmodem101 --out data/silent_speech
  python -m spikerbox.silent_speech.collect --list-ports
"""

import argparse
import copy
import os
import sys
import threading
import time

import numpy as np
import serial.tools.list_ports

from spikerbox import SpikerBox

from .config import (
    VOCAB,
    SAMPLE_RATE,
    SILENT_SPEECH_CHANNELS,
    EMG_OFFSET,
    BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ,
    load_calibration,
)
from .dataset import bandpass_emg

# Minimum length for bandpass in live plot (filtfilt needs input longer than padlen).
MIN_PLOT_SAMPLES_FOR_BANDPASS = 100


def list_ports() -> None:
    """Print available serial ports."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        return
    for p in ports:
        print(f"  {p.device}\t{p.description or '(no description)'}")


def _update_plot(axes, fig, buffers: list[np.ndarray], label: str) -> None:
    """Redraw live EMG plot from current buffers."""
    if axes is None or fig is None:
        return
    n = min(len(b) for b in buffers) if buffers else 0
    if n == 0:
        return
    t = np.arange(n) / SAMPLE_RATE
    for ch, ax in enumerate(axes):
        ax.clear()
        ax.set_ylabel(f"Ch{ch + 1}")
        ax.plot(t, buffers[ch][-n:])
        ax.set_xlim(max(0, t[-1] - 2.0), t[-1])
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title(f"Recording: {label}" if label else "Live EMG")
    fig.canvas.draw()
    fig.canvas.flush_events()


def next_filename(out_dir: str, label: str) -> str:
    """Return next available path like out_dir/yes_001.npz."""
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.startswith(label) and f.endswith(".npz")]
    indices = []
    for f in existing:
        try:
            base = f[: -len(".npz")]
            num = base.split("_")[-1]
            indices.append(int(num))
        except (ValueError, IndexError):
            continue
    next_num = max(indices, default=0) + 1
    return os.path.join(out_dir, f"{label}_{next_num:03d}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect labelled subvocal EMG for Silent Speech Decoder."
    )
    parser.add_argument(
        "--port",
        default="/dev/cu.usbmodem101",
        help="Serial port for SpikerBox",
    )
    parser.add_argument(
        "--out",
        default="data/silent_speech",
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=2000,
        help="Serial read buffer size (samples); smaller = finer chunks",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List serial ports and exit",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable live EMG plot during collection",
    )
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        return

    try:
        from pynput import keyboard
    except ImportError:
        print("Hold-to-record requires pynput. Install with: pip install pynput")
        sys.exit(1)

    use_plot = not args.no_plot
    axes = None
    fig = None
    if use_plot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, axes = plt.subplots(SILENT_SPEECH_CHANNELS, 1, sharex=True)
        if SILENT_SPEECH_CHANNELS == 1:
            axes = [axes]
        for ax in axes:
            ax.set_xlabel("time [s]")
        fig.show()

    # Baseline: from calibration (60s rest) or fallback offset.
    cal = load_calibration(args.out)
    baseline = cal if cal is not None else np.array([EMG_OFFSET, EMG_OFFSET], dtype=np.float64)
    if cal is None:
        print("No calibration found. Run: python3 -m spikerbox.silent_speech.calibrate --out", args.out)
        print("Using fixed offset for plot. For best results, calibrate first.\n")

    print(f"Silent Speech data collection: port={args.port}, out={args.out}")
    print(f"Vocabulary: {VOCAB}. Variable-length segments @ {SAMPLE_RATE} Hz.")
    print("Hold down ` (backtick) while think-speaking the word; release to save.")
    if use_plot:
        print("Live plot enabled. Ctrl+C to quit.\n")
    else:
        print("Ctrl+C to quit.\n")

    lock = threading.Lock()
    recording = False
    recording_chunks: list[tuple[np.ndarray, np.ndarray]] = []
    live_buffers: list[np.ndarray] = [np.array([]), np.array([])]
    vocab_index = [0]  # list so callback can mutate
    repeat_count = [0]  # number of times current word collected this round (0..REPEATS_PER_WORD-1)
    REPEATS_PER_WORD = 5
    out_dir = args.out
    input_buffer_size = args.buffer_size
    stop_thread = threading.Event()

    def reader_thread(box: SpikerBox) -> None:
        nonlocal live_buffers
        while not stop_thread.is_set():
            try:
                data = box.run()
            except Exception:
                break
            if not isinstance(data, tuple) or len(data) != SILENT_SPEECH_CHANNELS:
                continue
            with lock:
                if recording:
                    recording_chunks.append((np.array(data[0]).copy(), np.array(data[1]).copy()))
                for ch, arr in enumerate(data):
                    live_buffers[ch] = np.append(live_buffers[ch], arr)
                    max_keep = int(3.0 * SAMPLE_RATE)
                    if len(live_buffers[ch]) > max_keep:
                        live_buffers[ch] = live_buffers[ch][-max_keep:]

    def on_press(key) -> None:
        nonlocal recording
        try:
            if getattr(key, "char", None) == "`":
                with lock:
                    recording = True
                    recording_chunks.clear()
        except Exception:
            pass

    def on_release(key) -> None:
        nonlocal recording
        try:
            if getattr(key, "char", None) != "`":
                return
            with lock:
                recording = False
                chunks = copy.copy(recording_chunks)
                recording_chunks.clear()
            if not chunks:
                print("(no data)")
                return
            ch0 = np.concatenate([c[0] for c in chunks])
            ch1 = np.concatenate([c[1] for c in chunks])
            T = len(ch0)
            emg_raw = np.stack([ch0, ch1], axis=0).astype(np.float64)
            label = VOCAB[vocab_index[0]]
            path = next_filename(out_dir, label)
            np.savez(path, emg=emg_raw, label=label, sample_rate=SAMPLE_RATE, actual_len=T)
            print(f"Saved: {path}  ({T} samples, {T / SAMPLE_RATE:.2f}s)  [{label} {repeat_count[0] + 1}/{REPEATS_PER_WORD}]")
            repeat_count[0] += 1
            if repeat_count[0] >= REPEATS_PER_WORD:
                repeat_count[0] = 0
                vocab_index[0] = (vocab_index[0] + 1) % len(VOCAB)
            print(f"Next: [{VOCAB[vocab_index[0]]}] ({repeat_count[0] + 1}/{REPEATS_PER_WORD}) — hold ` to record.")
        except Exception as e:
            print(f"Error on release: {e}")

    try:
        with SpikerBox(
            port=args.port,
            input_buffer_size=input_buffer_size,
            num_channels=SILENT_SPEECH_CHANNELS,
        ) as box:
            print(f"Next: [{VOCAB[vocab_index[0]]}] (1/{REPEATS_PER_WORD}) — hold ` to record.")
            reader = threading.Thread(target=reader_thread, args=(box,), daemon=True)
            reader.start()
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            if use_plot:
                import matplotlib.pyplot as plt
            try:
                while not stop_thread.is_set():
                    with lock:
                        b = [live_buffers[0].copy(), live_buffers[1].copy()]
                    if use_plot and fig is not None and axes is not None:
                        emg = np.stack(b, axis=0).astype(np.float64) - baseline[:, np.newaxis]
                        n = emg.shape[1]
                        if n >= MIN_PLOT_SAMPLES_FOR_BANDPASS:
                            emg = bandpass_emg(emg, SAMPLE_RATE, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ)
                        b = [emg[0], emg[1]]
                        _update_plot(axes, fig, b, VOCAB[vocab_index[0]] if recording else "Live")
                        plt.pause(0.001)
                    else:
                        time.sleep(0.01)
            except KeyboardInterrupt:
                pass
            finally:
                stop_thread.set()
                listener.stop()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_thread.set()
    sys.exit(0)


if __name__ == "__main__":
    main()
