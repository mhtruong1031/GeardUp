"""
Real-time inference for Silent Speech Decoder: stream EMG from SpikerBox, run 1D CNN, output word (and optional TTS).

Usage:
  python -m spikerbox.silent_speech.inference --port /dev/cu.usbmodem101 --model models/silent_speech/silent_speech_cnn.pt
"""

import argparse
import time

import numpy as np
import torch

from spikerbox import SpikerBox

from .config import (
    IDX_TO_LABEL,
    SAMPLE_RATE,
    WINDOW_LEN,
    WINDOW_SEC,
    SILENT_SPEECH_CHANNELS,
    EMG_OFFSET,
    DEFAULT_DATA_DIR,
    load_calibration,
    BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ,
    FFT_BAND_EDGES_HZ,
)
from .dataset import bandpass_emg, normalize_per_channel, fft_decompose_emg_3bands
from .model import SilentSpeechCNN


def load_model(path: str, device: torch.device) -> SilentSpeechCNN:
    """Load trained SilentSpeechCNN from checkpoint."""
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    model = SilentSpeechCNN()
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_inference(
    port: str = "/dev/cu.usbmodem101",
    model_path: str = "models/silent_speech/silent_speech_cnn.pt",
    inference_interval_sec: float = 0.2,
    buffer_size: int = 2000,
    use_tts: bool = False,
    debounce_count: int = 2,
    confidence_threshold: float = 0.0,
    calibration_path: str | None = None,
    show_position: bool = False,
    show_confidence: bool = False,
) -> None:
    """
    Stream from SpikerBox, maintain ring buffer, run model at inference_interval_sec.
    Output when same class debounce_count times in a row and confidence >= confidence_threshold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Baseline: calibration mean (per channel) or fallback EMG_OFFSET.
    import os
    cal = None
    if calibration_path:
        cal = load_calibration(calibration_path)
    if cal is None:
        cal = load_calibration(os.path.dirname(model_path))
    if cal is None:
        cal = load_calibration(DEFAULT_DATA_DIR)
    baseline = cal if cal is not None else np.array([EMG_OFFSET, EMG_OFFSET], dtype=np.float64)

    # Ring buffer: list of two arrays, each of length WINDOW_LEN (overwritten in place).
    buffers = [np.zeros(WINDOW_LEN, dtype=np.float64), np.zeros(WINDOW_LEN, dtype=np.float64)]
    write_pos = [0, 0]  # next write index per channel

    last_inference_time = 0.0
    last_label = None
    debounce_accum = []

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

    print(f"Inference: port={port}, model={model_path}, interval={inference_interval_sec}s, debounce={debounce_count}, confidence>={confidence_threshold}")
    print("Ctrl+C to stop.\n")

    with SpikerBox(port=port, input_buffer_size=buffer_size, num_channels=SILENT_SPEECH_CHANNELS) as box:
        while True:
            data = box.run()
            now = time.perf_counter()

            if not isinstance(data, tuple) or len(data) != SILENT_SPEECH_CHANNELS:
                continue

            # Append to ring buffer (each channel).
            for ch, arr in enumerate(data):
                for s in arr:
                    buffers[ch][write_pos[ch] % WINDOW_LEN] = float(s)
                    write_pos[ch] += 1

            # Fixed sliding window: run inference once we have WINDOW_LEN samples.
            if write_pos[0] < WINDOW_LEN or write_pos[1] < WINDOW_LEN:
                continue
            if now - last_inference_time < inference_interval_sec:
                continue

            last_inference_time = now

            # Build fixed window: most recent WINDOW_LEN samples (handle wrap).
            emg = np.stack([
                np.roll(buffers[0], -write_pos[0] % WINDOW_LEN)[:WINDOW_LEN],
                np.roll(buffers[1], -write_pos[1] % WINDOW_LEN)[:WINDOW_LEN],
            ], axis=0).astype(np.float64) - baseline[:, np.newaxis]

            # Same preprocessing as training: bandpass -> FFT 3-band decomposition -> normalize.
            emg = bandpass_emg(emg, SAMPLE_RATE, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ)
            emg = fft_decompose_emg_3bands(emg, SAMPLE_RATE, FFT_BAND_EDGES_HZ)
            emg = normalize_per_channel(emg)

            x = torch.from_numpy(emg).float().unsqueeze(0).to(device)  # (1, 3, T)
            with torch.no_grad():
                out = model(x)  # (1, num_classes, T')
                logits = out.max(dim=2).values  # MIL: max over time
                probs = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item()
                # Localization: time within window where predicted class peaks (in seconds)
                t_peak = out[0, pred_idx, :].argmax().item()
                n_steps = out.shape[2]
                position_sec = (t_peak / max(1, n_steps)) * WINDOW_SEC

            debounce_accum.append(pred_idx)
            if len(debounce_accum) > debounce_count:
                debounce_accum.pop(0)
            debounce_ok = len(debounce_accum) == debounce_count and all(p == debounce_accum[0] for p in debounce_accum)
            if debounce_ok and confidence >= confidence_threshold:
                label = IDX_TO_LABEL[pred_idx]
                last_label = label
                parts = [label]
                if show_position:
                    parts.append(f"{position_sec:.2f}s")
                if show_confidence:
                    parts.append(f"{confidence:.2f}")
                if len(parts) > 1:
                    print(f"{parts[0]} ({', '.join(parts[1:])})")
                else:
                    print(label)
                if use_tts and tts_engine:
                    tts_engine.say(label)
                    tts_engine.runAndWait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Silent Speech inference from SpikerBox stream")
    parser.add_argument("--port", default="/dev/cu.usbmodem101", help="Serial port")
    parser.add_argument("--model", default="models/silent_speech/silent_speech_cnn.pt", help="Path to trained model .pt")
    parser.add_argument("--interval", type=float, default=0.2, help="Inference interval in seconds")
    parser.add_argument("--buffer-size", type=int, default=2000, help="Serial read buffer size (smaller = finer chunks)")
    parser.add_argument("--tts", action="store_true", help="Use text-to-speech for output")
    parser.add_argument("--debounce", type=int, default=2, help="Require same prediction N times before output")
    parser.add_argument("--calibration", default=None, help="Path to calibration dir or calibration.npz (default: next to model or data/silent_speech)")
    parser.add_argument("--show-position", action="store_true", help="Print time within window where word was localized (seconds)")
    parser.add_argument("--show-confidence", action="store_true", help="Print softmax confidence (0-1) with each prediction")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Only output when max class probability >= this (0-1); 0 = off")
    args = parser.parse_args()

    try:
        run_inference(
            port=args.port,
            model_path=args.model,
            inference_interval_sec=args.interval,
            buffer_size=args.buffer_size,
            use_tts=args.tts,
            debounce_count=args.debounce,
            confidence_threshold=args.confidence_threshold,
            calibration_path=args.calibration,
            show_position=args.show_position,
            show_confidence=args.show_confidence,
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
