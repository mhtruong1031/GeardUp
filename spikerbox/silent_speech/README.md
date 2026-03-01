# Silent Speech Decoder

Subvocal EMG → small vocabulary. Two EMG channels from a Human SpikerBox.

- **Preset actions (no model):** Count activations per channel; e.g. 3 activations on channel 1 → TTS "yes", 3 on channel 2 → TTS "np". No training required.
- **CNN mode:** Preprocessing decomposes the bandpass-filtered signal into **3 FFT bands** (low/mid/high); 1D FCN with temporal localization for a small vocabulary (e.g. Yes, No, Help, Water). Requires calibration, collection, and training.

Run from the **repository root** (parent of `spikerbox/`).

## User use case: accessibility and assistive communication

This pipeline is aimed at **people who cannot speak aloud or have very limited speech or motor control** (e.g. due to ALS, stroke, spinal injury, or other conditions). If someone can still **subvocalize**—form words in the throat or mouth without making sound—or produce small muscle activity that correlates with intent, the SpikerBox can pick up that EMG from skin-surface electrodes. After calibration and training on a small set of words (e.g. Yes, No, Help, Water), the system can:

- **Decode silent or near-silent utterances** in real time and show the predicted word on screen.
- Optionally **speak the word aloud** via text-to-speech (`--tts` in inference) so caregivers or family can hear the user’s choice without the user having to vocalize.

Typical workflow for a disabled user or caregiver: run **calibration** once (user stays still), then **collect** a few repetitions per word, **train** a model, and use **inference** with TTS for day-to-day communication of a small vocabulary. This is intended as an assistive communication aid, not a medical device.

## Setup

```bash
pip install -r spikerbox/requirements.txt
```

## Commands (python3)

### List serial ports

```bash
python3 -m spikerbox.silent_speech.collect --list-ports
```

```bash
python3 -m spikerbox.silent_speech.calibrate --list-ports
```

### Calibration (60 s rest)

Do this once; stay still while it records. Saves a per-channel baseline used for collection and inference.

```bash
python3 -m spikerbox.silent_speech.calibrate --port /dev/cu.usbmodem101 --out data/silent_speech
```

Optional: `--duration 60`, `--buffer-size 2000`.

### Collect labelled data

Hold the backtick key **`** while you think-speak the prompted word; release to save. Variable-length segments are saved; no min/max length.

```bash
python3 -m spikerbox.silent_speech.collect --port /dev/cu.usbmodem101 --out data/silent_speech
```

Optional: `--buffer-size 2000`, `--no-plot`.

### Train

```bash
python3 -m spikerbox.silent_speech.train --data data/silent_speech --out models/silent_speech/silent_speech_cnn.pt
```

Optional: `--epochs 50`, `--batch-size 16`, `--lr 0.001`, `--val-ratio 0.2`, `--no-cuda`, `--no-bandpass`, `--no-normalize`.

### Preset actions (no model)

Count EMG activations per channel; when a channel hits the configured count, speak the phrase (e.g. 3 on ch1 → "yes", 3 on ch2 → "np"). Edit `config.py` → `PRESET_ACTIONS` to change mappings.

```bash
python3 -m spikerbox.silent_speech.preset_inference --port /dev/cu.usbmodem101
```

Optional: `--no-tts`, `--calibration data/silent_speech`, `--threshold 400`, `--refractory 0.35`, `--window 4`.

### Inference (CNN model)

```bash
python3 -m spikerbox.silent_speech.inference --port /dev/cu.usbmodem101 --model models/silent_speech/silent_speech_cnn.pt
```

Optional: `--interval 0.2`, `--buffer-size 2000`, `--tts`, `--debounce 2`, `--calibration data/silent_speech`, `--show-position`.

## Workflow

**Preset actions (simplest):** Run calibration once, then `preset_inference`; no collection or training. Trigger phrases by producing the right number of activations on each channel.

**CNN vocabulary:** 1. **Calibrate** (60 s, do nothing). 2. **Collect** (hold \`, think-speak word, release; repeat). 3. **Train** on `data/silent_speech`. 4. **Inference** (live stream → predicted word; optional TTS).

## Port

Replace `/dev/cu.usbmodem101` with your SpikerBox port (e.g. `/dev/ttyACM0` on Linux, `COM3` on Windows). Use `--list-ports` to see available ports.
