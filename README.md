# Geard

Geard is a project for running ML inference on biosignal data (EEG/EMG) with a split architecture: **device code runs on your laptop**, while **Arduino Q** holds the firmware that will eventually run on an **Arduino Uno Q**.

---

## Repository layout

| Location | Role |
|----------|------|
| **Device (laptop)** | Python code for data handling, analysis, and model inference runs here. |
| **`arduino-q/`** | Code targeting the **Arduino Uno Q**, split into two runtimes (see below). |
| **`training/`** | Scripts used to train and evaluate models on data collected from the device. |

---

## Arduino Q (`arduino-q/`)

This directory contains firmware and related code for the **Arduino Uno Q**. It is organized into two runtimes:

### `main_runtime/`

Used **after the model is trained** and you want to run **inference** in production.

- **`MCU.cpp`** — Firmware that reads analog channels (EEG on A0/A1, EMG on A2/A3), uses `Arduino_Bridge` to expose `readAnalogChannels` via RPC, and communicates with the laptop (MPU).
- **`MCP.py`** — Laptop-side interface to the MCU (e.g. calling the bridge, sending/receiving data for inference).

Deploy this runtime when the pipeline is: **Arduino reads sensors → laptop runs the trained model → results used on device.**

### `training_runtime/`

Used **while you are gathering data** to train the model.

- **`MCU.cpp`** — Same sensor-reading firmware as `main_runtime`: reads EEG/EMG channels and exposes them over the Bridge so the laptop can log raw data for later training.

Use this runtime when you are **collecting datasets** that will be processed by the scripts in **`training/`**.

---

## Training (`training/`)

Scripts and model code for training and evaluation, run on the laptop using data collected via `arduino-q/training_runtime/`.

| File | Purpose |
|------|--------|
| **`Model.py`** | PyTorch model definition (e.g. for biosignal classification/regression). |
| **`train.py`** | Training script: load data, train the model, save checkpoints. |
| **`eval.py`** | Evaluation script: load a trained model and compute metrics on held-out data. |

Data is gathered with the Arduino running `training_runtime` and the laptop recording the RPC/bridge stream; the same preprocessing used in `arduino-q` (e.g. `pipelines/preprocessing.py`) can be applied before or during training.

---

## Laptop-side analysis (`arduino-q/`)

Code that runs on the laptop (device), not on the Arduino:

- **`analysis.py`** — Top-level analysis pipeline; uses `pipelines.preprocessing` to preprocess data before further analysis or inference.
- **`pipelines/preprocessing.py`** — Preprocessing for 2D sensor×time data (e.g. high-pass, low-pass, FFT stubs). Used for both data preparation and the main_runtime inference path.

---

## Workflow summary

1. **Data collection** — Flash `arduino-q/training_runtime/MCU.cpp` to the Arduino Uno Q, run laptop code to record sensor data via the Bridge.
2. **Training** — Use `training/train.py` (and `Model.py`) on the collected data; evaluate with `training/eval.py`.
3. **Deployment** — Flash `arduino-q/main_runtime/MCU.cpp`, run laptop-side code (`MCP.py` + analysis/preprocessing) to perform inference with the trained model on live sensor data.
