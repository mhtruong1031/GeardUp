# Geard

Geard is a project for running ML inference on biosignal data (EEG/EMG) with a split architecture: **device code runs on your laptop**, while **Arduino Q** holds the firmware that will eventually run on an **Arduino Uno Q**.

---

## Repository layout

| Location | Role |
|----------|------|
| **Device (laptop)** | Python code for data handling, analysis, and model inference runs here. |
| **`arduino-q/`** | Code targeting the **Arduino Uno Q**, with a single runtime (see below). |
| **`training/`** | Scripts used to train and evaluate models on data collected from the device. |

---

## Arduino Q (`arduino-q/`)

This directory contains firmware and related code for the **Arduino Uno Q**, in a single **`runtime/`** folder.

### `arduino-q/runtime/`

One MCU sketch and one MCP serve both **data collection** (training) and **inference** (main). A **mode** switch in the MCU selects behavior:

- **`MCU.cpp`** — Firmware that reads analog channels (EEG on A0/A1, EMG on A2/A3), uses `Arduino_RouterBridge` to expose `readAnalogChannels` via RPC. At the top of the file, set **`#define MODE MODE_TRAINING`** for data collection (readings also printed to Serial) or **`#define MODE MODE_MAIN`** for inference/production (no Serial printing). Reflash after changing.
- **`MCP.py`** and **`bridge_client.py`** — Laptop-side Bridge client and data-collection entrypoint; used for both recording training data and, with analysis/preprocessing, for inference on live sensor data.

- **Data collection**: Set MODE to **MODE_TRAINING**, flash `arduino-q/runtime/MCU.cpp`, run MCP to record sensor data via the Bridge.
- **Deployment**: Set MODE to **MODE_MAIN**, reflash, then run laptop-side code (MCP + analysis/preprocessing) to perform inference with the trained model.

---

## Training (`training/`)

Scripts and model code for training and evaluation, run on the laptop using data collected via `arduino-q/runtime/`.

| File | Purpose |
|------|--------|
| **`Model.py`** | PyTorch model definition (e.g. for biosignal classification/regression). |
| **`train.py`** | Training script: load data, train the model, save checkpoints. |
| **`eval.py`** | Evaluation script: load a trained model and compute metrics on held-out data. |

Data is gathered with the Arduino running `runtime` (with MODE set to MODE_TRAINING) and the laptop recording the RPC/bridge stream; the same preprocessing used in `arduino-q` (e.g. `pipelines/preprocessing.py`) can be applied before or during training.

---

## Laptop-side analysis (`arduino-q/`)

Code that runs on the laptop (device), not on the Arduino:

- **`analysis.py`** — Top-level analysis pipeline; uses `pipelines.preprocessing` to preprocess data before further analysis or inference.
- **`pipelines/preprocessing.py`** — Preprocessing for 2D sensor×time data (e.g. high-pass, low-pass, FFT stubs). Used for both data preparation and the runtime inference path.

---

## Workflow summary

1. **Data collection** — Set `MODE` to `MODE_TRAINING` in `arduino-q/runtime/MCU.cpp`, flash it to the Arduino Uno Q, run laptop code (e.g. `python -m runtime.MCP`) to record sensor data via the Bridge.
2. **Training** — Use `training/train.py` (and `Model.py`) on the collected data; evaluate with `training/eval.py`.
3. **Deployment** — Set `MODE` to `MODE_MAIN` in `arduino-q/runtime/MCU.cpp`, reflash, then run laptop-side code (MCP + analysis/preprocessing) to perform inference with the trained model on live sensor data.
