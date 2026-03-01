# Geard

Geard is a project for running ML inference on biosignal data (EEG/EMG) with a split architecture: **device code runs on your laptop** (or on the Arduino Uno Q's MPU), while the **Arduino Uno Q MCU** runs the firmware for analog input and motor control.

---

## Repository layout

| Location | Role |
|----------|------|
| **Device (laptop or MPU)** | Python code for data handling, analysis, and model inference. |
| **`arduino-q/`** | Code targeting the **Arduino Uno Q**: MCU sketch + MCP (Python) in `runtime/`. |
| **`training/`** | Scripts used to train and evaluate models on data collected from the device. |

---

## Arduino Q (`arduino-q/`)

Firmware and MCP live in **`arduino-q/runtime/`**.

### Modes

- **Training** — Collect EEG/EMG at 100 Hz; save CSV on Ctrl+C. Configure via variables at the top of `MCP.py` (interval, output path, preprocess, etc.).
- **Runtime** — 100 Hz control loop: EMG → steering (BLDC Motor 1), EEG → throttle/brake → velocity → speed wheel (PWM). Set `RUN_MODE = "runtime"` in `MCP.py` or run `python MCP.py --mode runtime`.

### MCU (`MCU.cpp`)

- **`#define MODE MODE_TRAINING`** — Data collection; `readAnalogChannels()` also prints to Serial.
- **`#define MODE MODE_MAIN`** — Production; no Serial printing. Use for runtime control.

Top-of-file options: `RUNTIME_RING_BUFFER_SIZE` (default 300), `EEG_MODE` (amp or ML), motor pin defines. RPCs: `readAnalogChannels`, `getRecentWindow`, `getConfig`, `setSteering`, `setSpeedWheel`.

### MCP (`MCP.py`)

- **Training**: Edit `COLLECT_INTERVAL_SEC`, `OUTPUT_CSV`, `PREPROCESS`, `PRINT_SAMPLES`, `SOCKET_PATH` at the top. Run `python MCP.py` (optional `-o file.csv`). Stop with Ctrl+C to write CSV.
- **Runtime**: Set `RUN_MODE = "runtime"` or use `--mode runtime`. Loop runs at 100 Hz: one sample per tick, steering from (EMG1−EMG2) normalized, throttle/brake from EEG (amp moving average or ML model stub), then `setSpeedWheel(velocity)`.

See **Wiring** and **Directions** below for hardware and step-by-step usage.

---

## Wiring

All analog and digital pins refer to the **Arduino Uno Q** (MCU). Use 3.3V logic for analog inputs unless your sensors are 5V-tolerant and the board allows it.

### Pin assignment

| Function | Pin | Notes |
|----------|-----|--------|
| EEG channel 1 | **A0** | Analog in, 12-bit ADC |
| EEG channel 2 | **A1** | Analog in, 12-bit ADC |
| EMG channel 1 | **A2** | Analog in, 12-bit ADC |
| EMG channel 2 | **A3** | Analog in, 12-bit ADC |
| Motor 1 (steering) Phase A | **9** | PWM, SimpleFOC shield |
| Motor 1 Phase B | **5** | PWM, SimpleFOC shield |
| Motor 1 Phase C | **6** | PWM, SimpleFOC shield |
| Motor 1 Enable | **8** | Digital out, HIGH = driver on |
| Speed wheel | **3** | PWM, DC or second motor |

### Wiring schematic (block diagram)

```
                    Arduino Uno Q (MCU)
    +----------------------------------------------------------+
    |  Analog inputs (12-bit, 0-3.3V)                          |
    |    A0 <---- EEG 1 (signal)    A1 <---- EEG 2 (signal)    |
    |    A2 <---- EMG 1 (signal)   A3 <---- EMG 2 (signal)    |
    |    GND ---- sensor grounds    3.3V ---- (if needed)      |
    +----------------------------------------------------------+
    |  Motor 1 (steering) - SimpleFOC shield                    |
    |    D9  ---> Phase A    D5 ---> Phase B    D6 ---> Phase C |
    |    D8  ---> Enable (HIGH = on)                          |
    |    Motor power: use shield VIN/Vbat, not 5V from MCU    |
    +----------------------------------------------------------+
    |  Speed wheel                                              |
    |    D3 (PWM) ---> DC motor driver or ESC input             |
    |    GND --------> driver ground                            |
    +----------------------------------------------------------+
```

### EEG sensors (A0, A1)

- Connect each EEG electrode amplifier output to the corresponding analog pin (A0, A1).
- Reference and ground: common reference to GND; ensure amplifier output is within 0–3.3 V (or use voltage divider / level shifter if needed).
- Typical setup: dry or gel electrodes → amplifier (e.g. instrumentation amp, bandpass) → A0/A1. Do not connect raw electrodes directly to the MCU.

### EMG sensors (A2, A3)

- Connect each EMG amplifier (envelope or raw, after appropriate gain) to A2 and A3.
- Same as EEG: keep signals within 0–3.3 V; use a common GND with the Arduino and the amplifier supply.

### Motor 1 (steering) — SimpleFOC shield

- **Pinout**: Pins **9, 5, 6** → shield Phase A, B, C; **Pin 8** → Enable (HIGH = driver on). This matches the typical SimpleFOC shield layout for Arduino (e.g. 3-phase driver using D9, D5, D6 and Enable on D8).
- **Firmware**: The current MCU code uses **open-loop PWM** (one or two phases driven by magnitude/direction), not the SimpleFOC library. So it is **compatible with the same pins** as the SimpleFOC board and will drive a BLDC connected through that shield, but it does **not** use FOC or the official SimpleFOC API. For sensorless FOC or full library features, integrate the [SimpleFOC](https://docs.simplefoc.com/) library (e.g. `BLDCMotor` + `BLDCDriver3PWM`) and call it from `setSteering()`.
- **Power**: Use the shield’s motor supply (VIN/Vbat), not the Arduino 5 V rail, for the motor.

### Speed wheel (pin 3)

- **Pin 3** (PWM) → input of a DC motor driver or ESC (e.g. throttle input). PWM duty = normalized speed (0–1).
- Ground the driver/ESC logic ground to Arduino GND.

---

## Directions

### 1. Data collection (training)

1. Wire EEG (A0, A1) and EMG (A2, A3) as above. Optionally connect motors for later; they stay off during collection if you do not call `setSteering`/`setSpeedWheel`.
2. In `arduino-q/runtime/MCU.cpp` set `#define MODE MODE_TRAINING`, then build and flash the sketch.
3. In `arduino-q/runtime/MCP.py` set `RUN_MODE = "training"` (default). Edit `COLLECT_INTERVAL_SEC`, `OUTPUT_CSV`, `PREPROCESS`, `PRINT_SAMPLES`, `SOCKET_PATH` as needed.
4. Run MCP from `arduino-q/runtime/`: `python MCP.py`. Optional: `python MCP.py -o my_data.csv`
5. Let it run; stop with **Ctrl+C**. CSV is written to `OUTPUT_CSV` (or `-o` path).

### 2. Training a model

1. Use the CSV(s) from step 1 with `training/` (e.g. `Model.py`, `train.py`, `eval.py`). Preprocessing in `arduino-q/pipelines/preprocessing.py` can be reused for consistency.

### 3. Runtime (live control)

1. Wire Motor 1 (steering) to pins 8, 9, 5, 6 and speed wheel to pin 3 as in the schematic.
2. In `MCU.cpp` set `#define MODE MODE_MAIN`. Set `EEG_MODE` to `EEG_MODE_AMP` or `EEG_MODE_ML` as needed. Reflash.
3. In `MCP.py` set `RUN_MODE = "runtime"` (or run with `--mode runtime`). Adjust `RUNTIME_INTERVAL_SEC`, `EMG_NORM_WINDOW`, `EEG_AMP_WINDOW`, velocity/friction constants if desired.
4. Run: `python MCP.py --mode runtime`. Steering follows (EMG1 − EMG2) normalized; throttle/brake from EEG (amp or ML); speed wheel reflects internal velocity. Stop with **Ctrl+C**.

### 4. EEG modes

- **Amp** (`EEG_MODE_AMP`): Throttle when EEG amplitude is above moving average, brake when below (dead zone in `MCP.py`: `EEG_DEADZONE`).
- **ML** (`EEG_MODE_ML`): MCP calls `getRecentWindow()` and will run a PyTorch model for binary throttle/brake once the model is loaded in code.

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

1. **Data collection** — Set `MODE` to `MODE_TRAINING` in `arduino-q/runtime/MCU.cpp`, flash it to the Arduino Uno Q, run `python MCP.py` from `arduino-q/runtime/`; stop with Ctrl+C to save CSV.
2. **Training** — Use `training/train.py` (and `Model.py`) on the collected data; evaluate with `training/eval.py`.
3. **Runtime** — Set `MODE` to `MODE_MAIN` in `arduino-q/runtime/MCU.cpp`, reflash; set `RUN_MODE = "runtime"` or run `python MCP.py --mode runtime` for the 100 Hz control loop (steering + speed wheel).
