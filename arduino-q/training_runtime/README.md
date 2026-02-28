# Training runtime

Collects analog signals from the Arduino Uno Q MCU pins, runs the preprocessing pipeline, and saves data as CSV (format: **time × sensors**).

## Context (MCP & MCU)

- **MCP** (this folder): Python code that runs on the MPU or host. It talks to the MCU via the Bridge (RPC). See `arduino_uno_q_knowledge_base_and_playground` for Bridge usage and examples.
- **MCU** (`MCU.cpp`): Sketch that reads analog pins and exposes `readAnalogChannels()` over the Bridge. Pins used: A0, A1 (EEG), A2, A3 (EMG) — 12-bit ADC.

## Flow

1. **MCU** reads analog pins (A0–A3) and returns a comma-separated string `"a0,a1,a2,a3"` when the MPU calls `readAnalogChannels()`.
2. **MCP** calls the Bridge at a fixed interval for a given duration, recording timestamps and the four values.
3. Raw data (samples × sensors) is converted to **sensors × time** and passed to `pipelines.preprocessing.PreprocessingPipeline.preprocess()`.
4. The result is written to a CSV with columns: `time`, `sensor_0`, … (one row per time step).

## Usage

From the `arduino-q` directory (so `pipelines` and `training_runtime` are importable):

```bash
# Collect 10 s at 100 Hz, save to training_data.csv (with preprocessing)
python -m training_runtime.MCP --duration 10 --interval 0.01 --output training_data.csv

# Raw data only (no preprocessing)
python -m training_runtime.MCP --duration 5 --no-preprocess -o raw.csv

# Custom Bridge socket (when using arduino-router)
python -m training_runtime.MCP --socket /var/run/arduino-router.sock --output data.csv
```

When running on the Arduino MPU (e.g. in App Lab), the script uses `arduino.app_utils.Bridge` if available; otherwise it uses the Unix socket and `msgpack` (see `bridge_client.py`).

## Files

| File | Role |
|------|------|
| `MCP.py` | Entrypoint: collect → preprocess → CSV (time × sensors). |
| `bridge_client.py` | Bridge client: `app_utils.Bridge` on device, or SocketBridge (msgpack) to arduino-router. |
| `MCU.cpp` | MCU sketch: `readAnalogChannels()` reading A0–A3, registered with `Arduino_Bridge`. |
| `pipelines/preprocessing.py` | `PreprocessingPipeline.preprocess(data)` — input `sensors × time`, output same shape (stub by default). |

## Dependencies

- Python 3
- `numpy`
- `msgpack` (only when using SocketBridge, i.e. when not on device with `arduino.app_utils`)

Install: `pip install numpy msgpack`
