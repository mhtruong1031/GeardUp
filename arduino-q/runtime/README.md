# Runtime (training & main)

Single runtime for Arduino Uno Q: one MCU sketch and one MCP for both data collection (training) and live control (runtime). The MCU uses a **mode** switch to select behavior.

## MCU mode switch

In `MCU.cpp`, set at the top:

- **`#define MODE MODE_MAIN`** — Production/runtime: no Serial printing; use when running the 100 Hz control loop.
- **`#define MODE MODE_TRAINING`** — Data collection: `readAnalogChannels()` also prints readings to Serial for visibility.

Rebuild and reflash the sketch after changing `MODE`.

Other top-of-file options in `MCU.cpp`: `RUNTIME_RING_BUFFER_SIZE` (default 300), `EEG_MODE` (amp vs ML), motor pins (Motor 1: 9, 5, 6, 8; speed wheel: 3).

## Context (MCP & MCU)

- **MCP** (this folder): Python code that runs on the MPU or host. It talks to the MCU via the Bridge (RPC).
- **MCU** (`MCU.cpp`): Sketch that reads analog pins (A0–A3: EEG, EMG), keeps a ring buffer, and exposes RPCs: `readAnalogChannels`, `getRecentWindow`, `getConfig`, `setSteering`, `setSpeedWheel`.

## Wiring (quick reference)

| Function      | Pin  |
|---------------|------|
| EEG 1, EEG 2  | A0, A1 |
| EMG 1, EMG 2  | A2, A3 |
| Motor 1 (steering) | D9, D5, D6 (PWM), D8 (Enable) |
| Speed wheel   | D3 (PWM) |

See the main project [README](../../README.md) for the full wiring schematic and directions.

## Config (MCP.py)

Edit variables at the **top** of `MCP.py` instead of CLI:

- **Training**: `COLLECT_INTERVAL_SEC`, `OUTPUT_CSV`, `PREPROCESS`, `PRINT_SAMPLES`, `SOCKET_PATH`
- **Mode**: `RUN_MODE` = `"training"` or `"runtime"`
- **Runtime**: `RUNTIME_INTERVAL_SEC`, `EMG_NORM_WINDOW`, `EEG_AMP_WINDOW`, `VELOCITY_MAX`, `ACCEL_THROTTLE`, `ACCEL_BRAKE`, `FRICTION`, `EEG_DEADZONE`

Optional CLI overrides: `--output` / `-o` (CSV path), `--mode` (training | runtime).

## Usage

Scripts are written to run **from inside `runtime/`**; `pipelines/` is resolved via the parent `arduino-q` directory on `sys.path`.

From **inside** `arduino-q/runtime/`:

```bash
# Training: collect at 100 Hz, save to OUTPUT_CSV on Ctrl+C (config at top of MCP.py)
python MCP.py

# Override output path
python MCP.py -o my_data.csv

# Runtime: 100 Hz control loop (EMG -> steering, EEG -> throttle/brake -> speed wheel)
python MCP.py --mode runtime
```

From the `arduino-q` directory:

```bash
python -m runtime.MCP -o training_data.csv
python -m runtime.MCP --mode runtime
```

When running on the Arduino MPU (e.g. in App Lab), the script uses `arduino.app_utils.Bridge` if available; otherwise it uses the Unix socket and `msgpack` (see `bridge_client.py`).

## Flow

**Training**

1. MCU returns `"a0,a1,a2,a3"` when MCP calls `readAnalogChannels()` at the configured interval.
2. MCP records timestamps and the four values until Ctrl+C.
3. Data is converted to sensors × time, optionally preprocessed, and written to CSV (time × sensors).

**Runtime**

1. MCP runs a 100 Hz loop: each tick it calls `readAnalogChannels()` (one sample).
2. EMG (A2−A3) is normalized and sent to `setSteering()`; EEG drives throttle/brake (amp moving average or ML stub); velocity state is updated and sent to `setSpeedWheel()`.

## Files

| File | Role |
|------|------|
| `MCP.py` | Entrypoint: training (collect → preprocess → CSV) or runtime (100 Hz control loop). |
| `bridge_client.py` | Bridge client: `app_utils.Bridge` on device, or SocketBridge (msgpack) to arduino-router. |
| `MCU.cpp` | MCU sketch: analog read, ring buffer, motor PWM; RPCs via `Arduino_RouterBridge`. |
| `pipelines/preprocessing.py` | `PreprocessingPipeline.preprocess(data)` — sensors × time in/out. |

## Dependencies

- Python 3
- `numpy`
- `msgpack` (only when using SocketBridge)
- `scipy` (for preprocessing pipeline)

Install: `pip install -r requirements.txt` or `pip install numpy msgpack scipy`
