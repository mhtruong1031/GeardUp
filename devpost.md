## Inspiration

Many people with limited hand or arm mobility—due to spinal injury, ALS, stroke, or other conditions—miss out on experiences the rest of us take for granted, including **driving** and even **driving simulations**. We wanted to give them a way to control a car in a game. GeardUp! started from the question: *Can we use affordable biosignal hardware (EMG from a SpikerBox) to simulate driving and to communicate—without needing fine motor control or voice?*

## What it does

GeardUp! is a biosignal-based control and communication suite:

- **Keyboard control (driving simulation)** — Two EMG channels from a Human SpikerBox are streamed in real time, bandpass-filtered, and mapped to keyboard keys: **Channel 0 → 'd' (e.g. steer right), Channel 1 → 'a' (steer left), both active → 'w' (e.g. accelerate)**. This is aimed at **simulating driving for disabled users**: they can play driving games or use driving sims by flexing different muscles instead of using a keyboard or wheel. A short calibration (hold still for a few seconds) sets a per-channel baseline; an adaptive baseline and hysteresis reduce false triggers and keep control stable.

## How we built it

- **Hardware:** Human SpikerBox for 2-channel EMG (and optional EEG) at 5 kHz; Arduino Uno Q for motor/vehicle control.
- **Keyboard control pipeline:** Python streams EMG from the SpikerBox over serial, applies a 50–400 Hz bandpass, and computes a level metric (we use p95 of the filtered signal) per chunk. We use **percentile-based rest detection** (e.g. “at rest” when level ≤ 75th percentile of a sliding window) and update the baseline with an EMA only when at rest. Activation uses a configurable threshold above baseline with hysteresis (separate deactivate threshold) so keys don’t flicker. We simulate key presses with **pynput** (Ch0 → `d`, Ch1 → `a`, both → `w`), so any driving game or sim that uses these keys works without modification.
- **Calibration:** A separate step records a few seconds of rest-only EMG to compute per-channel baseline and noise standard deviation; this is saved and loaded at runtime for robust activation thresholds.

## Challenges we ran into

- **Baseline drift** — EMG baseline changes with posture and fatigue. We addressed this with an adaptive baseline (EMA updated only when “at rest”) and optional calibration so the system stays usable over time.
- **Rest detection** — Deciding when the user is “at rest” without a separate rest gesture was tricky. We used a sliding window of recent levels and a percentile cutoff so we don’t need a fixed absolute threshold.
- **False activations** — Single spikes or noise could trigger keys. Using the **p95** level metric instead of max, plus hysteresis (higher threshold to activate, lower to deactivate), and noise-based threshold scaling from calibration all helped.
- **Mapping both channels** — When both muscles fire (e.g. “go forward”), we map to a single key (`w`) instead of pressing both `a` and `d`, so the game receives a clear “accelerate” intent.

## Accomplishments that we're proud of

- **Real-time EMG-to-keyboard** for driving sims with no game-specific integration—any title that uses WASD works.
- **Calibration + adaptive baseline** so the same setup works across users and sessions without constant retuning.
- **Clear use case** for **simulating driving for disabled people** using only two EMG channels and a SpikerBox.
- **Unified project** that connects keyboard control (sims), silent speech (communication), and Arduino Q (hardware control) under one accessibility-focused vision.

## What we learned

- Rest detection via percentiles over a short history works well for EMG when you don’t have a dedicated “rest” gesture.
- Hysteresis and a robust level metric (p95) are essential for stable, flicker-free key output.
- Affordable hardware (SpikerBox) is enough to get usable, low-latency control for driving sims and communication with the right signal processing.

## What's next for GeardUp!

- **Tighter driving-sim integration:** Optional overlay or plugin for popular sims (e.g. steering wheel API) so EMG can map to analog steering and throttle instead of discrete keys.
- **More channels and gestures:** Support for more EMG channels and distinct gestures (e.g. brake, handbrake) for richer driving control.
- **User studies:** Partner with accessibility groups to test keyboard control and silent speech with people who have limited mobility or speech.
- **Arduino Q in the loop:** Use the same EMG processing ideas from keyboard control in the Arduino Uno Q runtime for real steering and throttle in a safe, controlled environment (e.g. go-kart or simulator rig).
