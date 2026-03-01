# Human SpikerBox – Documentation (from Backyard Brains)

Scraped from [SpikerBox USB Communication Guide](https://docs.backyardbrains.com/software/spike-recorder/usb-communication-guide) (BYB docs, version R7, April 2024).

---

## Human SpikerBox hardware

- **USB**: Composite device (CDC ACM + vendor-specific for iOS).
- **Vendor ID**: 0x2E73 (Backyard Brains)  
- **Product ID**: 0x0004 (Human SpikerBox)
- **Channels**: **4 @ 5 kHz**, **3 @ 5 kHz**, or **2 @ 5 kHz** (configurable).
- **Serial baud rate**: Any (e.g. 230400 used in examples).
- **Expansion port**: Reaction timer, Reflex hammer, Game controller, etc.

### Physical inputs: 2 stereo jacks, each with 2 power clips + 1 ground

The Human SpikerBox has **2 stereo (TRS) jacks**. Each jack has **2 signal (power) clips and 1 common ground clip**. So there are **4 separate signal inputs** in total (2 per jack).

- **2-channel mode**: The device sends 2 streams. Typically these are one channel per jack (e.g. Ch1 = jack 1, Ch2 = jack 2), and each channel may be the **difference** of the two clips on that jack (bipolar derivation) or one clip referenced to ground—firmware-dependent.
- **4-channel mode**: The device sends 4 streams. Each stream corresponds to **one of the 4 power clips** (one signal per clip, referenced to common ground). This is how you get a **separate signal from each power clip**.

**To get a separate signal from each power clip:** set the device to 4-channel mode and use `NUM_CHANNELS = 4` in code. Then Ch1–Ch4 in the plot correspond to the four clip inputs (exact order—which channel is which jack and which clip—is device-specific; check the hardware manual or label, or identify by moving one clip and watching which plot changes).

---

## SpikerBox custom protocol (sample stream)

- Stream is a **byte stream divided into frames**.
- **Each frame = one sample per recording channel** (e.g. 2 channels → 2 samples per frame = 4 bytes).
- **Sample**: 10–14 bit value, **two bytes per sample**.
- **Frame start**: The **MSB of a byte is the frame flag**. If this bit is 1, that byte is the **start of a frame** (and the first byte of the first sample in that frame).
- So: byte ≥ 128 → start of frame; then decode two bytes per sample; for 2 channels, one frame = 4 bytes (ch0 high, ch0 low, ch1 high, ch1 low).

---

## Escape sequences (messages in the stream)

SpikerBox can embed messages in the same stream. Host must detect these and skip them when parsing samples.

- **Start of message block**: `0xFF 0xFF 0x01 0x01 0x80 0xFF`
- **End of message block**: `0xFF 0xFF 0x01 0x01 0x81 0xFF`

---

## Useful host commands (Human SpikerBox)

- **Hardware type**: Send `b:;` → reply `HWT:HUMANSB;`
- **Firmware / hardware / type**: Send `?:;` → reply `FWV:...;HWT:HUMANSB;HWV:...;`
- **Gain (per channel 1 or 2)**  
  - High gain on: `gainon:1;` or `gainon:2;`  
  - High gain off: `gainoff:1;` or `gainoff:2;`
- **High-pass filter (per channel 1 or 2)**  
  - Higher cutoff: `hpfon:1;` or `hpfon:2;`  
  - Lower cutoff: `hpfoff:1;` or `hpfoff:2;`
- **Expansion board**: Send `board:;` → reply `BRD:<number>;` (e.g. 0=events, 1=two analog ch, 4=Reflex Hammer, 5=Game Controller)
- **P300 stimulation**: `stimon:;` / `stimoff:;`, query with `p300?:;`
- **P300 audio**: `sounon:;` / `sounoff:;`, query with `sound?:;`

---

## Implementation notes for this repo

- **Channel count**: Set `NUM_CHANNELS` in `spikerbox/spikerbox.py` to match the device mode. Use **4** for one signal per power clip (4 separate traces); use **2** for one channel per jack (stereo).
- On connect, the code sends `c:N;` (e.g. `c:4;`) to request N-channel mode. The official BYB guide lists this command only for Shield devices; Human SpikerBox may ignore it. If you still see only 2 distinct traces, try setting 4-channel mode in the Spike Recorder app (Config / gear icon) before using this code, or confirm the device is in 4-channel mode.
- Current `_process_data()` decodes **one channel**; `_process_data_multichannel()` decodes **N channels** per frame (N = 2, 3, or 4). Each plotted channel is one stream from the device. Frame layout: one byte with MSB=1 (frame start + high 7 bits of Ch0), then Ch0 low, Ch1 high, Ch1 low, … so frame length = `num_channels * 2` bytes.
- **Escape sequences**: Raw bytes are passed through `_strip_message_blocks()` before decoding. Message blocks (`0xFF 0xFF 0x01 0x01 0x80 ... 0x81 0xFF`) are removed so that `0xFF` and other message bytes are never interpreted as frame starts or sample data. Without this, stream alignment can be corrupted and channels can appear duplicated or wrong.
