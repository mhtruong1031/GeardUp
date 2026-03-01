"""
SpikerBox data collection: serial read and 14-bit frame decode.

Protocol references:
  - spikerboxliveexample.py
  - SpikerStream_Python3_Script.py (Heart & Brain SpikerBox streaming)
  - DOCS.md (Human SpikerBox: 2/3/4 channels @ 5 kHz)
"""

import time

import serial
import numpy as np

# Number of channels (1 = single-channel / Heart&Brain; 2 = one per jack; 4 = one per power clip).
# Human SpikerBox: 2 stereo jacks × 2 power clips each = 4 inputs; use 4 for a separate signal per clip.
NUM_CHANNELS = 2


# Escape sequences (BYB protocol): bytes inside message blocks must not be parsed as samples.
MSG_BLOCK_START = (0xFF, 0xFF, 0x01, 0x01, 0x80, 0xFF)
MSG_BLOCK_END = (0xFF, 0xFF, 0x01, 0x01, 0x81, 0xFF)


def _strip_message_blocks(data: list[int]) -> list[int]:
    """
    Remove message blocks (escape sequences) from the stream so 0xFF etc. are not
    interpreted as frame starts. Returns only bytes that are sample data.
    """
    result: list[int] = []
    n_start, n_end = len(MSG_BLOCK_START), len(MSG_BLOCK_END)
    i = 0
    while i < len(data):
        if i <= len(data) - n_start and tuple(data[i : i + n_start]) == MSG_BLOCK_START:
            i += n_start
            while i <= len(data) - n_end:
                if tuple(data[i : i + n_end]) == MSG_BLOCK_END:
                    i += n_end
                    break
                i += 1
            else:
                i = len(data)
        else:
            result.append(data[i])
            i += 1
    return result


def _read_arduino(ser: serial.Serial, size: int) -> list[int]:
    """Read size bytes from serial and return list of ints (0-255)."""
    data = ser.read(size)
    return [int(data[i]) for i in range(len(data))]


def _process_data(data: list[int]) -> np.ndarray:
    """
    Decode 14-bit frames: byte > 127 starts frame, two bytes per sample.
    Returns NumPy array of sample values (single channel).
    (Same protocol as process_byte_data in SpikerStream_Python3_Script.py.)
    """
    data_in = np.array(data)
    result = []
    i = 0
    while i < len(data_in) - 1:
        if data_in[i] > 127:
            intout = (np.bitwise_and(data_in[i], 127)) * 128
            i += 1
            intout += data_in[i]
            result = np.append(result, intout)
        i += 1
    return result


def _process_data_multichannel(
    data: list[int], num_channels: int
) -> tuple[np.ndarray, ...]:
    """
    Decode 14-bit frames with multiple samples per frame (Human SpikerBox).
    Frame: one byte with MSB=1 (frame start) then num_channels samples, 2 bytes each.
    Returns tuple of arrays (one per channel).
    """
    data_in = np.array(data)
    frame_len = num_channels * 2
    channels = [[] for _ in range(num_channels)]
    i = 0
    while i <= len(data_in) - frame_len:
        if data_in[i] > 127:
            for c in range(num_channels):
                high = np.bitwise_and(data_in[i + c * 2], 127)
                low = data_in[i + c * 2 + 1]
                sample = high * 128 + low
                channels[c].append(sample)
            i += frame_len
        else:
            i += 1
    return tuple(np.array(ch) for ch in channels)


class SpikerBox:
    """
    Initializes serial connection and config for SpikerBox data collection.
    Use run() to read one chunk of processed samples (one data point).

    References: spikerboxliveexample.py, SpikerStream_Python3_Script.py.
    Baud rate 230400 per SpikerBox specs (SpikerStream_Python3_Script.py).
    """

    def __init__(
        self,
        port: str = "/dev/cu.usbmodem101",
        baudrate: int = 230400,
        input_buffer_size: int = 10000,
        num_channels: int | None = None,
    ) -> None:
        self._num_channels = num_channels if num_channels is not None else NUM_CHANNELS
        self._ser = serial.Serial(port=port, baudrate=baudrate)
        self._ser.timeout = input_buffer_size / 20000.0
        self._input_buffer_size = input_buffer_size
        if self._num_channels > 1:
            self._ser.write(f"c:{self._num_channels};".encode())
            time.sleep(0.1)
            self._ser.reset_input_buffer()

    def run(self) -> np.ndarray | tuple[np.ndarray, ...]:
        """Read one chunk, decode frames. Returns 1D array (1 ch) or tuple of arrays (2+ ch)."""
        raw = _read_arduino(self._ser, self._input_buffer_size)
        raw = _strip_message_blocks(raw)
        if self._num_channels == 1:
            return _process_data(raw)
        return _process_data_multichannel(raw, self._num_channels)

    def close(self) -> None:
        """Flush and close the serial port."""
        try:
            self._ser.flushInput()
            self._ser.flushOutput()
        finally:
            self._ser.close()

    def __enter__(self) -> "SpikerBox":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
