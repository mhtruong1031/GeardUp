"""
Live data printing from SpikerBox. Run until Ctrl+C.

From repo root: python -m spikerbox.main [port] [buffer_size]
Example: python -m spikerbox.main /dev/ttyACM0 10000

List available serial ports: python -m spikerbox.main --list-ports
"""

import sys

import numpy as np
import serial.tools.list_ports

from spikerbox import NUM_CHANNELS, SpikerBox

# Set to True for live plot, False for print-only.
plot = True


def list_ports() -> None:
    """Print available serial ports (e.g. to find SpikerBox)."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        return
    for p in ports:
        print(f"  {p.device}\t{p.description or '(no description)'}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--list-ports":
        list_ports()
        return

    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbmodem101"
    input_buffer_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    # Human SpikerBox: 5 kHz per channel for 2+ ch; single-channel examples use 10–20 kHz
    sample_rate = 5000.0 if NUM_CHANNELS >= 2 else 10000.0
    max_plot_seconds = 10.0
    max_samples = int(max_plot_seconds * sample_rate)

    if plot:
        import matplotlib.pyplot as plt
        plt.ion()
        if NUM_CHANNELS >= 2:
            fig, axes = plt.subplots(NUM_CHANNELS, 1, sharex=True)
            if NUM_CHANNELS == 1:
                axes = [axes]
            data_buffers = [np.array([]) for _ in range(NUM_CHANNELS)]
        else:
            fig, ax = plt.subplots()
            axes = [ax]
            data_buffers = [np.array([])]
        for a in axes:
            a.set_xlabel("time [s]")
        fig.show()

    mode = "plot" if plot else "print"
    print(f"SpikerBox live {mode} (port={port}, buffer={input_buffer_size}, channels={NUM_CHANNELS}). Ctrl+C to stop.")
    with SpikerBox(port=port, input_buffer_size=input_buffer_size, num_channels=NUM_CHANNELS) as box:
        while True:
            data = box.run()
            if plot:
                if isinstance(data, tuple):
                    for ch, arr in enumerate(data):
                        data_buffers[ch] = np.append(data_buffers[ch], arr)
                        if len(data_buffers[ch]) > max_samples:
                            data_buffers[ch] = data_buffers[ch][-max_samples:]
                    n = len(data_buffers[0])
                else:
                    data_buffers[0] = np.append(data_buffers[0], data)
                    if len(data_buffers[0]) > max_samples:
                        data_buffers[0] = data_buffers[0][-max_samples:]
                    n = len(data_buffers[0])
                if n > 0:
                    t = np.arange(n) / sample_rate
                    for ch, ax in enumerate(axes):
                        ax.clear()
                        ax.set_xlim(max(0, t[-1] - max_plot_seconds), t[-1])
                        ax.set_ylabel(f"Ch{ch + 1}")
                        ax.plot(t, data_buffers[ch])
                    axes[-1].set_xlabel("time [s]")
                    fig.canvas.draw()
                plt.pause(0.001)
            else:
                print(data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
