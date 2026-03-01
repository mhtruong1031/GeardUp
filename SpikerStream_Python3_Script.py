import serial  # Note: pip install pyserial NOT pip install serial.
import numpy as np

"""
Streams data from Heart & Brain SpikerBox.

Written by Damian Lin (https://github.com/DamianJLin/), based off SpikerStream_Python3.ipynb, but as a bare-bones script
instead of notebook.
"""


def process_byte_data(b_data):
    """
    Converts the weird data from SpikerBox to list of amplitude measurements.

    Data from SpikerBox uses two bytes for a single unit of data. Hence returned array with be ~1/2 the size of the
    passed array.
    """
    data_raw = np.array(b_data)
    data_processed = np.zeros(0)

    i = 0
    while i < len(data_raw) - 1:

        if data_raw[i] > 127:
            # Found beginning of frame. Extract one sample from two bytes.
            int_processed = (np.bitwise_and(data_raw[i], 127)) * 128
            i += 1
            int_processed += data_raw[i]
            # Allocates, fills and returns new array. Likely inefficient.
            data_processed = np.append(data_processed, int_processed)

        i += 1

    return data_processed


# SpikerBox Specifications (Cannot be changed)
b_rate = 230400

# My Specifications (Can be changed)
# Set the correct port and baudrate, buffer size for the serial.
c_port = input('Which port?\n')
# Determines frequency of buffer filling up. 20000 = 1s.
input_buffer_size = 20000
# Set maximum time to read the buffer for. None for no timeout.
timeout = None


try:
    ser = serial.Serial(port=c_port, baudrate=b_rate)
    ser.set_buffer_size(rx_size=input_buffer_size)
    ser.timeout = timeout
except serial.serialutil.SerialException:
    raise Exception(f'Could not open port {c_port}.\nFind port from:\nDevice Manager > Ports (COM & LPT) [Windows]')


with ser as s:

    while True:

        # Read data from SpikerBox into a buffer of size input_buffer_size.
        byte_data = s.read(input_buffer_size)

        # Cast to list of ints.
        byte_data = [int(byte_data[i]) for i in range(len(byte_data))]

        # Process with above function.
        data = process_byte_data(byte_data)

        print(f'Data from serial: {str(data[0:4])[:-1]} ... ] (length {len(data)})')