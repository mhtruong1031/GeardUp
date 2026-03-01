"""
EEG/EMG preprocessing pipeline.

Input: 2D array [num_sensors, num_timepoints]
Channel layout: rows 0-1 = EEG, rows 2-3 = EMG (per MCU.cpp: A0/A1 EEG, A2/A3 EMG)
"""

import numpy as np
from scipy import signal


class Preprocess:
    """
    Preprocessing for biosignal data: filters, EMG envelope, EEG FFT.
    """

    def __init__(
        self,
        sample_rate_hz: float = 250.0,
        eeg_channels: tuple = (0, 1),
        emg_channels: tuple = (2, 3),
        eeg_highpass_hz: float = 0.5,
        eeg_lowpass_hz: float = 50.0,
        emg_highpass_hz: float = 10.0,
        emg_lowpass_hz: float = 450.0,
        emg_envelope_lowpass_hz: float = 5.0,
        filter_order: int = 4,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.eeg_channels = list(eeg_channels)
        self.emg_channels = list(emg_channels)
        self.eeg_highpass_hz = eeg_highpass_hz
        self.eeg_lowpass_hz = eeg_lowpass_hz
        self.emg_highpass_hz = emg_highpass_hz
        self.emg_lowpass_hz = emg_lowpass_hz
        self.emg_envelope_lowpass_hz = emg_envelope_lowpass_hz
        self.filter_order = filter_order

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Run full preprocessing pipeline.

        Input: 2D array [num_sensors, num_timepoints]
        Output: Preprocessed 2D array. EEG rows = FFT magnitudes, EMG rows = envelope.
        """
        data = np.atleast_2d(np.asarray(data, dtype=float))
        n_sensors, n_samples = data.shape

        out = np.zeros_like(data)

        # EEG: bandpass + FFT magnitude spectrum
        for i in self.eeg_channels:
            if i < n_sensors:
                x = data[i]
                x = self._bandpass_filter(x, self.eeg_highpass_hz, self.eeg_lowpass_hz)
                out[i] = self._fft_magnitude(x)

        # EMG: bandpass + envelope (rectify + low-pass)
        for i in self.emg_channels:
            if i < n_sensors:
                x = data[i]
                x = self._bandpass_filter(x, self.emg_highpass_hz, self.emg_lowpass_hz)
                out[i] = self._emg_envelope(x)

        return out

    def _bandpass_filter(self, x: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
        """Zero-phase bandpass filter (high-pass + low-pass)."""
        nyq = 0.5 * self.sample_rate_hz
        low_norm = max(low_hz / nyq, 0.001)
        high_norm = min(high_hz / nyq, 0.99)
        b, a = signal.butter(self.filter_order, [low_norm, high_norm], btype="band")
        return signal.filtfilt(b, a, x)

    def _high_pass_filter(self, x: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """Zero-phase high-pass filter."""
        nyq = 0.5 * self.sample_rate_hz
        cutoff_norm = min(cutoff_hz / nyq, 0.99)
        b, a = signal.butter(self.filter_order, cutoff_norm, btype="high")
        return signal.filtfilt(b, a, x)

    def _low_pass_filter(self, x: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """Zero-phase low-pass filter."""
        nyq = 0.5 * self.sample_rate_hz
        cutoff_norm = min(cutoff_hz / nyq, 0.99)
        b, a = signal.butter(self.filter_order, cutoff_norm, btype="low")
        return signal.filtfilt(b, a, x)

    def _emg_envelope(self, x: np.ndarray) -> np.ndarray:
        """Full-wave rectification + linear envelope (low-pass on rectified signal)."""
        rectified = np.abs(x)
        envelope = self._low_pass_filter(rectified, self.emg_envelope_lowpass_hz)
        return envelope

    def _fft_magnitude(self, x: np.ndarray) -> np.ndarray:
        """
        FFT magnitude spectrum. Returns same length as input (rfft + pad back to T).
        For shorter feature vectors, consider downsampling or taking only low bins.
        """
        spectrum = np.fft.rfft(x)
        magnitudes = np.abs(spectrum)
        # Map back to time-domain length: rfft yields n//2+1 points
        # Pad magnitudes to match input length for consistent 2D output shape
        n = len(x)
        n_out = len(magnitudes)
        if n_out < n:
            padded = np.zeros(n)
            padded[:n_out] = magnitudes
        else:
            padded = magnitudes[:n]
        return padded

    # Expose helpers for direct use
    def high_pass_filter(self, data: np.ndarray, cutoff_hz: float | None = None) -> np.ndarray:
        """Apply high-pass filter. If data is 2D, filters each row."""
        cutoff = cutoff_hz if cutoff_hz is not None else self.eeg_highpass_hz
        data = np.atleast_2d(data)
        return np.array([self._high_pass_filter(row, cutoff) for row in data])

    def low_pass_filter(self, data: np.ndarray, cutoff_hz: float | None = None) -> np.ndarray:
        """Apply low-pass filter. If data is 2D, filters each row."""
        cutoff = cutoff_hz if cutoff_hz is not None else self.eeg_lowpass_hz
        data = np.atleast_2d(data)
        return np.array([self._low_pass_filter(row, cutoff) for row in data])

    def fast_fourier_transform(self, data: np.ndarray) -> np.ndarray:
        """Return FFT magnitude spectrum for each row of 2D data."""
        data = np.atleast_2d(data)
        return np.array([self._fft_magnitude(row) for row in data])


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Running mean of 1D array with given window size (full only where enough samples)."""
    x = np.asarray(x, dtype=float)
    if window <= 0 or len(x) == 0:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


class PreprocessingPipeline:
    """
    Legacy wrapper for backwards compatibility. Delegates to Preprocess.
    """

    def __init__(self, sample_rate_hz: float = 250.0, **kwargs):
        self._preprocess = Preprocess(sample_rate_hz=sample_rate_hz, **kwargs)

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Input: 2D array [sensors, timepoints]. Returns preprocessed data."""
        return self._preprocess.run(data)

    def high_pass_filter(self, data: np.ndarray, cutoff_hz: float | None = None) -> np.ndarray:
        return self._preprocess.high_pass_filter(data, cutoff_hz)

    def low_pass_filter(self, data: np.ndarray, cutoff_hz: float | None = None) -> np.ndarray:
        return self._preprocess.low_pass_filter(data, cutoff_hz)

    def fast_fourier_transform(self, data: np.ndarray) -> np.ndarray:
        return self._preprocess.fast_fourier_transform(data)
