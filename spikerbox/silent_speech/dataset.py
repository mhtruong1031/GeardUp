"""
Dataset loader for Silent Speech: load .npz files, optional preprocessing, PyTorch Dataset.
"""

from pathlib import Path

import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset

from .config import (
    LABEL_TO_IDX,
    SAMPLE_RATE,
    WINDOW_LEN,
    EMG_OFFSET,
    BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ,
    FFT_BAND_EDGES_HZ,
    load_calibration,
)


def bandpass_emg(emg: np.ndarray, sample_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Apply bandpass filter per channel. emg shape (channels, time)."""
    nyq = sample_rate / 2.0
    low = max(low_hz / nyq, 0.001)
    high = min(high_hz / nyq, 0.999)
    b, a = signal.butter(4, [low, high], btype="band")
    out = np.zeros_like(emg, dtype=np.float64)
    for c in range(emg.shape[0]):
        out[c] = signal.filtfilt(b, a, emg[c].astype(np.float64))
    return out


def fft_decompose_emg_3bands(
    emg: np.ndarray,
    sample_rate: float,
    band_edges_hz: list[float] | None = None,
) -> np.ndarray:
    """
    Decompose 2-channel EMG into 3 time-domain signals via FFT band splitting.
    Averages the two physical channels, then splits the spectrum into 3 bands
    (low, mid, high), IFFT back to time domain. Returns (3, T).
    """
    if band_edges_hz is None:
        band_edges_hz = FFT_BAND_EDGES_HZ
    # Combine channels: (2, T) -> (T,)
    x = np.asarray(emg, dtype=np.float64).mean(axis=0)
    n = len(x)
    X = np.fft.rfft(x)
    n_bins = len(X)
    bands = []
    for i in range(len(band_edges_hz) - 1):
        f_lo, f_hi = band_edges_hz[i], band_edges_hz[i + 1]
        # Bin indices: freq_k = k * sample_rate / n, so k = freq * n / sample_rate
        k_lo = max(0, int(f_lo * n / sample_rate))
        k_hi = min(n_bins, int(f_hi * n / sample_rate) + 1)
        X_band = np.zeros_like(X, dtype=np.complex128)
        X_band[k_lo:k_hi] = X[k_lo:k_hi]
        y_band = np.fft.irfft(X_band, n=n).astype(np.float64)
        bands.append(y_band)
    return np.stack(bands, axis=0)  # (3, T)


def to_fixed_window(emg: np.ndarray, target_len: int = WINDOW_LEN) -> np.ndarray:
    """
    Convert variable-length (channels, T) to fixed length for the model.
    T > target_len: take center target_len samples. T < target_len: zero-pad at end.
    """
    _, T = emg.shape
    if T >= target_len:
        start = (T - target_len) // 2
        return emg[:, start : start + target_len].astype(np.float64)
    out = np.zeros((emg.shape[0], target_len), dtype=np.float64)
    out[:, :T] = emg.astype(np.float64)
    return out


def normalize_per_channel(emg: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel (mean=0, std=1)."""
    out = np.zeros_like(emg, dtype=np.float64)
    for c in range(emg.shape[0]):
        x = emg[c].astype(np.float64)
        mean, std = x.mean(), x.std()
        if std < 1e-8:
            std = 1.0
        out[c] = (x - mean) / std
    return out


def scan_data_dir(data_dir: str) -> list[tuple[str, str]]:
    """
    Scan directory for .npz files. Returns list of (file_path, label).
    Expects filenames like yes_001.npz, no_002.npz, etc.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    pairs: list[tuple[str, str]] = []
    valid_labels = set(LABEL_TO_IDX.keys())
    for p in sorted(data_dir.glob("*.npz")):
        # label is the first part before _NNN
        name = p.stem
        parts = name.split("_")
        if len(parts) >= 1:
            label = parts[0].lower()
            if label in valid_labels:
                pairs.append((str(p), label))
    return pairs


class SilentSpeechDataset(Dataset):
    """
    PyTorch Dataset for Silent Speech .npz files.
    Loads (emg, label_idx) with optional bandpass and per-channel normalization.
    """

    def __init__(
        self,
        data_dir: str,
        apply_bandpass: bool = True,
        apply_normalize: bool = True,
        sample_rate: float = SAMPLE_RATE,
        low_hz: float = BANDPASS_LOW_HZ,
        high_hz: float = BANDPASS_HIGH_HZ,
    ) -> None:
        self.pairs = scan_data_dir(data_dir)
        self.apply_bandpass = apply_bandpass
        self.apply_normalize = apply_normalize
        self.sample_rate = sample_rate
        self.low_hz = low_hz
        self.high_hz = high_hz
        # Per-channel baseline: from calibration (mean) or fallback to EMG_OFFSET.
        cal = load_calibration(data_dir)
        self.baseline = cal if cal is not None else np.array([EMG_OFFSET, EMG_OFFSET], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label_str = self.pairs[idx]
        data = np.load(path, allow_pickle=True)
        emg = data["emg"]  # (2, T) — variable length
        if "sample_rate" in data:
            sr = float(data["sample_rate"])
        else:
            sr = self.sample_rate
        emg = to_fixed_window(emg.astype(np.float64), WINDOW_LEN)
        emg = emg - self.baseline[:, np.newaxis]
        if self.apply_bandpass:
            emg = bandpass_emg(emg, sr, self.low_hz, self.high_hz)
        # FFT decompose into 3 standard band signals (low, mid, high) -> (3, T)
        emg = fft_decompose_emg_3bands(emg, sr, FFT_BAND_EDGES_HZ)
        if self.apply_normalize:
            emg = normalize_per_channel(emg)
        label_idx = LABEL_TO_IDX[label_str]
        x = torch.from_numpy(emg).float()  # (3, T)
        return x, label_idx


def build_train_val_splits(
    data_dir: str,
    val_ratio: float = 0.2,
    stratify: bool = True,
    seed: int = 42,
    **dataset_kwargs,
) -> tuple[SilentSpeechDataset, SilentSpeechDataset]:
    """
    Build train and validation datasets from data_dir.
    Splits by utterance (file); optionally stratify by label.
    """
    pairs = scan_data_dir(data_dir)
    if not pairs:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    indices = np.arange(len(pairs))
    labels = [p[1] for p in pairs]

    if stratify:
        rng = np.random.default_rng(seed)
        idx_train, idx_val = [], []
        for label in set(labels):
            idx_label = [i for i, p in enumerate(pairs) if p[1] == label]
            rng.shuffle(idx_label)
            n_val = max(1, int(len(idx_label) * val_ratio)) if val_ratio > 0 else 0
            idx_val.extend(idx_label[:n_val])
            idx_train.extend(idx_label[n_val:])
        rng.shuffle(idx_train)
        rng.shuffle(idx_val)
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        n_val = int(len(indices) * val_ratio)
        idx_val = indices[:n_val].tolist()
        idx_train = indices[n_val:].tolist()

    train_pairs = [pairs[i] for i in idx_train]
    val_pairs = [pairs[i] for i in idx_val]

    class _SubsetDataset(SilentSpeechDataset):
        def __init__(self, pair_list: list, data_dir: str, **kw):
            self.pairs = pair_list
            self.apply_bandpass = kw.get("apply_bandpass", True)
            self.apply_normalize = kw.get("apply_normalize", True)
            self.sample_rate = kw.get("sample_rate", SAMPLE_RATE)
            self.low_hz = kw.get("low_hz", BANDPASS_LOW_HZ)
            self.high_hz = kw.get("high_hz", BANDPASS_HIGH_HZ)
            cal = load_calibration(data_dir)
            self.baseline = cal if cal is not None else np.array([EMG_OFFSET, EMG_OFFSET], dtype=np.float64)

        def __len__(self):
            return len(self.pairs)

    train_ds = _SubsetDataset(train_pairs, data_dir, **dataset_kwargs)
    val_ds = _SubsetDataset(val_pairs, data_dir, **dataset_kwargs)
    return train_ds, val_ds
