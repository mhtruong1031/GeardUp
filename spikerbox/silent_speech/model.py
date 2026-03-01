"""
Lightweight 1D FCN for Silent Speech: 3-band FFT-decomposed EMG -> per-timestep class logits (temporal localization).
"""

import torch
import torch.nn as nn

from .config import NUM_CLASSES, FFT_NUM_BANDS


class SilentSpeechCNN(nn.Module):
    """
    1D fully convolutional network for subvocal EMG. Input (batch, channels=3, time_steps) — three
    FFT band signals (low, mid, high). Output (batch, num_classes, time_steps') — per-timestep
    logits so the model can localize the word within the window. Use max-over-time (MIL) for
    window-level classification.
    """

    def __init__(
        self,
        in_channels: int = FFT_NUM_BANDS,
        num_classes: int = NUM_CLASSES,
        base_filters: int = 32,
        kernel_size: int = 31,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.pool1 = nn.MaxPool1d(pool_size)

        self.conv2 = nn.Conv1d(base_filters, base_filters * 2, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(base_filters * 2)
        self.pool2 = nn.MaxPool1d(pool_size)

        self.conv3 = nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(base_filters * 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.25)
        self.conv_classifier = nn.Conv1d(base_filters * 4, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T) -> (B, num_classes, T')
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.conv_classifier(x)
        return x
