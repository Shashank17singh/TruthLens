"""Audio frontends: log-mel, LFCC and CQT, computed inside the model.

Each one is a fixed filterbank written as a 1-D convolution and matrix
products. That exports to plain ONNX, so the served model computes features
with exactly the code it was trained with, as the image DCT does (D25).

Every frontend maps a waveform batch [B, samples] to [B, 1, bins, frames].
"""

import math

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_FFT = 512  # 32 ms at 16 kHz
WIN_LENGTH = 400  # 25 ms
HOP_LENGTH = 160  # 10 ms
LOG_FLOOR = 1e-6
KINDS = ("mel", "lfcc", "cqt")


def dft_kernels(n_fft: int, win_length: int) -> torch.Tensor:
    """Real and imaginary DFT rows times a Hann window centred in the frame: [2 * bins, 1, n_fft]."""
    window = np.zeros(n_fft)
    start = (n_fft - win_length) // 2
    window[start : start + win_length] = np.hanning(win_length + 1)[:-1]  # periodic Hann
    bins = n_fft // 2 + 1
    time = np.arange(n_fft)
    angle = 2 * np.pi * np.outer(np.arange(bins), time) / n_fft
    kernels = np.concatenate([np.cos(angle), -np.sin(angle)]) * window
    return torch.tensor(kernels, dtype=torch.float32).unsqueeze(1)


class PowerSpectrogram(nn.Module):
    """|STFT|^2 with centred frames, matching librosa.stft(center=True, pad_mode="reflect")."""

    def __init__(self, n_fft: int = N_FFT, win_length: int = WIN_LENGTH, hop_length: int = HOP_LENGTH):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.bins = n_fft // 2 + 1
        self.register_buffer("kernels", dft_kernels(n_fft, win_length), persistent=False)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        padded = F.pad(wave.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        spectrum = F.conv1d(padded, self.kernels, stride=self.hop_length)
        real, imaginary = spectrum[:, : self.bins], spectrum[:, self.bins :]
        return real.square() + imaginary.square()  # [B, bins, frames]


def linear_filterbank(sample_rate: int, n_fft: int, filters: int) -> np.ndarray:
    """Triangular filters evenly spaced in Hz: the LFCC counterpart of the mel filterbank."""
    bin_hz = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    edges = np.linspace(0, sample_rate / 2, filters + 2)
    lower, centre, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bin_hz - lower) / (centre - lower)
    falling = (upper - bin_hz) / (upper - centre)
    return np.maximum(0.0, np.minimum(rising, falling)).astype(np.float32)


def dct_matrix(inputs: int, outputs: int) -> torch.Tensor:
    """Orthonormal DCT-II, first `outputs` rows."""
    n = np.arange(inputs)
    k = np.arange(outputs)[:, None]
    basis = np.cos(np.pi / inputs * (n + 0.5) * k) * np.sqrt(2.0 / inputs)
    basis[0] /= np.sqrt(2.0)
    return torch.tensor(basis, dtype=torch.float32)


def deltas(features: torch.Tensor) -> torch.Tensor:
    """First difference over time with edge frames repeated: (x[t+1] - x[t-1]) / 2."""
    padded = torch.cat([features[..., :1], features, features[..., -1:]], dim=-1)
    return (padded[..., 2:] - padded[..., :-2]) / 2


class LogMel(nn.Module):
    def __init__(self, sample_rate: int, mels: int = 128):
        super().__init__()
        self.power = PowerSpectrogram()
        bank = librosa.filters.mel(sr=sample_rate, n_fft=N_FFT, n_mels=mels, fmin=20.0, fmax=sample_rate / 2)
        self.register_buffer("bank", torch.tensor(bank, dtype=torch.float32), persistent=False)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return torch.log(self.bank @ self.power(wave) + LOG_FLOOR).unsqueeze(1)


class Lfcc(nn.Module):
    """Linear-frequency cepstra with deltas, as in the ASVspoof LFCC baselines: 20 static + 20 delta + 20 delta-delta."""

    def __init__(self, sample_rate: int, filters: int = 70, coefficients: int = 20):
        super().__init__()
        self.power = PowerSpectrogram()
        self.register_buffer("bank", torch.tensor(linear_filterbank(sample_rate, N_FFT, filters)), persistent=False)
        self.register_buffer("dct", dct_matrix(filters, coefficients), persistent=False)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        static = self.dct @ torch.log(self.bank @ self.power(wave) + LOG_FLOOR)
        first = deltas(static)
        return torch.cat([static, first, deltas(first)], dim=1).unsqueeze(1)


class Cqt(nn.Module):
    """Constant-Q magnitude from librosa's wavelet filters applied as a strided convolution.

    Starts at 50 Hz: the 32.7 Hz default would need 8,000-sample filters, and
    speech carries little below 50 Hz. 12 bins per octave up to about 7.6 kHz.
    """

    def __init__(self, sample_rate: int, fmin: float = 50.0, bins_per_octave: int = 12, hop_length: int = HOP_LENGTH):
        super().__init__()
        top = 0.95 * sample_rate / 2
        bins = math.floor(bins_per_octave * math.log2(top / fmin))
        frequencies = librosa.cqt_frequencies(n_bins=bins, fmin=fmin, bins_per_octave=bins_per_octave)
        filters, _ = librosa.filters.wavelet(freqs=frequencies, sr=sample_rate, pad_fft=False)
        width = filters.shape[1] | 1  # odd width, so frames are centred on their sample
        filters = np.pad(filters, ((0, 0), (0, width - filters.shape[1])))
        kernels = np.concatenate([filters.real, filters.imag]).astype(np.float32)
        self.bins, self.hop_length, self.width = bins, hop_length, width
        self.register_buffer("kernels", torch.tensor(kernels).unsqueeze(1), persistent=False)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        padded = F.pad(wave.unsqueeze(1), (self.width // 2, self.width // 2))
        response = F.conv1d(padded, self.kernels, stride=self.hop_length)
        magnitude = response[:, : self.bins].square() + response[:, self.bins :].square()
        return torch.log(magnitude + LOG_FLOOR).unsqueeze(1)


def build_frontend(kind: str, sample_rate: int) -> nn.Module:
    if kind == "mel":
        return LogMel(sample_rate)
    if kind == "lfcc":
        return Lfcc(sample_rate)
    if kind == "cqt":
        return Cqt(sample_rate)
    raise ValueError(f"Unknown feature {kind!r}; choose from {KINDS}")
