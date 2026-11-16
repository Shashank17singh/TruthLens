"""Waveform augmentations for robustness. Each can be switched off for the ablation.

Order follows the path a shared clip takes: speed change (a resampled or
re-timed upload), the room, the microphone's noise, then lossy compression.
Speed runs before the 4-second crop because it changes the length; the rest
run on the crop.

Noise is synthetic coloured noise, not MUSAN, so the pipeline needs no extra
download. Codecs go through ffmpeg, the one tool that encodes MP3 and Opus at
an exact bitrate.
"""

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile
import soxr
from scipy.signal import fftconvolve

CODEC_ARGUMENTS = {
    "mp3": ["-c:a", "libmp3lame", "-f", "mp3"],
    "opus": ["-c:a", "libopus", "-f", "ogg"],
}
DECAY_60_DB = np.log(1000.0)  # amplitude falls 60 dB when the exponent reaches this


def change_speed(wave: np.ndarray, factor: float, sample_rate: int) -> np.ndarray:
    """Kaldi-style speed perturbation: faster and higher by `factor`, length divided by it."""
    return soxr.resample(wave, sample_rate * factor, sample_rate).astype(np.float32)


def coloured_noise(length: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """Power spectrum proportional to 1 / f**exponent: 0 white, 1 pink, 2 brown."""
    spectrum = np.fft.rfft(rng.standard_normal(length))
    frequencies = np.fft.rfftfreq(length)
    frequencies[0] = frequencies[1]
    noise = np.fft.irfft(spectrum / frequencies ** (exponent / 2), n=length)
    # RMS, not standard deviation: pink and brown noise drift, and the mean carries power too.
    return (noise / (np.sqrt(np.mean(noise**2)) + 1e-12)).astype(np.float32)


def add_noise(wave: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = coloured_noise(wave.size, float(rng.choice([0.0, 1.0, 2.0])), rng)
    signal_power = float(np.mean(wave**2)) + 1e-12
    return (wave + noise * np.sqrt(signal_power / 10 ** (snr_db / 10))).astype(np.float32)


def room_impulse(rt60: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Direct path plus exponentially decaying noise that falls 60 dB in `rt60` seconds."""
    time = np.arange(int(rt60 * sample_rate)) / sample_rate
    tail = rng.standard_normal(time.size) * np.exp(-DECAY_60_DB * time / rt60) * 0.3
    tail[0] = 1.0
    return (tail / np.sqrt(np.sum(tail**2))).astype(np.float32)


def add_reverb(wave: np.ndarray, rt60: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    wet = fftconvolve(wave, room_impulse(rt60, sample_rate, rng))[: wave.size]
    peak = np.abs(wet).max()
    return (wet * (np.abs(wave).max() / peak) if peak > 0 else wet).astype(np.float32)


def _ffmpeg(arguments: list[str], data: bytes) -> bytes:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", *arguments]
    return subprocess.run(command, input=data, capture_output=True, check=True).stdout


def codec_roundtrip(wave: np.ndarray, codec: str, bitrate_kbps: int, sample_rate: int) -> np.ndarray:
    """Encode and decode through ffmpeg, keeping the input's length and alignment.

    The encoded file goes to disk, not a pipe: ffmpeg records the MP3 encoder
    delay in a header it can only write by seeking back to the start, and
    without it the decoded audio is shifted by about 1,100 samples.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Codec augmentation needs ffmpeg on PATH; install it or set augment.codec.enabled=false")
    pcm = io.BytesIO()
    soundfile.write(pcm, np.clip(wave, -1, 1), sample_rate, format="WAV", subtype="PCM_16")
    with tempfile.TemporaryDirectory() as folder:
        encoded = str(Path(folder) / f"clip.{codec}")
        _ffmpeg(["-f", "wav", "-i", "pipe:0", *CODEC_ARGUMENTS[codec], "-b:a", f"{bitrate_kbps}k", "-y", encoded], pcm.getvalue())
        decoded = _ffmpeg(["-i", encoded, "-ac", "1", "-ar", str(sample_rate), "-f", "wav", "pipe:1"], b"")
    result, _ = soundfile.read(io.BytesIO(decoded), dtype="float32")
    result = result[: wave.size]
    return np.pad(result, (0, wave.size - result.size)) if result.size < wave.size else result


class Augmenter:
    def __init__(self, config: dict, sample_rate: int, seed: int):
        self.config = config
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def set_random_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _fires(self, name: str) -> bool:
        step = self.config[name]
        return bool(step["enabled"]) and self.rng.random() < step["p"]

    def before_crop(self, wave: np.ndarray) -> np.ndarray:
        if self._fires("speed"):
            wave = change_speed(wave, float(self.rng.uniform(*self.config["speed"]["factors"])), self.sample_rate)
        return wave

    def __call__(self, wave: np.ndarray) -> np.ndarray:
        if self._fires("reverb"):
            wave = add_reverb(wave, float(self.rng.uniform(*self.config["reverb"]["rt60"])), self.sample_rate, self.rng)
        if self._fires("noise"):
            wave = add_noise(wave, float(self.rng.uniform(*self.config["noise"]["snr_db"])), self.rng)
        if self._fires("codec"):
            codec = self.config["codec"]
            wave = codec_roundtrip(wave, str(self.rng.choice(codec["codecs"])), codec["bitrate_kbps"], self.sample_rate)
        return wave
