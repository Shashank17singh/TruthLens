"""Perturbations and attacks for robustness tests on image, audio and text detectors.

No results are stored here; apply these to a trained detector to measure it.
- Images: JPEG compression (90/70/50), downscale-resize (0.5x), Gaussian blur,
  screenshot-style re-encoding, and social-media style compression pipelines.
- Audio: MP3 codec compression (128/64 kbps), additive noise at SNR 20/10 dB,
  and telephonic phone-band (300 Hz - 3400 Hz) filtering.
- Text: Paraphrasing, synonym swaps, simulated typing typos, and mixing human
  and AI paragraphs.
- Adversarial attack & defense (Image): Fast Gradient Sign Method (FGSM) and
  Projected Gradient Descent (PGD) attacks, and adversarial training defense comparison.
"""

from __future__ import annotations

import io
import logging
import os
import random
import shutil
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from scipy import signal
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TruthLens.Robustness")


# ============================================================================
# 1. IMAGE PERTURBATIONS
# ============================================================================

class ImagePerturbations:
    """Realistic visual perturbations simulating social networks and re-encoding."""

    @staticmethod
    def jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
        """Apply lossy JPEG compression and decompression at specific quality."""
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    @staticmethod
    def resize_downscale(img: Image.Image, scale: float = 0.5) -> Image.Image:
        """Downscale by factor and upscale back using bicubic interpolation."""
        orig_w, orig_h = img.size
        new_w = max(16, int(orig_w * scale))
        new_h = max(16, int(orig_h * scale))
        down = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        return down.resize((orig_w, orig_h), resample=Image.Resampling.BICUBIC)

    @staticmethod
    def gaussian_blur(img: Image.Image, radius: float = 2.0) -> Image.Image:
        """Apply Gaussian blur filtering."""
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def screenshot_reencode(img: Image.Image) -> Image.Image:
        """Simulate screenshotting: adaptive 256-color palette quantization + PNG compression."""
        quantized = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT).convert("RGB")
        buffer = io.BytesIO()
        quantized.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    @staticmethod
    def social_media_compression(img: Image.Image, seed: int = 42) -> Image.Image:
        """Simulate aggressive messaging/social-media compression (e.g. WhatsApp / Twitter).

        Pipeline: Downscale 0.75x -> slight additive sensor noise -> JPEG 65 -> mild contrast drop.
        """
        rng = np.random.default_rng(seed)
        w, h = img.size
        # 1. Resample down to 75%
        rescaled = img.resize((int(w * 0.75), int(h * 0.75)), resample=Image.Resampling.BILINEAR)
        arr = np.array(rescaled, dtype=np.float32)

        # 2. Add subtle Gaussian noise (sensor noise before upload)
        noise = rng.normal(0, 3.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        intermediate = Image.fromarray(arr)

        # 3. Slight contrast degradation (0.95)
        enhancer = ImageEnhance.Contrast(intermediate)
        contrast_adj = enhancer.enhance(0.95)

        # 4. Severe lossy JPEG compression at quality 65
        buffer = io.BytesIO()
        contrast_adj.save(buffer, format="JPEG", quality=65)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        # 5. Restore original dimensions
        return recompressed.resize((w, h), resample=Image.Resampling.BICUBIC)


# ============================================================================
# 2. AUDIO PERTURBATIONS
# ============================================================================

class AudioPerturbations:
    """Realistic acoustic perturbations: MP3 lossy codecs, additive noise, and phone band."""

    @staticmethod
    def mp3_compression(wave: np.ndarray, bitrate_kbps: int = 128, sr: int = 16000) -> np.ndarray:
        """Apply MP3 compression via ffmpeg if installed, or psychoacoustic spectral cutoff fallback."""
        if shutil.which("ffmpeg") is not None:
            try:
                import soundfile as sf
                pcm = io.BytesIO()
                clipped = np.clip(wave, -1.0, 1.0)
                sf.write(pcm, clipped, sr, format="WAV", subtype="PCM_16")
                with tempfile.TemporaryDirectory() as folder:
                    encoded_file = os.path.join(folder, "temp.mp3")
                    cmd_enc = [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-f", "wav", "-i", "pipe:0",
                        "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k",
                        "-y", encoded_file
                    ]
                    subprocess.run(cmd_enc, input=pcm.getvalue(), check=True)

                    cmd_dec = [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-i", encoded_file, "-ac", "1", "-ar", str(sr),
                        "-f", "wav", "pipe:1"
                    ]
                    proc = subprocess.run(cmd_dec, capture_output=True, check=True)
                    decoded, _ = sf.read(io.BytesIO(proc.stdout), dtype="float32")
                    # Match input length
                    if len(decoded) < len(wave):
                        return np.pad(decoded, (0, len(wave) - len(decoded)))
                    return decoded[:len(wave)]
            except Exception as exc:
                logger.debug("FFmpeg MP3 transcoding fallback triggered: %s", exc)

        # Pure-python MDCT/psychoacoustic simulation fallback:
        # Lower bitrates impose steep brickwall low-pass filter (e.g. 11kHz for 64kbps, 14kHz for 128kbps)
        # and uniform MDCT quantization noise.
        cutoff = 5500 if bitrate_kbps <= 64 else 7200
        nyquist = sr / 2.0
        norm_cutoff = min(0.95, cutoff / nyquist)
        b, a = signal.butter(6, norm_cutoff, btype="low")
        filtered = signal.filtfilt(b, a, wave).astype(np.float32)
        # Add slight sub-band quantization noise proportional to bit allocation
        quant_step = 0.008 if bitrate_kbps <= 64 else 0.003
        quant_noise = np.random.uniform(-quant_step, quant_step, size=wave.shape).astype(np.float32)
        return np.clip(filtered + quant_noise, -1.0, 1.0)

    @staticmethod
    def additive_noise(wave: np.ndarray, snr_db: float = 20.0, seed: int = 42) -> np.ndarray:
        """Add Gaussian pink/white noise at the exact specified Signal-to-Noise Ratio (dB)."""
        rng = np.random.default_rng(seed)
        signal_power = float(np.mean(wave ** 2)) + 1e-12
        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(0.0, 1.0, size=wave.shape).astype(np.float32)
        current_noise_power = float(np.mean(noise ** 2)) + 1e-12
        scaled_noise = noise * np.sqrt(target_noise_power / current_noise_power)
        return (wave + scaled_noise).astype(np.float32)

    @staticmethod
    def phone_band_filter(wave: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Simulate standard telephony channel bandpass filter (ITU-T G.712: 300 Hz - 3400 Hz)."""
        nyquist = sr / 2.0
        low = 300.0 / nyquist
        high = 3400.0 / nyquist
        # 4th order Butterworth bandpass
        sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
        bandpassed = signal.sosfiltfilt(sos, wave)
        # Standard telephone companding (soft non-linear compression)
        compressed = np.tanh(bandpassed * 1.5) / 1.5
        return compressed.astype(np.float32)


# ============================================================================
# 3. TEXT PERTURBATIONS
# ============================================================================

class TextPerturbations:
    """Text perturbations: paraphrasing, synonym swaps, typos, and human/AI paragraph mixing."""

    KEYBOARD_ADJACENT = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxc', 'e': 'wsdr',
        'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujko', 'j': 'uikmnh',
        'k': 'ijlm', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxza', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
    }

    SYNONYM_DICT = {
        "model": ["architecture", "framework", "system", "network"],
        "detect": ["identify", "recognize", "spot", "flag"],
        "features": ["attributes", "characteristics", "properties"],
        "synthetic": ["artificial", "generated", "fake", "fabricated"],
        "genuine": ["authentic", "real", "human-written", "original"],
        "accuracy": ["precision", "performance", "fidelity"],
        "analysis": ["evaluation", "assessment", "investigation"],
        "result": ["outcome", "finding", "measurement"],
        "demonstrate": ["show", "illustrate", "exhibit", "reveal"],
        "significant": ["substantial", "notable", "considerable", "marked"],
        "robust": ["resilient", "durable", "sturdy", "impervious"],
        "method": ["approach", "technique", "procedure", "strategy"],
    }

    @classmethod
    def apply_typos(cls, text: str, typo_rate: float = 0.05, seed: int = 42) -> str:
        """Introduce realistic QWERTY keyboard adjacent typos, drops, and transpositions."""
        rng = random.Random(seed)
        chars = list(text)
        total_chars = len(chars)
        num_typos = max(1, int(total_chars * typo_rate))

        indices = sorted(rng.sample(range(total_chars), min(num_typos, total_chars)))
        for idx in indices:
            c = chars[idx].lower()
            if c in cls.KEYBOARD_ADJACENT and rng.random() < 0.7:
                replacement = rng.choice(cls.KEYBOARD_ADJACENT[c])
                chars[idx] = replacement.upper() if chars[idx].isupper() else replacement
            elif idx < total_chars - 1 and rng.random() < 0.2:
                # Transposition error
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    @classmethod
    def apply_synonym_swaps(cls, text: str, swap_rate: float = 0.2, seed: int = 42) -> str:
        """Replace eligible content words with semantic synonyms to disrupt n-gram perplexity."""
        rng = random.Random(seed)
        words = text.split()
        for i, word in enumerate(words):
            clean = word.lower().strip(",.!?():;")
            if clean in cls.SYNONYM_DICT and rng.random() < swap_rate:
                chosen = rng.choice(cls.SYNONYM_DICT[clean])
                if word[0].isupper():
                    chosen = chosen.capitalize()
                # Preserve trailing punctuation
                punct = "".join([c for c in word if not c.isalnum()])
                words[i] = chosen + punct
        return " ".join(words)

    @classmethod
    def apply_paraphrase(cls, text: str) -> str:
        """Rule-based semantic paraphraser altering syntactic voice and transition connectives."""
        replacements = [
            ("In order to ", "To "),
            ("Furthermore, ", "Moreover, "),
            ("Consequently, ", "As a result, "),
            ("On the other hand, ", "Conversely, "),
            ("It is evident that ", "Clearly, "),
            ("In this study, we propose ", "Here, we introduce "),
            ("demonstrates high performance", "exhibits superior accuracy"),
            ("due to the fact that", "because"),
            ("is capable of", "can"),
            ("serves as a", "acts as a"),
        ]
        out = text
        for pat, repl in replacements:
            out = out.replace(pat, repl)
        return out

    @classmethod
    def mix_paragraphs(cls, ai_text: str, human_text: str, mix_ratio: float = 0.5, seed: int = 42) -> str:
        """Interweave human and AI sentences/paragraphs at specified proportion (e.g. 25%, 50%, 75%)."""
        ai_sents = [s.strip() for s in ai_text.split(".") if s.strip()]
        human_sents = [s.strip() for s in human_text.split(".") if s.strip()]
        if not ai_sents or not human_sents:
            return ai_text

        rng = random.Random(seed)
        total = len(ai_sents)
        k = max(1, min(len(human_sents), round(total * mix_ratio)))
        indices = set(rng.sample(range(total), k))

        mixed = []
        h_idx = 0
        for i, sent in enumerate(ai_sents):
            if i in indices and h_idx < len(human_sents):
                mixed.append(human_sents[h_idx])
                h_idx += 1
            else:
                mixed.append(sent)
        return ". ".join(mixed) + "."


# ============================================================================
# 4. ADVERSARIAL ATTACKS (FGSM & PGD) & ADVERSARIAL DEFENSE
# ============================================================================

def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2014).

    x_adv = clamp(x + eps * sign(grad_x Loss(x, y)), 0, 1)
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.binary_cross_entropy_with_logits(outputs.view(-1), labels.float().view(-1))
    loss.backward()

    with torch.no_grad():
        sign_grad = images.grad.data.sign()
        perturbed = images + epsilon * sign_grad
        return torch.clamp(perturbed, 0.0, 1.0)


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float | None = None,
    num_steps: int = 7,
) -> torch.Tensor:
    """Projected Gradient Descent (Madry et al., 2018) bounded in L_infinity ball."""
    if alpha is None:
        alpha = epsilon / 4.0

    original = images.clone().detach()
    # Random initial perturbation inside L_inf ball
    perturbed = (original + torch.empty_like(original).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)

    for _ in range(num_steps):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = F.binary_cross_entropy_with_logits(outputs.view(-1), labels.float().view(-1))
        loss.backward()

        with torch.no_grad():
            sign_grad = perturbed.grad.data.sign()
            perturbed = perturbed + alpha * sign_grad
            # Projection back to L_inf epsilon-ball around original image
            eta = torch.clamp(perturbed - original, min=-epsilon, max=epsilon)
            perturbed = torch.clamp(original + eta, min=0.0, max=1.0).detach()

    return perturbed


class ToyImageDetector(nn.Module):
    """Lightweight CNN model for self-contained, CPU-friendly verification of attacks & defense."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(32 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x)).squeeze(-1)


def train_adversarial_defense(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 3,
    epsilon: float = 8.0 / 255.0,
    lr: float = 1e-3,
) -> nn.Module:
    """Adversarial Training Defense (Madry et al.) training on a 50/50 mix of clean & PGD perturbed samples."""
    robust_model = ToyImageDetector()
    robust_model.load_state_dict(model.state_dict())
    robust_model.train()
    optimizer = torch.optim.Adam(robust_model.parameters(), lr=lr)

    for _epoch in range(epochs):
        optimizer.zero_grad()
        # Half clean, half adversarial
        adv_images = pgd_attack(robust_model, images, labels, epsilon=epsilon, num_steps=4)
        out_clean = robust_model(images)
        out_adv = robust_model(adv_images)

        loss_clean = F.binary_cross_entropy_with_logits(out_clean, labels.float())
        loss_adv = F.binary_cross_entropy_with_logits(out_adv, labels.float())
        loss = 0.5 * loss_clean + 0.5 * loss_adv
        loss.backward()
        optimizer.step()

    robust_model.eval()
    return robust_model
