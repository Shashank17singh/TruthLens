"""Image detector: a timm RGB backbone, optionally fused with a DCT frequency branch."""

import math
from pathlib import Path

import timm
import torch
from timm.data import resolve_model_data_config
from torch import nn

LUMA = (0.299, 0.587, 0.114)
LOG_FLOOR = 1e-3


def dct_basis(size: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix."""
    positions = torch.arange(size, dtype=torch.float32) + 0.5
    frequencies = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(math.pi / size * positions * frequencies) * math.sqrt(2.0 / size)
    basis[0] /= math.sqrt(2.0)
    return basis


class DctSpectrum(nn.Module):
    """Log-magnitude DCT of the image's luminance.

    Written as matrix products, so it exports to plain ONNX and runs inside the
    model at training and at serving. An FFT in preprocessing would be a second
    implementation that could drift from the training version.
    """

    def __init__(self, size: int, mean: tuple[float, ...], std: tuple[float, ...]):
        super().__init__()
        self.register_buffer("basis", dct_basis(size), persistent=False)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("luma", torch.tensor(LUMA).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Backbones differ in normalization, so recover pixels first. The spectrum then
        # means the same thing for every backbone.
        pixels = image * self.std + self.mean
        gray = (pixels * self.luma).sum(dim=1)
        coefficients = self.basis @ gray @ self.basis.T
        return torch.log(coefficients.abs() + LOG_FLOOR).unsqueeze(1)


class FrequencyBranch(nn.Module):
    """A small CNN over the spectrum.

    It ends in a 4×4 grid and not in global pooling. Position in a spectrum is
    frequency, and generator artefacts sit at particular frequencies, so
    averaging position away would discard the evidence.
    """

    def __init__(self, size: int, mean: tuple[float, ...], std: tuple[float, ...], channels: tuple[int, ...]):
        super().__init__()
        self.spectrum = DctSpectrum(size, mean, std)
        layers: list[nn.Module] = [nn.BatchNorm2d(1)]
        previous = 1
        for width in channels:
            layers += [
                nn.Conv2d(previous, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
                nn.MaxPool2d(2),
            ]
            previous = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(4), nn.Flatten())
        self.classifier = nn.Linear(previous * 16, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            spectrum = self.spectrum(image.float())
        return self.classifier(self.features(spectrum))


class ImageDetector(nn.Module):
    """Returns one logit per image; a positive value leans towards fake.

    Late fusion learns one weight per branch, so each branch keeps its own score
    and the ablation can compare RGB alone with RGB plus frequency directly.
    """

    def __init__(self, backbone: str, use_frequency: bool, pretrained: bool, image_size: int = 224,
                 frequency_channels: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=1)
        data_config = resolve_model_data_config(self.backbone)
        self.mean = tuple(float(value) for value in data_config["mean"])
        self.std = tuple(float(value) for value in data_config["std"])
        self.frequency = None
        self.fusion = None
        if use_frequency:
            self.frequency = FrequencyBranch(image_size, self.mean, self.std, tuple(frequency_channels))
            self.fusion = nn.Linear(2, 1, bias=False)
            nn.init.ones_(self.fusion.weight)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rgb = self.backbone(image)
        if self.frequency is None:
            return rgb.squeeze(1)
        return self.fusion(torch.cat([rgb, self.frequency(image)], dim=1)).squeeze(1)


def build_model(config: dict, pretrained: bool | None = None) -> ImageDetector:
    model = config["model"]
    return ImageDetector(
        model["backbone"],
        config["use_frequency"],
        model["pretrained"] if pretrained is None else pretrained,
        config["image_size"],
        tuple(config["frequency_channels"]),
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[ImageDetector, dict]:
    """Rebuild a trained model. Weights come from the file, so nothing is downloaded."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
