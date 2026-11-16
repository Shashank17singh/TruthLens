"""Audio deepfake detectors. Each takes a 16 kHz waveform batch [B, samples] and returns one logit per clip; positive leans fake.

- ResNet-18 (timm) on a log-mel, LFCC or CQT image.
- LCNN with max-feature-map activations, the light CNN of the ASVspoof baselines.
- A wav2vec2 or WavLM backbone with a pooled linear head, trained frozen and then partly unfrozen.
"""

from pathlib import Path

import timm
import torch
from torch import nn

from ml.audio.features import build_frontend


class SpectrogramDetector(nn.Module):
    """A frontend, a single-channel batch norm, then any image-style classifier."""

    def __init__(self, frontend: nn.Module, classifier: nn.Module):
        super().__init__()
        self.frontend = frontend
        # Log features sit far from zero mean and unit variance; this learns the shift once.
        self.norm = nn.BatchNorm2d(1)
        self.classifier = classifier

    def features(self, wave: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wave.device.type, enabled=False):
            return self.norm(self.frontend(wave.float()))

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features).squeeze(1)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.classify(self.features(wave))


class MaxFeatureMap(nn.Module):
    """Split the channels in two and keep the element-wise maximum (Wu et al., 2018)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return torch.maximum(first, second)


def _mfm_conv(inputs: int, outputs: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inputs, 2 * outputs, kernel, padding=kernel // 2), MaxFeatureMap())


class Lcnn(nn.Module):
    """The LCNN of Lavrentyev et al. (2019) as used in the ASVspoof 2021 baselines.

    The original ends in a fully connected layer sized for a fixed input. This
    pools to a 4x4 grid first, so clips and feature types of any size fit.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.body = nn.Sequential(
            _mfm_conv(1, 32, 5), nn.MaxPool2d(2),
            _mfm_conv(32, 32, 1), nn.BatchNorm2d(32), _mfm_conv(32, 48, 3), nn.MaxPool2d(2), nn.BatchNorm2d(48),
            _mfm_conv(48, 48, 1), nn.BatchNorm2d(48), _mfm_conv(48, 64, 3), nn.MaxPool2d(2),
            _mfm_conv(64, 64, 1), nn.BatchNorm2d(64), _mfm_conv(64, 32, 3), nn.BatchNorm2d(32),
            _mfm_conv(32, 32, 1), nn.BatchNorm2d(32), _mfm_conv(32, 32, 3), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(32 * 16, 2 * 80), nn.Unflatten(1, (2 * 80, 1)), MaxFeatureMap(),
            nn.Flatten(), nn.BatchNorm1d(80), nn.Linear(80, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(features))


class SslDetector(nn.Module):
    """Self-supervised speech backbone, mean-pooled over time, then one linear layer.

    The convolutional feature encoder always stays frozen, as in most
    fine-tuning recipes; `set_trainable` controls the transformer layers.
    """

    def __init__(self, backbone: nn.Module, normalize_input: bool):
        super().__init__()
        self.backbone = backbone
        self.normalize_input = normalize_input
        self.head = nn.Linear(backbone.config.hidden_size, 1)
        for parameter in self.backbone.feature_extractor.parameters():
            parameter.requires_grad = False

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:  # what the wav2vec2 feature extractor does before the model
            wave = (wave - wave.mean(dim=1, keepdim=True)) / torch.sqrt(wave.var(dim=1, keepdim=True) + 1e-7)
        hidden = self.backbone(wave).last_hidden_state
        return self.head(hidden.mean(dim=1)).squeeze(1)

    def set_trainable(self, top_layers: int) -> None:
        """Train only the head (0) or the head plus the top `top_layers` transformer layers."""
        layers = self.backbone.encoder.layers
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for layer in layers[len(layers) - top_layers :] if top_layers else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True


def ssl_backbone(name: str, pretrained: bool, overrides: dict | None = None) -> nn.Module:
    from transformers import AutoConfig, AutoModel

    if pretrained:
        return AutoModel.from_pretrained(name)
    config = AutoConfig.for_model(**overrides) if overrides else AutoConfig.from_pretrained(name)
    return AutoModel.from_config(config)


def build_model(config: dict, pretrained: bool | None = None, ssl_config: dict | None = None) -> nn.Module:
    """`ssl_config` is the backbone's own config saved in a checkpoint, so loading one needs no download."""
    model = config["model"]
    pretrained = model["pretrained"] if pretrained is None else pretrained
    kind = model["kind"]
    if kind == "ssl":
        backbone = ssl_backbone(model["backbone"], pretrained, ssl_config or model.get("config_overrides"))
        detector = SslDetector(backbone, model["normalize_input"])
        detector.set_trainable(0 if model["freeze_epochs"] > 0 else model["unfreeze_layers"])
        return detector
    frontend = build_frontend(config["feature"]["name"], config["sample_rate"])
    if kind == "resnet":
        classifier = timm.create_model(model["backbone"], pretrained=pretrained, in_chans=1, num_classes=1)
    elif kind == "lcnn":
        classifier = Lcnn(model["dropout"])
    else:
        raise ValueError(f"Unknown model kind {kind!r}")
    return SpectrogramDetector(frontend, classifier)


def checkpoint_metadata(model: nn.Module) -> dict:
    return {"ssl_config": model.backbone.config.to_dict()} if isinstance(model, SslDetector) else {}


def unfreeze_schedule(config: dict):
    """For SSL models: the head trains alone for `freeze_epochs`, then the top layers join."""
    model_config = config["model"]

    def before_epoch(model: nn.Module, epoch: int) -> None:
        if isinstance(model, SslDetector):
            model.set_trainable(model_config["unfreeze_layers"] if epoch >= model_config["freeze_epochs"] else 0)

    return before_epoch


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a trained model. Weights come from the file, so nothing is downloaded."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"], pretrained=False, ssl_config=checkpoint.get("ssl_config"))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
