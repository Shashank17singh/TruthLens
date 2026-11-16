"""Saliency for audio: which time-frequency regions drove the decision.

    python -m ml.audio.explain <checkpoint> clip.wav --output outputs/explain

For spectrogram models this is gradient times input on the feature image the
classifier saw, so a region scores high only when it is both present and
influential. Self-supervised models read the waveform, so their saliency is
gradient times input on samples, summed into 10 ms frames: a time profile with
no frequency axis.

The explained window is the most suspicious one, the window with the highest
logit, since that is the evidence a listener would want to hear.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from ml.audio.features import HOP_LENGTH
from ml.audio.inference import windows
from ml.audio.model import SpectrogramDetector, load_checkpoint
from ml.audio.preprocess import load_mono

SMOOTHING = 1.5  # in feature cells; single-cell gradient spikes are not evidence


def _normalized(values: np.ndarray) -> np.ndarray:
    values = values - values.min()
    peak = values.max()
    return values / peak if peak > 0 else values


def saliency(model: torch.nn.Module, clip: np.ndarray) -> dict:
    """`clip` is one window of samples. Returns the logit, the map in [0, 1] and, for spectrogram models, the features."""
    model.eval()
    wave = torch.from_numpy(clip.astype(np.float32)).unsqueeze(0)
    if isinstance(model, SpectrogramDetector):
        with torch.no_grad():
            features = model.features(wave)
        features.requires_grad_(True)
        logit = model.classify(features)
        logit.sum().backward()
        relevance = (features.grad * features).detach().abs()[0, 0].numpy()
        return {
            "logit": float(logit.detach()[0]),
            "kind": "time-frequency",
            "map": _normalized(gaussian_filter(relevance, SMOOTHING)),
            "features": features.detach()[0, 0].numpy(),
        }
    wave.requires_grad_(True)
    logit = model(wave)
    logit.sum().backward()
    relevance = (wave.grad * wave).abs()[0].detach().numpy()
    frames = relevance[: relevance.size // HOP_LENGTH * HOP_LENGTH].reshape(-1, HOP_LENGTH).sum(axis=1)
    return {"logit": float(logit.detach()[0]), "kind": "time", "map": _normalized(gaussian_filter(frames, SMOOTHING))[None], "features": None}


def most_suspicious_window(model: torch.nn.Module, wave: np.ndarray, sample_rate: int, clip_seconds: float) -> tuple[np.ndarray, float]:
    batch, starts = windows(wave, int(clip_seconds * sample_rate))
    with torch.inference_mode():
        logits = model(torch.from_numpy(batch.astype(np.float32))).numpy()
    best = int(np.argmax(logits))
    return batch[best], starts[best] / sample_rate


def save_figure(result: dict, start_seconds: float, sample_rate: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seconds = result["map"].shape[1] * HOP_LENGTH / sample_rate
    extent = (start_seconds, start_seconds + seconds, 0, result["map"].shape[0])
    rows = 2 if result["features"] is not None else 1
    figure, axes = plt.subplots(rows, 1, figsize=(6.0, 2.2 * rows), sharex=True, squeeze=False)
    if result["features"] is not None:
        axes[0, 0].imshow(result["features"], origin="lower", aspect="auto", cmap="magma", extent=extent)
        axes[0, 0].set(ylabel="Feature bin", title="Input features")
    axis = axes[-1, 0]
    axis.imshow(result["map"], origin="lower", aspect="auto", cmap="viridis", extent=extent, vmin=0, vmax=1)
    axis.set(xlabel="Time (s)", ylabel="Feature bin" if result["kind"] == "time-frequency" else "", title="Saliency (gradient x input)")
    if result["kind"] == "time":
        axis.set_yticks([])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/explain"))
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    wave = load_mono(args.audio, config["sample_rate"])
    clip, start = most_suspicious_window(model, wave, config["sample_rate"], config["clip_seconds"])
    result = saliency(model, clip)
    save_figure(result, start, config["sample_rate"], args.output / f"{args.audio.stem}_saliency.png")
    summary = {"window_start": start, "logit": result["logit"], "kind": result["kind"]}
    (args.output / f"{args.audio.stem}_saliency.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
