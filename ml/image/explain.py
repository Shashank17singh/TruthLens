"""Explain an image verdict: a heatmap, the top regions, a sentence, and checks that the heatmap is real.

    python -m ml.image.explain image <checkpoint> <image> --output outputs/explain
    python -m ml.image.explain spectra <checkpoint> --split test --output paper/figures
    python -m ml.image.explain sanity <checkpoint> --split test --count 32

CNN backbones use Grad-CAM or Grad-CAM++ on their last feature map. ViT has no
spatial feature map, so it uses attention rollout instead.
"""

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

from ml.common.manifest import REAL_GENERATOR, read_manifest
from ml.image.augment import eval_transform
from ml.image.model import DctSpectrum, ImageDetector, load_checkpoint
from ml.image.preprocess import Detection, FaceDetector, resize_square

METHODS = ("gradcam", "gradcam++", "rollout")
TOP_REGIONS = 3
REGION_LEVEL = 0.5  # a region is the area above half of its own peak
MIN_PEAK = 0.2  # weaker peaks are noise, not evidence
FACE_MARGIN = 0.1  # boxes just outside the detector's box still belong to the face


@dataclass
class Region:
    box: tuple[int, int, int, int]  # left, top, right, bottom in model-input pixels
    score: float
    part: str = ""


@dataclass
class Explanation:
    logit: float
    method: str
    heatmap: np.ndarray = field(repr=False)
    regions: list[Region]
    text: str
    seconds: float

    def summary(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "heatmap"}


def is_vit(model: ImageDetector) -> bool:
    return hasattr(model.backbone, "blocks") and hasattr(model.backbone, "patch_embed")


def default_method(model: ImageDetector) -> str:
    return "rollout" if is_vit(model) else "gradcam"


def target_layer(model: ImageDetector) -> torch.nn.Module:
    """The last spatial feature map: the most class-specific one that still has a position."""
    backbone = model.backbone
    if hasattr(backbone, "conv_head"):  # EfficientNet
        return backbone.bn2
    if hasattr(backbone, "stages"):  # ConvNeXt
        return backbone.stages[-1]
    raise ValueError(f"No Grad-CAM layer for {type(backbone).__name__}; use attention rollout")


def _normalized(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap - heatmap.min()
    peak = heatmap.max()
    return heatmap / peak if peak > 0 else heatmap


def grad_cam(model: ImageDetector, tensor: torch.Tensor, plus: bool = False) -> np.ndarray:
    """Heatmap for the fake logit. The model returns one logit per image, so the target is the output itself."""
    method = GradCAMPlusPlus if plus else GradCAM
    with method(model=model, target_layers=[target_layer(model)]) as cam:
        heatmap = cam(input_tensor=tensor, targets=[lambda output: output])[0]
    return _normalized(heatmap.astype(np.float32))


@torch.no_grad()
def attention_rollout(model: ImageDetector, tensor: torch.Tensor) -> np.ndarray:
    """Abnar and Zuidema (2020): multiply head-averaged attention, plus identity for the skip path, through the layers."""
    backbone = model.backbone
    attentions: list[torch.Tensor] = []
    hooks, fused = [], []
    for block in backbone.blocks:
        fused.append(block.attn.fused_attn)
        block.attn.fused_attn = False  # the fused kernel never materializes the attention matrix
        hooks.append(block.attn.attn_drop.register_forward_hook(lambda _m, _i, output: attentions.append(output)))
    try:
        model(tensor)
    finally:
        for hook in hooks:
            hook.remove()
        for block, value in zip(backbone.blocks, fused):
            block.attn.fused_attn = value
    tokens = attentions[0].shape[-1]
    identity = torch.eye(tokens)
    rollout = identity
    for attention in attentions:
        layer = attention[0].mean(dim=0) + identity
        rollout = (layer / layer.sum(dim=-1, keepdim=True)) @ rollout
    prefix = backbone.num_prefix_tokens
    grid = backbone.patch_embed.grid_size
    patches = rollout[0, prefix:].reshape(grid).numpy()
    size = tensor.shape[-1]
    return _normalized(cv2.resize(patches.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC))


def heatmap_for(model: ImageDetector, tensor: torch.Tensor, method: str) -> np.ndarray:
    if method == "rollout":
        return attention_rollout(model, tensor)
    if method in ("gradcam", "gradcam++"):
        return grad_cam(model, tensor, plus=method == "gradcam++")
    raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")


def top_regions(heatmap: np.ndarray, count: int = TOP_REGIONS) -> list[Region]:
    """Local maxima, strongest first, each boxed by the area above half of its own peak.

    Local maxima and not repeated argmax: blanking a found region leaves its
    sloping edge behind, and argmax would report that edge as a second region.
    """
    window = max(3, heatmap.shape[0] // 16) | 1
    peaks = (heatmap == cv2.dilate(heatmap, np.ones((window, window), np.uint8))) & (heatmap >= MIN_PEAK)
    regions: list[Region] = []
    for y, x in sorted(zip(*np.nonzero(peaks)), key=lambda point: -heatmap[point]):
        if len(regions) == count:
            break
        if any(r.box[0] <= x < r.box[2] and r.box[1] <= y < r.box[3] for r in regions):
            continue  # a plateau or shoulder of a region already reported
        peak = float(heatmap[y, x])
        _, labels = cv2.connectedComponents((heatmap >= REGION_LEVEL * peak).astype(np.uint8), connectivity=8)
        ys, xs = np.nonzero(labels == labels[y, x])
        regions.append(Region((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), round(peak, 3)))
    return regions


def face_anchors(face: Detection) -> list[tuple[str, tuple[float, float]]]:
    left, top, right, bottom = face.box
    centre_x = (left + right) / 2
    anchors = [("hairline", (centre_x, top)), ("chin", (centre_x, bottom))]
    if len(face.landmarks) == 5:
        left_eye, right_eye, nose, mouth_left, mouth_right = face.landmarks
        mouth = ((mouth_left[0] + mouth_right[0]) / 2, (mouth_left[1] + mouth_right[1]) / 2)
        anchors += [("eyes", left_eye), ("eyes", right_eye), ("nose", nose), ("mouth", mouth)]
    return anchors


def name_part(region: Region, face: Detection | None, size: int) -> str:
    """Nearest facial landmark to the region's centre, or its position in the frame when there is no face."""
    left, top, right, bottom = region.box
    x, y = (left + right) / 2, (top + bottom) / 2
    if face is None:
        row = ("upper", "middle", "lower")[min(int(3 * y / size), 2)]
        column = ("left", "centre", "right")[min(int(3 * x / size), 2)]
        return "centre" if (row, column) == ("middle", "centre") else f"{row} {column}"
    f_left, f_top, f_right, f_bottom = face.box
    margin_x, margin_y = FACE_MARGIN * (f_right - f_left), FACE_MARGIN * (f_bottom - f_top)
    if not (f_left - margin_x <= x <= f_right + margin_x and f_top - margin_y <= y <= f_bottom + margin_y):
        return "background"
    return min(face_anchors(face), key=lambda anchor: (anchor[1][0] - x) ** 2 + (anchor[1][1] - y) ** 2)[0]


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def describe(parts: list[str], leans_fake: bool) -> str:
    """One plain sentence. Parts arrive strongest first and repeats are dropped."""
    names = list(dict.fromkeys(parts))
    if not names:
        return "No region stood out, so there is no localized evidence for this result."
    places = [name for name in names if name != "background"]
    phrase = f"around the {_join(places)}" if places else ""
    if "background" in names:
        phrase = f"{phrase} and in the background" if phrase else "in the background"
    if leans_fake:
        return f"Unusual texture detected {phrase}."
    return f"No strong sign of manipulation. The model looked most closely {phrase}."


def to_tensor(rgb: np.ndarray, checkpoint: dict) -> tuple[np.ndarray, torch.Tensor]:
    """Resize as preprocessing does, then normalize as evaluation does."""
    image = resize_square(rgb, checkpoint["config"]["image_size"])
    tensor = eval_transform(checkpoint["mean"], checkpoint["std"])(image=image)["image"].unsqueeze(0)
    return image, tensor


def explain(
    model: ImageDetector,
    checkpoint: dict,
    rgb: np.ndarray,
    method: str | None = None,
    detector: FaceDetector | None = None,
) -> tuple[np.ndarray, Explanation]:
    """Returns the model-sized image and its explanation. The logit is uncalibrated; the caller applies calibration."""
    started = time.perf_counter()
    method = method or default_method(model)
    image, tensor = to_tensor(rgb, checkpoint)
    model.eval()
    with torch.no_grad():
        logit = float(model(tensor)[0])
    heatmap = heatmap_for(model, tensor, method)
    regions = top_regions(heatmap)
    faces = detector(image) if detector else []
    face = max(faces, key=lambda found: found.score) if faces else None
    for region in regions:
        region.part = name_part(region, face, image.shape[0])
    text = describe([region.part for region in regions], leans_fake=logit > 0)
    return image, Explanation(logit, method, heatmap, regions, text, round(time.perf_counter() - started, 3))


def overlay(image: np.ndarray, explanation: Explanation, opacity: float = 0.5) -> np.ndarray:
    """Heatmap over the image with numbered boxes, strongest first."""
    picture = show_cam_on_image(image.astype(np.float32) / 255.0, explanation.heatmap, use_rgb=True, image_weight=1 - opacity)
    for rank, region in enumerate(explanation.regions, start=1):
        left, top, right, bottom = region.box
        cv2.rectangle(picture, (left, top), (right - 1, bottom - 1), (255, 255, 255), 2)
        cv2.putText(picture, f"{rank} {region.score:.2f}", (left + 3, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return picture


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


# Frequency evidence


@torch.no_grad()
def average_spectra(frame: pd.DataFrame, root: Path, checkpoint: dict, per_group: int, seed: int) -> dict[str, np.ndarray]:
    """Mean log-DCT spectrum of real images and of each generator, from the same transform the model uses."""
    size = checkpoint["config"]["image_size"]
    spectrum = DctSpectrum(size, checkpoint["mean"], checkpoint["std"])
    averages = {}
    for generator, rows in frame.groupby("generator_name"):
        rows = rows.sample(n=min(per_group, len(rows)), random_state=seed)
        total = torch.zeros(size, size)
        for row in rows.itertuples():
            _, tensor = to_tensor(load_rgb(root / row.source_dataset / row.path), checkpoint)
            total += spectrum(tensor)[0, 0]
        averages[str(generator)] = (total / len(rows)).numpy()
    return averages


def save_spectra(averages: dict[str, np.ndarray], output: Path) -> Path:
    """Real average, then each generator's difference from it, where generator artefacts stand out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if REAL_GENERATOR not in averages:
        raise ValueError("Spectra need real images to compare against")
    real = averages[REAL_GENERATOR]
    fakes = sorted(name for name in averages if name != REAL_GENERATOR)
    figure, axes = plt.subplots(1, len(fakes) + 1, figsize=(3.2 * (len(fakes) + 1), 3.4), squeeze=False)
    axes[0, 0].imshow(real, cmap="magma")
    axes[0, 0].set_title("Real average (log DCT)", fontsize=9)
    limit = max((np.abs(averages[name] - real).max() for name in fakes), default=1.0)
    for axis, name in zip(axes[0, 1:], fakes):
        image = axis.imshow(averages[name] - real, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"{name} minus real", fontsize=9)
    for axis in axes[0]:
        axis.set_xticks([])
        axis.set_yticks([])
    if fakes:
        figure.colorbar(image, ax=axes[0, 1:].tolist(), shrink=0.8)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "image_spectra.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


# Sanity check (Adebayo et al., 2018)


def randomize_top(model: ImageDetector, layers: int) -> ImageDetector:
    """Copy of the model with the last `layers` backbone children re-initialized, top first, as in the cascading test."""
    randomized = copy.deepcopy(model)
    children = list(randomized.backbone.children())
    for child in children[len(children) - layers :]:
        for module in child.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    return randomized.eval()


def similarity(original: np.ndarray, randomized: np.ndarray) -> dict:
    rank = spearmanr(original.ravel(), randomized.ravel()).statistic
    return {
        "spearman": float(np.nan_to_num(rank)),
        "ssim": float(structural_similarity(original, randomized, data_range=1.0)),
    }


def sanity_check(model: ImageDetector, checkpoint: dict, images: list[np.ndarray], method: str | None = None) -> dict:
    """If heatmaps barely change when learned weights are destroyed, they show edges, not the model's reasoning."""
    method = method or default_method(model)
    tensors = [to_tensor(image, checkpoint)[1] for image in images]
    originals = [heatmap_for(model, tensor, method) for tensor in tensors]
    depth = len(list(model.backbone.children()))
    steps = []
    for layers in range(1, depth + 1):
        randomized = randomize_top(model, layers)
        scores = [similarity(original, heatmap_for(randomized, tensor, method)) for original, tensor in zip(originals, tensors)]
        steps.append({
            "randomized_layers": layers,
            "spearman": float(np.mean([score["spearman"] for score in scores])),
            "ssim": float(np.mean([score["ssim"] for score in scores])),
        })
    return {"method": method, "images": len(images), "steps": steps}


# Command line


def _frame(checkpoint: dict, manifest: Path | None, split: str) -> tuple[pd.DataFrame, Path]:
    config = checkpoint["config"]
    frame = read_manifest(manifest or Path(config["manifest"]))
    return frame[frame["split"] == split].reset_index(drop=True), Path(config["data_root"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser("image", help="explain one image")
    single.add_argument("checkpoint", type=Path)
    single.add_argument("image", type=Path)
    single.add_argument("--method", choices=METHODS)
    single.add_argument("--no-face", action="store_true", help="skip face detection; regions are named by position")
    single.add_argument("--output", type=Path, default=Path("outputs/explain"))

    spectra = commands.add_parser("spectra", help="average spectrum per generator against real images")
    spectra.add_argument("checkpoint", type=Path)
    spectra.add_argument("--manifest", type=Path)
    spectra.add_argument("--data-root", type=Path)
    spectra.add_argument("--split", default="test")
    spectra.add_argument("--per-group", type=int, default=500)
    spectra.add_argument("--output", type=Path, default=Path("paper/figures"))

    sanity = commands.add_parser("sanity", help="model-randomization test")
    sanity.add_argument("checkpoint", type=Path)
    sanity.add_argument("--manifest", type=Path)
    sanity.add_argument("--data-root", type=Path)
    sanity.add_argument("--split", default="test")
    sanity.add_argument("--count", type=int, default=32)
    sanity.add_argument("--method", choices=METHODS)
    sanity.add_argument("--output", type=Path, default=Path("outputs/explain"))

    arguments = parser.parse_args()
    model, checkpoint = load_checkpoint(arguments.checkpoint)
    seed = checkpoint["config"]["seed"]

    if arguments.command == "image":
        from ml.image.preprocess import retinaface_detector

        detector = None if arguments.no_face else retinaface_detector(threshold=0.9)
        image, explanation = explain(model, checkpoint, load_rgb(arguments.image), arguments.method, detector)
        arguments.output.mkdir(parents=True, exist_ok=True)
        stem = arguments.image.stem
        Image.fromarray(overlay(image, explanation)).save(arguments.output / f"{stem}_overlay.png")
        (arguments.output / f"{stem}_explanation.json").write_text(json.dumps(explanation.summary(), indent=2))
        print(json.dumps(explanation.summary(), indent=2))
        return

    frame, root = _frame(checkpoint, arguments.manifest, arguments.split)
    root = arguments.data_root or root
    if arguments.command == "spectra":
        path = save_spectra(average_spectra(frame, root, checkpoint, arguments.per_group, seed), arguments.output)
        print(f"Saved {path}")
        return

    rows = frame.sample(n=min(arguments.count, len(frame)), random_state=seed)
    images = [load_rgb(root / row.source_dataset / row.path) for row in rows.itertuples()]
    result = sanity_check(model, checkpoint, images, arguments.method)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "sanity_check.json").write_text(json.dumps(result, indent=2))
    for step in result["steps"]:
        print(f"top {step['randomized_layers']:>2} layers random: spearman {step['spearman']:.3f}  ssim {step['ssim']:.3f}")


if __name__ == "__main__":
    main()
