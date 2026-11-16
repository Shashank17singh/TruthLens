"""Explainability tests on generated images and untrained models. They check the mechanics, not what a trained model attends to."""

import os
import time

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from ml.common.manifest import REAL_GENERATOR
from ml.image import explain
from ml.image.model import ImageDetector
from ml.image.preprocess import Detection

IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


def checkpoint_for(model: ImageDetector, size: int) -> dict:
    return {"config": {"image_size": size}, "mean": model.mean, "std": model.std}


def blob(size: int, centre: tuple[int, int], sigma: float, height: float) -> np.ndarray:
    y, x = np.indices((size, size))
    return height * np.exp(-((x - centre[0]) ** 2 + (y - centre[1]) ** 2) / (2 * sigma**2))


@pytest.fixture(scope="module")
def cnn() -> ImageDetector:
    torch.manual_seed(0)
    return ImageDetector("efficientnet_b0", use_frequency=True, pretrained=False).eval()


@pytest.fixture(scope="module")
def photo() -> np.ndarray:
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur(rng.integers(0, 256, (300, 260, 3), dtype=np.uint8), (0, 0), 2)


def test_regions_follow_the_peaks_strongest_first():
    heatmap = blob(224, (50, 60), 8, 1.0) + blob(224, (170, 40), 8, 0.7) + blob(224, (120, 180), 8, 0.4) + blob(224, (20, 200), 6, 0.1)
    regions = explain.top_regions(heatmap)
    centres = [((r.box[0] + r.box[2]) // 2, (r.box[1] + r.box[3]) // 2) for r in regions]
    assert [round(r.score, 1) for r in regions] == [1.0, 0.7, 0.4]
    for found, expected in zip(centres, [(50, 60), (170, 40), (120, 180)]):
        assert abs(found[0] - expected[0]) <= 2 and abs(found[1] - expected[1]) <= 2


def test_weak_peaks_are_not_reported():
    assert explain.top_regions(blob(64, (30, 30), 4, 0.1)) == []


FACE = Detection((60.0, 40.0, 164.0, 190.0), 0.99, ((90.0, 90.0), (134.0, 90.0), (112.0, 120.0), (95.0, 150.0), (129.0, 150.0)))


@pytest.mark.parametrize("centre,part", [
    ((112, 152), "mouth"), ((90, 92), "eyes"), ((134, 88), "eyes"), ((112, 118), "nose"),
    ((112, 42), "hairline"), ((112, 188), "chin"), ((10, 10), "background"),
])
def test_regions_are_named_after_the_nearest_face_part(centre, part):
    region = explain.Region((centre[0] - 4, centre[1] - 4, centre[0] + 4, centre[1] + 4), 1.0)
    assert explain.name_part(region, FACE, 224) == part


def test_without_a_face_regions_are_named_by_position():
    assert explain.name_part(explain.Region((0, 0, 10, 10), 1.0), None, 224) == "upper left"
    assert explain.name_part(explain.Region((107, 107, 117, 117), 1.0), None, 224) == "centre"


def test_sentences_read_plainly():
    assert explain.describe(["mouth", "hairline", "mouth"], leans_fake=True) == "Unusual texture detected around the mouth and hairline."
    assert explain.describe(["eyes", "nose", "background"], leans_fake=True) == (
        "Unusual texture detected around the eyes and nose and in the background."
    )
    assert explain.describe(["background"], leans_fake=False).endswith("closely in the background.")
    assert "No region stood out" in explain.describe([], leans_fake=True)


@pytest.mark.skipif(os.environ.get("TRUTHLENS_PERF") != "1", reason="timing needs an idle machine; set TRUTHLENS_PERF=1")
def test_explanation_finishes_under_two_seconds_on_cpu(cnn, photo):
    checkpoint = checkpoint_for(cnn, 224)
    explain.explain(cnn, checkpoint, photo)  # warm-up, as a serving worker would be
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        explain.explain(cnn, checkpoint, photo, detector=lambda _: [FACE])
        timings.append(time.perf_counter() - started)
    assert float(np.median(timings)) < 2.0, f"median explanation took {np.median(timings):.2f} s on CPU"


def test_grad_cam_explanation_is_complete(cnn, photo):
    image, result = explain.explain(cnn, checkpoint_for(cnn, 224), photo, detector=lambda _: [FACE])
    assert image.shape == (224, 224, 3)
    assert result.heatmap.shape == (224, 224)
    assert 0.0 <= result.heatmap.min() and result.heatmap.max() <= 1.0
    assert 1 <= len(result.regions) <= 3
    assert all(region.part for region in result.regions)
    assert result.method == "gradcam" and result.text.endswith(".")
    assert explain.overlay(image, result).shape == (224, 224, 3)
    assert "heatmap" not in result.summary()


def test_grad_cam_plus_plus_and_convnext_layer(photo):
    model = ImageDetector("convnext_atto", use_frequency=False, pretrained=False).eval()
    _, result = explain.explain(model, checkpoint_for(model, 224), photo, method="gradcam++")
    assert result.heatmap.shape == (224, 224) and result.method == "gradcam++"


def test_vit_uses_attention_rollout_and_restores_fused_attention(photo):
    model = ImageDetector("vit_tiny_patch16_224", use_frequency=False, pretrained=False).eval()
    fused = [block.attn.fused_attn for block in model.backbone.blocks]
    _, result = explain.explain(model, checkpoint_for(model, 224), photo)
    assert result.method == "rollout"
    assert result.heatmap.shape == (224, 224) and np.isclose(result.heatmap.max(), 1.0)
    assert [block.attn.fused_attn for block in model.backbone.blocks] == fused
    with pytest.raises(ValueError):
        explain.target_layer(model)


def test_randomizing_weights_changes_the_heatmap(cnn, photo):
    result = explain.sanity_check(cnn, checkpoint_for(cnn, 224), [photo])
    steps = result["steps"]
    assert [step["randomized_layers"] for step in steps] == list(range(1, len(steps) + 1))
    assert steps[-1]["spearman"] < 0.9
    original = explain.heatmap_for(cnn, explain.to_tensor(photo, checkpoint_for(cnn, 224))[1], "gradcam")
    assert np.allclose(original, explain.heatmap_for(cnn, explain.to_tensor(photo, checkpoint_for(cnn, 224))[1], "gradcam"))


def test_average_spectra_reveal_a_planted_checkerboard(tmp_path):
    size, rng = 64, np.random.default_rng(1)
    folder = tmp_path / "toy"
    folder.mkdir()
    rows = []
    for index in range(12):
        for generator, strength in ((REAL_GENERATOR, 0.0), ("gan", 30.0)):
            base = cv2.GaussianBlur(rng.normal(127, 40, (size, size, 3)).astype(np.float32), (0, 0), 3)
            checker = (np.indices((size, size)).sum(axis=0) % 2)[..., None] * strength
            name = f"{generator}-{index}.png"
            Image.fromarray(np.clip(base + checker, 0, 255).astype(np.uint8)).save(folder / name)
            rows.append({"path": name, "source_dataset": "toy", "generator_name": generator})
    checkpoint = {"config": {"image_size": size}, **IMAGENET}
    averages = explain.average_spectra(pd.DataFrame(rows), tmp_path, checkpoint, per_group=12, seed=0)
    difference = averages["gan"] - averages[REAL_GENERATOR]
    assert difference[-1, -1] > 5 * np.abs(difference[: size // 2, : size // 2]).mean()
    path = explain.save_spectra(averages, tmp_path / "figures")
    assert path.exists() and path.stat().st_size > 0


def test_spectra_need_real_images(tmp_path):
    with pytest.raises(ValueError):
        explain.save_spectra({"gan": np.zeros((4, 4))}, tmp_path)
