"""Model, training, evaluation and export tests on generated images.

Fake images carry a faint checkerboard, which is the classic trace left by
upsampling in a GAN. Training on it shows the pipeline learns, and the numbers
here say nothing about real-world accuracy.
"""

import json
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from ml.common.manifest import FAKE, REAL, REAL_GENERATOR, Record, records_to_frame, write_manifest
from ml.common.splits import SplitConfig, assign_splits
from ml.image import augment, export, results_table
from ml.image.evaluate import evaluate
from ml.image.model import DctSpectrum, FrequencyBranch, ImageDetector, dct_basis, load_checkpoint
from ml.image.train import run_name, smooth_targets, train, warmup_cosine

CONFIG_DIR = str(Path(__file__).parents[1] / "configs")
SIZE = 64
CNN_MEAN = (0.485, 0.456, 0.406)
CNN_STD = (0.229, 0.224, 0.225)


def experiment_config(tmp_path: Path, *overrides: str) -> dict:
    values = {
        "model": "efficientnet_b0", "model.pretrained": "false", "model.batch_size": "8", "tracker": "none",
        "image_size": str(SIZE), "frequency_channels": "[8,16]", "num_workers": "0", "device": "cpu",
        "epochs": "4", "patience": "4", "model.learning_rate": "0.001",
        "manifest": (tmp_path / "toy.csv").as_posix(), "data_root": (tmp_path / "data").as_posix(),
        "checkpoint_dir": (tmp_path / "checkpoints").as_posix(),
    }
    values.update(override.split("=", 1) for override in overrides)
    with initialize_config_dir(CONFIG_DIR, version_base=None):
        config = compose("train", overrides=[f"{key}={value}" for key, value in values.items()])
    return OmegaConf.to_container(config, resolve=True)


def make_image(rng: np.random.Generator, strength: float) -> np.ndarray:
    base = rng.normal(127, 40, (SIZE, SIZE, 3)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 3.0)
    checker = (np.indices((SIZE, SIZE)).sum(axis=0) % 2)[..., None] * strength
    return np.clip(base + checker, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def toy_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("toy")
    rng = np.random.default_rng(0)
    images = tmp_path / "data" / "toy"
    images.mkdir(parents=True)
    records = []
    for index in range(80):
        for label, generator, strength in ((REAL, REAL_GENERATOR, 0.0), (FAKE, "seen_gan", 24.0)):
            name = f"{generator}-{index}.png"
            Image.fromarray(make_image(rng, strength)).save(images / name)
            records.append(Record(name, label, "toy", generator, f"{generator}{index}", content_hash=name))
    for index in range(20):
        name = f"new_gan-{index}.png"
        Image.fromarray(make_image(rng, 24.0)).save(images / name)
        records.append(Record(name, FAKE, "toy", "new_gan", f"new{index}", content_hash=name))
    result = assign_splits(records_to_frame(records), SplitConfig(heldout_generators=frozenset({"new_gan"})))
    write_manifest(result.frame, tmp_path / "toy.csv")
    return tmp_path


def test_dct_basis_is_orthonormal():
    basis = dct_basis(32)
    assert torch.allclose(basis @ basis.T, torch.eye(32), atol=1e-5)


def test_spectrum_of_a_flat_image_has_only_a_dc_term():
    module = DctSpectrum(16, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    spectrum = module(torch.full((1, 3, 16, 16), 0.5))[0, 0]
    assert spectrum[0, 0] > 1.0
    assert spectrum[0, 1:].max() < -6.0


def test_spectrum_does_not_depend_on_backbone_normalization():
    pixels = torch.rand(1, 3, 32, 32)
    cnn = torch.tensor(CNN_MEAN).view(1, 3, 1, 1), torch.tensor(CNN_STD).view(1, 3, 1, 1)
    vit = torch.full((1, 3, 1, 1), 0.5), torch.full((1, 3, 1, 1), 0.5)
    first = DctSpectrum(32, CNN_MEAN, CNN_STD)((pixels - cnn[0]) / cnn[1])
    second = DctSpectrum(32, (0.5,) * 3, (0.5,) * 3)((pixels - vit[0]) / vit[1])
    assert torch.allclose(first, second, atol=1e-3)


def test_checkerboard_is_visible_in_the_high_frequency_corner():
    module = DctSpectrum(SIZE, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    rng = np.random.default_rng(1)
    clean, checkered = (
        torch.from_numpy(make_image(rng, strength).astype(np.float32) / 255).permute(2, 0, 1)[None]
        for strength in (0.0, 24.0)
    )
    assert module(checkered)[0, 0, -1, -1] > module(clean)[0, 0, -1, -1] + 2.0


def test_frequency_branch_keeps_position():
    branch = FrequencyBranch(SIZE, CNN_MEAN, CNN_STD, (8, 16))
    assert branch.classifier.in_features == 16 * 4 * 4


def test_models_follow_each_backbones_normalization():
    cnn = ImageDetector("efficientnet_b0.ra_in1k", False, False)
    vit = ImageDetector("vit_base_patch16_224.augreg_in21k_ft_in1k", False, False)
    assert cnn.mean == pytest.approx(CNN_MEAN)
    assert vit.mean == pytest.approx((0.5, 0.5, 0.5))


@pytest.mark.parametrize("use_frequency", [False, True])
def test_detector_returns_one_logit_per_image(use_frequency):
    model = ImageDetector("efficientnet_b0.ra_in1k", use_frequency, False, SIZE, (8, 16))
    assert model(torch.randn(3, 3, SIZE, SIZE)).shape == (3,)


def test_gradients_reach_both_branches():
    model = ImageDetector("efficientnet_b0.ra_in1k", True, False, SIZE, (8, 16))
    model(torch.randn(2, 3, SIZE, SIZE)).sum().backward()
    assert model.backbone.classifier.weight.grad.abs().sum() > 0
    assert model.frequency.classifier.weight.grad.abs().sum() > 0


def test_smooth_targets_move_both_labels_towards_the_middle():
    assert smooth_targets(torch.tensor([0.0, 1.0]), 0.1).tolist() == pytest.approx([0.05, 0.95])


def test_learning_rate_warms_up_then_decays_to_zero():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = warmup_cosine(optimizer, 10, 100)
    rates = []
    for _ in range(100):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert rates[0] == pytest.approx(0.1)
    assert rates[9] == pytest.approx(1.0)
    assert rates[50] < rates[9]
    assert rates[-1] < 0.01


def test_every_augmentation_can_be_switched_off(tmp_path):
    robust = experiment_config(tmp_path)["augment"]
    none = experiment_config(tmp_path, "augment=none")["augment"]
    names = lambda config: {type(step).__name__ for step in augment.train_transform(config, 64, CNN_MEAN, CNN_STD, 0).transforms}  # noqa: E731
    assert {"RandomResizedCrop", "ColorJitter", "GaussianBlur", "Downscale", "ImageCompression"} <= names(robust)
    assert names(none) == {"HorizontalFlip", "Normalize", "ToTensorV2"}


def test_each_ablation_flag_controls_one_augmentation(tmp_path):
    robust = experiment_config(tmp_path, "augment.jpeg.enabled=false")["augment"]
    names = {type(step).__name__ for step in augment.train_transform(robust, 64, CNN_MEAN, CNN_STD, 0).transforms}
    assert "ImageCompression" not in names
    assert "GaussianBlur" in names


def test_jpeg_range_matches_the_prompt(tmp_path):
    assert experiment_config(tmp_path)["augment"]["jpeg"]["quality"] == [30, 95]


def test_seeded_augmentation_repeats_and_keeps_shape(tmp_path):
    config = experiment_config(tmp_path)["augment"]
    pixels = make_image(np.random.default_rng(0), 0.0)
    first = augment.train_transform(config, SIZE, CNN_MEAN, CNN_STD, 5)(image=pixels)["image"]
    second = augment.train_transform(config, SIZE, CNN_MEAN, CNN_STD, 5)(image=pixels)["image"]
    assert first.shape == (3, SIZE, SIZE)
    assert torch.equal(first, second)


def test_jpeg_transform_really_alters_pixels():
    pixels = make_image(np.random.default_rng(0), 24.0)
    compressed = A.ImageCompression(quality_range=(30, 30), p=1.0)(image=pixels)["image"]
    assert np.abs(compressed.astype(int) - pixels.astype(int)).mean() > 1.0


def test_config_lists_all_three_architectures():
    backbones = {path.stem for path in (Path(CONFIG_DIR) / "model").glob("*.yaml")}
    assert backbones == {"efficientnet_b0", "convnext_tiny", "vit_b16"}


@pytest.fixture(scope="module")
def trained(toy_data: Path) -> tuple[Path, Path]:
    """One model shared by the tests that only read it."""
    config = experiment_config(toy_data)
    return train(config), toy_data


def test_training_learns_the_planted_artefact(trained):
    best, _ = trained
    checkpoint = torch.load(best, weights_only=False)
    assert checkpoint["best_auc"] > 0.95


def test_checkpoint_name_carries_the_experiment_and_version(trained):
    best, _ = trained
    assert best.name == "image-efficientnet_b0-freq-robust-1.0.0-best.pt"


def test_resume_continues_from_the_last_epoch(trained, tmp_path, capsys):
    _, root = trained
    shutil.copytree(root / "checkpoints", tmp_path / "checkpoints")
    config = experiment_config(
        root, "epochs=5", "patience=9", f"checkpoint_dir={(tmp_path / 'checkpoints').as_posix()}"
    )
    train(config)
    captured = capsys.readouterr().out
    assert "resuming" in captured
    assert "epoch 4:" in captured
    assert "epoch 0:" not in captured


def test_early_stopping_ends_a_stalled_run(toy_data, tmp_path):
    config = experiment_config(
        toy_data, "patience=1", "epochs=30", "model.learning_rate=1e-9", "warmup_epochs=0",
        f"checkpoint_dir={tmp_path.as_posix()}",
    )
    path = train(config)
    log = (path.parent / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) < 30
    assert path.exists()


def test_evaluation_reports_every_required_metric_and_plot(trained, tmp_path):
    best, root = trained
    output = tmp_path / "results"
    report = evaluate(best, None, None, output, batch_size=16, workers=0)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "auc_interval"):
        assert key in report["test"]
    assert report["unseen"]["samples"] > 0
    assert report["unseen_generators"] == ["new_gan"]
    assert {row["generator_name"] for row in report["per_generator"]} == {REAL_GENERATOR, "seen_gan", "new_gan"}
    assert report["threshold_choice"]["target_precision"] == 0.95
    assert report["calibration"]["temperature"] > 0
    for figure in ("roc.png", "confusion.png", "reliability.png", "threshold.png", "generators.png"):
        assert (output / figure).stat().st_size > 1000
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8"))["name"] == report["name"]


def test_threshold_is_chosen_without_seeing_test_labels(trained, tmp_path):
    best, root = trained
    first = evaluate(best, None, None, tmp_path / "one", batch_size=16, workers=0)
    manifest = pd.read_csv(root / "toy.csv", keep_default_na=False)
    flipped = manifest.copy()
    is_test = flipped["split"] == "test"
    flipped.loc[is_test, "label"] = 1 - flipped.loc[is_test, "label"]
    flipped.loc[is_test, "generator_name"] = np.where(flipped.loc[is_test, "label"] == 0, REAL_GENERATOR, "seen_gan")
    changed = tmp_path / "flipped.csv"
    flipped.to_csv(changed, index=False)
    second = evaluate(best, changed, None, tmp_path / "two", batch_size=16, workers=0)
    assert first["threshold_choice"] == second["threshold_choice"]
    assert first["calibration"] == second["calibration"]


def test_onnx_export_matches_pytorch_and_handles_any_batch(trained, tmp_path):
    best, _ = trained
    report = export.export_checkpoint(best, tmp_path, thread_counts=(1,))
    assert report["max_output_difference"] <= 1e-3
    assert report["verification_inputs"] == "validation images"
    assert report["latency"][0]["median_ms"] > 0
    model, checkpoint = load_checkpoint(best)
    runtime = export.session(tmp_path / report["onnx"])
    images, _ = export.verification_images(checkpoint)
    assert len(images) == 7
    with torch.inference_mode():
        expected = model(images).numpy()
    assert np.abs(runtime.run(["logit"], {"image": images.numpy()})[0] - expected).max() <= 1e-3
    assert len(report["sha256"]) == 64
    assert (tmp_path / report["onnx"]).with_suffix(".json").exists()


def test_rgb_only_model_also_exports(toy_data, tmp_path):
    config = experiment_config(
        toy_data, "use_frequency=false", "epochs=1", f"checkpoint_dir={(tmp_path / 'checkpoints').as_posix()}"
    )
    assert "-rgb-" in run_name(config)
    report = export.export_checkpoint(train(config), tmp_path, thread_counts=(1,))
    assert report["max_output_difference"] <= 1e-3


def test_verify_detects_a_wrong_export(trained, tmp_path):
    best, _ = trained
    model, checkpoint = load_checkpoint(best)
    path = tmp_path / "model.onnx"
    export.export_onnx(model, path, SIZE)
    images, _ = export.verification_images(checkpoint)
    assert export.verify(model, path, images)["max_output_difference"] <= 1e-3
    with torch.no_grad():
        model.fusion.weight.add_(0.01)
    with pytest.raises(export.ExportMismatch):
        export.verify(model, path, images)


def test_verification_falls_back_to_generated_images(trained):
    best, _ = trained
    _, checkpoint = load_checkpoint(best)
    moved = {**checkpoint, "config": {**checkpoint["config"], "manifest": "missing.csv"}}
    images, inputs = export.verification_images(moved)
    assert inputs == "generated images"
    assert images.shape == (export.VERIFY_SAMPLES, 3, SIZE, SIZE)
    pixels = images * torch.tensor(checkpoint["std"]).view(1, 3, 1, 1) + torch.tensor(checkpoint["mean"]).view(1, 3, 1, 1)
    assert pixels.min() >= -1e-5
    assert pixels.max() <= 1 + 1e-5


def test_results_table_is_built_from_files(trained, tmp_path):
    best, _ = trained
    results = tmp_path / "results"
    evaluate(best, None, None, results / best.parent.name, batch_size=16, workers=0)
    export.export_checkpoint(best, tmp_path / "onnx", thread_counts=(1,))
    frame = results_table.collect(results, tmp_path / "onnx")
    assert frame.loc[0, "model"] == "EfficientNet-B0"
    assert frame.loc[0, "branches"] == "RGB + frequency"
    assert frame.loc[0, "latency_ms"] > 0
    assert frame.loc[0, "warning"] == False  # noqa: E712
    latex = results_table.to_latex(frame)
    assert r"\begin{tabular}" in latex
    assert "EfficientNet-B0" in latex
