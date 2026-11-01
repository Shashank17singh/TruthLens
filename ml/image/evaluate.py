"""Evaluate a checkpoint.

    python -m ml.image.evaluate checkpoints/image/<run>/<run>-best.pt

Calibration and threshold come only from validation data. The test and unseen
splits are read once, afterwards, so neither can influence a choice.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.common import metrics, plots
from ml.common.manifest import REAL_GENERATOR, read_manifest, unseen_evaluation_frame
from ml.image.augment import eval_transform
from ml.image.data import eval_loader, predict_logits
from ml.image.model import load_checkpoint

TARGET_PRECISION = 0.95
TRADEOFF_TARGETS = (0.80, 0.90, 0.95, 0.99)


def score(model, frame: pd.DataFrame, root: Path, checkpoint: dict, batch_size: int, workers: int, device: str) -> np.ndarray:
    loader = eval_loader(frame, root, eval_transform(checkpoint["mean"], checkpoint["std"]), batch_size, workers)
    return predict_logits(model, loader, device, amp=device == "cuda")


def evaluate(
    checkpoint_path: Path,
    manifest: Path | None,
    data_root: Path | None,
    output: Path,
    batch_size: int,
    workers: int,
    target_precision: float = TARGET_PRECISION,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    frame = read_manifest(manifest or Path(config["manifest"]))
    root = data_root or Path(config["data_root"])

    validation = frame[frame["split"] == "val"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    unseen = unseen_evaluation_frame(frame)

    def scored(part: pd.DataFrame) -> np.ndarray:
        return score(model, part, root, checkpoint, batch_size, workers, device)

    logits = scored(validation)
    labels = validation["label"].to_numpy()
    calibration = metrics.fit_calibration(labels, logits)
    uncalibrated = metrics.sigmoid(logits)
    calibrated = metrics.calibrated_probability(logits, calibration)
    choice = metrics.threshold_for_precision(labels, calibrated, target_precision)
    threshold = choice["threshold"]
    tradeoff = metrics.precision_tradeoff(labels, calibrated, TRADEOFF_TARGETS)

    test_logits = scored(test)
    test_uncalibrated = metrics.sigmoid(test_logits)
    test_probabilities = metrics.calibrated_probability(test_logits, calibration)
    test_labels = test["label"].to_numpy()
    test_frame = test.assign(probability=test_probabilities)
    test_metrics = metrics.classification_metrics(test_labels, test_probabilities, threshold)
    test_metrics["auc_interval"] = metrics.grouped_auc_interval(
        test_labels, test_probabilities, test["group_id"].to_numpy(), config["seed"]
    )
    ece_before = metrics.expected_calibration_error(test_labels, test_uncalibrated)
    ece_after = metrics.expected_calibration_error(test_labels, test_probabilities)

    report = {
        "name": checkpoint["name"],
        "model": config["model"]["name"],
        "use_frequency": config["use_frequency"],
        "augment": config["augment"]["name"],
        "best_epoch": checkpoint["epoch"],
        "best_val_auc": checkpoint["best_auc"],
        "warnings": metrics.quality_warnings(checkpoint["best_auc"]),
        "calibration": calibration,
        "threshold_choice": choice,
        "precision_tradeoff_on_val": tradeoff,
        "val_ece_uncalibrated": metrics.expected_calibration_error(labels, uncalibrated),
        "val_ece_calibrated": metrics.expected_calibration_error(labels, calibrated),
        "test": test_metrics,
        "test_ece_uncalibrated": ece_before,
        "test_ece_calibrated": ece_after,
        "unseen": None,
    }

    generators = test_frame
    curves = {"Test": (test_labels, test_probabilities)}
    if (unseen["label"] == 1).any():
        unseen_probabilities = metrics.calibrated_probability(scored(unseen), calibration)
        unseen_labels = unseen["label"].to_numpy()
        report["unseen"] = metrics.classification_metrics(unseen_labels, unseen_probabilities, threshold)
        report["unseen"]["auc_interval"] = metrics.grouped_auc_interval(
            unseen_labels, unseen_probabilities, unseen["group_id"].to_numpy(), config["seed"]
        )
        curves["Unseen generators"] = (unseen_labels, unseen_probabilities)
        unseen_fakes = unseen.assign(probability=unseen_probabilities)
        unseen_fakes = unseen_fakes[unseen_fakes["generator_name"] != REAL_GENERATOR]
        generators = pd.concat([test_frame, unseen_fakes], ignore_index=True)

    per_generator = metrics.per_generator_accuracy(generators, threshold)
    report["per_generator"] = per_generator.to_dict("records")
    unseen_names = set(frame.loc[frame["split"] == "unseen_test", "generator_name"])
    report["unseen_generators"] = sorted(unseen_names)

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plots.roc_figure(curves, output / "roc.png")
    plots.confusion_figure(test_metrics["confusion_matrix"], output / "confusion.png")
    plots.reliability_figure(
        metrics.reliability_bins(test_labels, test_uncalibrated),
        metrics.reliability_bins(test_labels, test_probabilities),
        ece_before, ece_after, output / "reliability.png",
    )
    plots.precision_tradeoff_figure(labels, calibrated, threshold, target_precision, output / "threshold.png")
    plots.generator_figure(report["per_generator"], unseen_names, output / "generators.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, help="defaults to the manifest used in training")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, help="defaults to ml/eval/results/image/<run>")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    name = args.checkpoint.parent.name
    output = args.output or Path("ml/eval/results/image") / name
    report = evaluate(args.checkpoint, args.manifest, args.data_root, output, args.batch_size, args.workers)
    print(json.dumps({key: report[key] for key in ("name", "warnings", "calibration", "threshold_choice", "test")}, indent=2))
    print(f"results: {output}")


if __name__ == "__main__":
    main()
