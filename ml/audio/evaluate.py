"""Evaluate an audio checkpoint on whole files, each scored in 4-second windows.

    python -m ml.audio.evaluate checkpoints/audio/<run>/<run>-best.pt
    python -m ml.audio.evaluate <checkpoint> --asv-scores data/raw/asvspoof2019_la/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt

EER is the main metric. Both aggregations (mean and max over windows) are
reported. Calibration and the decision threshold (the validation EER point)
come from validation files only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.audio.data import ManifestAudio
from ml.audio.inference import score_long
from ml.audio.model import load_checkpoint
from ml.common import metrics, plots
from ml.common.manifest import REAL_GENERATOR, read_manifest, unseen_evaluation_frame

AGGREGATES = ("mean", "max")


def torch_logits(model: torch.nn.Module, device: str):
    @torch.inference_mode()
    def logits_of(batch: np.ndarray) -> np.ndarray:
        return model(torch.from_numpy(batch).to(device)).float().cpu().numpy()

    return logits_of


def score_files(model: torch.nn.Module, frame: pd.DataFrame, root: Path, config: dict, device: str) -> pd.DataFrame:
    """One row per file with `mean_logit` and `max_logit`."""
    dataset = ManifestAudio(frame, root, config["sample_rate"], config["clip_seconds"])
    logits_of = torch_logits(model, device)
    scores = [score_long(logits_of, dataset.load(index), config["sample_rate"], config["clip_seconds"]) for index in range(len(dataset))]
    return frame.reset_index(drop=True).assign(
        mean_logit=[score["mean_logit"] for score in scores], max_logit=[score["max_logit"] for score in scores]
    )


def read_asv_scores(path: Path) -> dict:
    """Official ASVspoof 2019 ASV score file: attack id or bonafide, key (target, nontarget or spoof), score."""
    table = pd.read_csv(path, sep=r"\s+", header=None, names=["source", "key", "score"])
    return metrics.asv_error_rates(table["key"].to_numpy(), table["score"].to_numpy())


def evaluate_aggregate(validation: pd.DataFrame, test: pd.DataFrame, unseen: pd.DataFrame | None, column: str, seed: int, asv: dict | None) -> dict:
    labels = validation["label"].to_numpy()
    logits = validation[column].to_numpy()
    calibration = metrics.fit_calibration(labels, logits)
    validation_eer = metrics.equal_error_rate(labels, logits)
    threshold = float(metrics.calibrated_probability(np.array([validation_eer["threshold"]]), calibration)[0])

    def part(frame: pd.DataFrame) -> dict:
        part_labels = frame["label"].to_numpy()
        part_logits = frame[column].to_numpy()
        probabilities = metrics.calibrated_probability(part_logits, calibration)
        result = metrics.classification_metrics(part_labels, probabilities, threshold)
        result["eer"] = metrics.equal_error_rate(part_labels, part_logits)["eer"]
        result["auc_interval"] = metrics.grouped_auc_interval(part_labels, probabilities, frame["group_id"].to_numpy(), seed)
        result["ece_uncalibrated"] = metrics.expected_calibration_error(part_labels, metrics.sigmoid(part_logits))
        result["ece_calibrated"] = metrics.expected_calibration_error(part_labels, probabilities)
        return result

    report = {
        "calibration": calibration,
        "threshold": threshold,
        "val_eer": validation_eer["eer"],
        "test": part(test),
        "unseen": part(unseen) if unseen is not None else None,
    }
    if asv is not None:
        report["test"]["min_tdcf"] = metrics.min_tdcf(test["label"].to_numpy(), test[column].to_numpy(), asv)
    attacks = test if unseen is None else pd.concat([test, unseen[unseen["generator_name"] != REAL_GENERATOR]], ignore_index=True)
    report["per_attack_eer"] = metrics.per_generator_eer(attacks, column).to_dict("records")
    return report


def evaluate(checkpoint_path: Path, manifest: Path | None, data_root: Path | None, output: Path, asv_scores: Path | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    frame = read_manifest(manifest or Path(config["manifest"]))
    root = data_root or Path(config["data_root"])

    def scored(part: pd.DataFrame) -> pd.DataFrame:
        return score_files(model, part, root, config, device)

    validation = scored(frame[frame["split"] == "val"])
    test = scored(frame[frame["split"] == "test"])
    unseen_frame = unseen_evaluation_frame(frame)
    unseen = scored(unseen_frame) if (unseen_frame["label"] == 1).any() else None
    asv = read_asv_scores(asv_scores) if asv_scores else None

    report = {
        "name": checkpoint["name"],
        "model": config["model"]["name"],
        "feature": config["feature"]["name"] if config["model"]["kind"] != "ssl" else "waveform",
        "augment": config["augment"]["name"],
        "best_epoch": checkpoint["epoch"],
        "best_val_auc": checkpoint["best_auc"],
        "warnings": metrics.quality_warnings(checkpoint["best_auc"]),
        "window_seconds": config["clip_seconds"],
        "window_overlap": 0.5,
        "unseen_generators": sorted(set(frame.loc[frame["split"] == "unseen_test", "generator_name"])),
    }
    for aggregate in AGGREGATES:
        report[aggregate] = evaluate_aggregate(validation, test, unseen, f"{aggregate}_logit", config["seed"], asv)

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    test_labels = test["label"].to_numpy()
    probabilities = {
        aggregate: metrics.calibrated_probability(test[f"{aggregate}_logit"].to_numpy(), report[aggregate]["calibration"])
        for aggregate in AGGREGATES
    }
    plots.roc_figure({f"Test, {aggregate} of windows": (test_labels, probabilities[aggregate]) for aggregate in AGGREGATES}, output / "roc.png")
    plots.confusion_figure(report["mean"]["test"]["confusion_matrix"], output / "confusion.png")
    plots.reliability_figure(
        metrics.reliability_bins(test_labels, metrics.sigmoid(test["mean_logit"].to_numpy())),
        metrics.reliability_bins(test_labels, probabilities["mean"]),
        report["mean"]["test"]["ece_uncalibrated"], report["mean"]["test"]["ece_calibrated"], output / "reliability.png",
    )
    if report["mean"]["per_attack_eer"]:
        plots.generator_figure(
            report["mean"]["per_attack_eer"], set(report["unseen_generators"]), output / "attacks.png",
            value="eer", xlabel="Equal error rate against real clips (lower is better)", title="EER by attack",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, help="defaults to the manifest used in training")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--asv-scores", type=Path, help="official ASV score file; enables min t-DCF")
    parser.add_argument("--output", type=Path, help="defaults to ml/eval/results/audio/<run>")
    args = parser.parse_args()

    output = args.output or Path("ml/eval/results/audio") / args.checkpoint.parent.name
    report = evaluate(args.checkpoint, args.manifest, args.data_root, output, args.asv_scores)
    summary = {aggregate: {key: report[aggregate]["test"][key] for key in ("eer", "accuracy", "f1", "roc_auc")} for aggregate in AGGREGATES}
    print(json.dumps({"name": report["name"], "warnings": report["warnings"], "test": summary}, indent=2))
    print(f"results: {output}")


if __name__ == "__main__":
    main()
