"""Build the architecture comparison from evaluation output. Numbers are never typed by hand.

    python -m ml.image.results_table
"""

import argparse
import json
from pathlib import Path

import pandas as pd

DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_b16": "ViT-B/16",
}


def collect(results_dir: Path, onnx_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        test, unseen = report["test"], report["unseen"]
        interval = test["auc_interval"]
        row = {
            "run": report["name"],
            "warning": bool(report["warnings"]),
            "model": DISPLAY_NAMES.get(report["model"], report["model"]),
            "branches": "RGB + frequency" if report["use_frequency"] else "RGB",
            "augment": report["augment"],
            "accuracy": test["accuracy"],
            "precision": test["precision"],
            "recall": test["recall"],
            "f1": test["f1"],
            "auc": test["roc_auc"],
            "auc_low": interval["low"],
            "auc_high": interval["high"],
            "ece": report["test_ece_calibrated"],
            "unseen_recall": unseen["recall"] if unseen else None,
            "unseen_auc": unseen["roc_auc"] if unseen else None,
            "latency_ms": None,
        }
        sidecar = onnx_dir / f"{report['name']}.json"
        if sidecar.exists():
            latency = json.loads(sidecar.read_text(encoding="utf-8"))["latency"]
            row["latency_ms"] = next(item["median_ms"] for item in latency if item["threads"] == 1)
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(frame: pd.DataFrame) -> str:
    """An IEEE table with the best value in each metric column in bold."""
    columns = (("accuracy", "Acc."), ("f1", "F1"), ("auc", "AUC"), ("ece", "ECE"), ("unseen_auc", "Unseen AUC"))
    lowest_wins = {"ece"}
    best = {
        column: (frame[column].min() if column in lowest_wins else frame[column].max())
        for column, _ in columns
    }

    def cell(column: str, value: float | None) -> str:
        if value is None or pd.isna(value):
            return "--"
        text = f"{value:.3f}"
        return rf"\textbf{{{text}}}" if value == best[column] else text

    header = " & ".join(["Model", "Branches", "Aug."] + [label for _, label in columns] + ["CPU ms"])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Image detector comparison}",
        r"\label{tab:image-models}",
        r"\footnotesize",
        r"\begin{tabular}{lll" + "c" * (len(columns) + 1) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        latency = "--" if pd.isna(row["latency_ms"]) else f"{row['latency_ms']:.0f}"
        values = [row["model"], row["branches"].replace("frequency", "freq."), row["augment"]]
        values += [cell(column, row[column]) for column, _ in columns] + [latency]
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("ml/eval/results/image"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("checkpoints/onnx"))
    parser.add_argument("--csv", type=Path, default=Path("ml/eval/results/image_models.csv"))
    parser.add_argument("--latex", type=Path, default=Path("paper/tables/image_models.tex"))
    args = parser.parse_args()

    frame = collect(args.results_dir, args.onnx_dir)
    if frame.empty:
        raise SystemExit(f"no metrics.json files under {args.results_dir}")
    if frame["warning"].any():
        print("WARNING: remove these runs before using this table:")
        print(frame.loc[frame["warning"], "run"].to_string(index=False))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.csv, index=False, lineterminator="\n")
    args.latex.write_text(to_latex(frame), encoding="utf-8", newline="\n")
    print(frame.drop(columns="run").round(3).to_string(index=False))
    print(f"csv: {args.csv}\nlatex: {args.latex}")


if __name__ == "__main__":
    main()
