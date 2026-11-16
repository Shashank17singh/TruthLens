"""Statistical features, the logistic-regression meta-classifier, and its evaluation.

    python -m ml.text.ensemble extract --manifest data/manifests/mage.csv --split train --limit 3000 --output features/mage-train.parquet
    python -m ml.text.ensemble fit --train features/mage-train.parquet --output checkpoints/text/meta.joblib
    python -m ml.text.ensemble evaluate --model checkpoints/text/meta.joblib --validation features/mage-val.parquet \
        --test features/mage-test.parquet --extra features/hc3.parquet features/bias.parquet --output ml/eval/results/text

Extraction appends to its output as it goes, so an interrupted run resumes.
A fine-tuned transformer joins the ensemble as one more column, `transformer_logit`,
when `--checkpoint` is given at extraction.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.common import metrics
from ml.common.manifest import read_manifest
from ml.common.training import resolve_device
from ml.text.data import read_texts
from ml.text.datasets import prompt_group
from ml.text.features import FEATURES, ModelPair, text_features

TRANSFORMER = "transformer_logit"
SAVE_EVERY = 100


def sample(frame: pd.DataFrame, split: str | None, limit: int | None, seed: int) -> pd.DataFrame:
    """A label- and generator-stratified sample, so small runs still cover every generator."""
    part = frame if split is None else frame[frame["split"] == split]
    if limit is None or limit >= len(part):
        return part.reset_index(drop=True)
    shuffled = part.sample(frac=1.0, random_state=seed)
    strata = shuffled.groupby(["label", "generator_name"])
    quota = strata["label"].transform("size").mul(limit / len(part)).round().clip(lower=1)
    return shuffled[strata.cumcount() < quota].reset_index(drop=True)


def opens_like_training(frame: pd.DataFrame, root: Path, training: pd.DataFrame) -> np.ndarray:
    """Rows that open with the same words as a training or validation text (MAGE groups by opening words)."""
    seen = set(training.loc[training["split"].isin(["train", "val"]), "group_id"])
    return read_texts(frame, root).map(prompt_group).isin(seen).to_numpy()


def extract(frame: pd.DataFrame, root: Path, output: Path, pair: ModelPair, transformer=None) -> pd.DataFrame:
    """One feature row per manifest row, keyed by path. `transformer` maps text to a logit."""
    done = pd.read_parquet(output) if output.exists() else pd.DataFrame(columns=["path"])
    todo = frame[~frame["path"].isin(done["path"])].reset_index(drop=True)
    texts = read_texts(todo, root)
    rows = []
    for index, text in enumerate(texts):
        row = {**todo.iloc[index].to_dict(), **text_features(pair, text)}
        if transformer is not None:
            row[TRANSFORMER] = transformer(text)
        rows.append(row)
        if len(rows) == SAVE_EVERY or index == len(texts) - 1:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            done.to_parquet(output, index=False)
            rows = []
            print(f"{len(done)}/{len(frame)} texts", flush=True)
    return done[done["path"].isin(frame["path"])].reset_index(drop=True)


def columns_of(frame: pd.DataFrame) -> list[str]:
    return [*FEATURES, *([TRANSFORMER] if TRANSFORMER in frame else [])]


def fit(train: pd.DataFrame, seed: int = 1337):
    """Scaled logistic regression: few features, so a linear model is enough and its weights are readable."""
    columns = columns_of(train)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    model.fit(train[columns].to_numpy(), train["label"].to_numpy())
    model.feature_names = columns
    return model


def probabilities(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[model.feature_names].to_numpy())[:, 1]


def weights(model) -> dict[str, float]:
    """Standardized coefficients: positive pushes toward AI-written."""
    return dict(zip(model.feature_names, model[-1].coef_[0].round(4).tolist()))


def false_positive_rates(frame: pd.DataFrame, threshold: float) -> list[dict]:
    """Share of human texts flagged as AI, per domain. This is the non-native bias check on detector_bias."""
    human = frame[frame["label"] == 0]
    flagged = (human["probability"] >= threshold).groupby(human["domain"]).agg(["mean", "size"])
    return [{"domain": domain, "false_positive_rate": float(row["mean"]), "samples": int(row["size"])} for domain, row in flagged.iterrows()]


def report_part(frame: pd.DataFrame, threshold: float, seed: int) -> dict:
    labels = frame["label"].to_numpy()
    result = metrics.classification_metrics(labels, frame["probability"].to_numpy(), threshold)
    if len(set(labels)) == 2:
        result["auc_interval"] = metrics.grouped_auc_interval(labels, frame["probability"].to_numpy(), frame["group_id"].to_numpy(), seed)
    result["per_generator"] = metrics.per_generator_accuracy(frame, threshold).to_dict("records")
    return result


def evaluate(model, validation: pd.DataFrame, parts: dict[str, pd.DataFrame], output: Path, seed: int = 1337) -> dict:
    """Threshold at the validation EER point, then score every part with it."""
    validation = validation.assign(probability=probabilities(model, validation))
    threshold = metrics.equal_error_rate(validation["label"].to_numpy(), validation["probability"].to_numpy())["threshold"]
    report = {"weights": weights(model), "threshold": float(threshold), "validation": report_part(validation, threshold, seed)}
    for name, frame in parts.items():
        frame = frame.assign(probability=probabilities(model, frame))
        report[name] = report_part(frame, threshold, seed)
        report[name]["human_false_positive_rate_by_domain"] = false_positive_rates(frame, threshold)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    extract_command = commands.add_parser("extract")
    extract_command.add_argument("--manifest", type=Path, required=True)
    extract_command.add_argument("--split")
    extract_command.add_argument("--limit", type=int)
    extract_command.add_argument("--exclude-groups-of", type=Path, help="drop rows whose group appears in this manifest's train and val splits")
    extract_command.add_argument("--data-root", type=Path, default=Path("data/processed"))
    extract_command.add_argument("--checkpoint", type=Path, help="fine-tuned transformer to add as a feature")
    extract_command.add_argument("--output", type=Path, required=True)
    extract_command.add_argument("--seed", type=int, default=1337)
    extract_command.add_argument("--device", default="auto", help="auto picks CUDA when it is available")
    fit_command = commands.add_parser("fit")
    fit_command.add_argument("--train", type=Path, required=True)
    fit_command.add_argument("--output", type=Path, required=True)
    evaluate_command = commands.add_parser("evaluate")
    evaluate_command.add_argument("--model", type=Path, required=True)
    evaluate_command.add_argument("--validation", type=Path, required=True)
    evaluate_command.add_argument("--test", type=Path, nargs="+", required=True, help="feature files; each is reported under its file name")
    evaluate_command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "extract":
        frame = read_manifest(args.manifest)
        if args.exclude_groups_of:
            frame = frame[~opens_like_training(frame, args.data_root, read_manifest(args.exclude_groups_of))]
        device = resolve_device(args.device)
        transformer = None
        if args.checkpoint:
            from ml.text.model import load_checkpoint, score_text

            model, tokenizer, config = load_checkpoint(args.checkpoint, device)
            transformer = lambda text: score_text(model, tokenizer, text, config["max_length"], config["stride"])
        extract(sample(frame, args.split, args.limit, args.seed), args.data_root, args.output, ModelPair.pretrained(device=device), transformer)
    elif args.command == "fit":
        model = fit(pd.read_parquet(args.train))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.output)
        print(weights(model))
    else:
        model = joblib.load(args.model)
        parts = {path.stem: pd.read_parquet(path) for path in args.test}
        report = evaluate(model, pd.read_parquet(args.validation), parts, args.output)
        print(json.dumps({name: part.get("roc_auc") for name, part in report.items() if isinstance(part, dict) and "roc_auc" in part}, indent=2))


if __name__ == "__main__":
    main()
