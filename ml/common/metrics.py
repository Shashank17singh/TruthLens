"""Evaluation metrics shared by all three modalities."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ECE_BINS = 15
BOOTSTRAP_ROUNDS = 1000
TEMPERATURE_BOUNDS = (0.05, 20.0)
BIAS_BOUNDS = (-10.0, 10.0)
CHANCE_AUC = 0.55


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        "confusion_matrix": matrix.tolist(),
        "samples": int(len(labels)),
    }


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = ECE_BINS) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for index in range(bins):
        selected = indices == index
        rows.append({
            "low": float(edges[index]),
            "high": float(edges[index + 1]),
            "count": int(selected.sum()),
            "confidence": float(probabilities[selected].mean()) if selected.any() else None,
            "accuracy": float(labels[selected].mean()) if selected.any() else None,
        })
    return rows


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = ECE_BINS) -> float:
    total = len(labels)
    return float(sum(
        row["count"] / total * abs(row["confidence"] - row["accuracy"])
        for row in reliability_bins(labels, probabilities, bins)
        if row["count"]
    ))


def fit_calibration(labels: np.ndarray, logits: np.ndarray) -> dict:
    """Fit a temperature and a bias on validation logits.

    Both preserve the ranking, so AUC stays the same. The temperature undoes the
    low confidence that label smoothing trains in. The bias undoes the shift
    caused by training on a balanced sampler and validating on another class mix,
    which a temperature alone cannot move.
    """
    logits = logits.astype(np.float64)

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        scaled = logits / np.exp(parameters[0]) + parameters[1]
        return float(np.mean(np.logaddexp(0.0, -scaled) * labels + np.logaddexp(0.0, scaled) * (1 - labels)))

    # Unbounded, a model worse than chance sends the temperature to infinity. Every
    # probability then becomes 0.5, which reads as perfect calibration.
    bounds = [np.log(TEMPERATURE_BOUNDS), BIAS_BOUNDS]
    result = minimize(
        negative_log_likelihood, np.zeros(2), method="Nelder-Mead", bounds=bounds,
        options={"xatol": 1e-4, "fatol": 1e-7},
    )
    return {"temperature": float(np.exp(result.x[0])), "bias": float(result.x[1])}


def calibrated_probability(logits: np.ndarray, calibration: dict) -> np.ndarray:
    return sigmoid(logits.astype(np.float64) / calibration["temperature"] + calibration["bias"])


def quality_warnings(validation_auc: float) -> list[str]:
    """Flags for results that must not be read as a working model."""
    if validation_auc <= CHANCE_AUC:
        return [f"validation AUC {validation_auc:.3f} is at or below chance, so these metrics are not meaningful"]
    return []


def threshold_for_precision(labels: np.ndarray, probabilities: np.ndarray, target: float) -> dict:
    """Pick the threshold with the highest recall that still meets the target precision."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    meets = precision >= target
    if meets.any():
        index = int(np.flatnonzero(meets)[np.argmax(recall[meets])])
    else:
        index = int(np.argmax(precision))
    return {
        "threshold": float(thresholds[index]),
        "target_precision": target,
        "reached": bool(meets.any()),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
    }


def precision_tradeoff(labels: np.ndarray, probabilities: np.ndarray, targets: tuple[float, ...]) -> list[dict]:
    return [threshold_for_precision(labels, probabilities, target) for target in targets]


def grouped_auc_interval(
    labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray, seed: int, rounds: int = BOOTSTRAP_ROUNDS
) -> dict:
    """Bootstrap whole groups, because frames of one person are not independent samples."""
    codes, unique = pd.factorize(groups)
    members = [np.flatnonzero(codes == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(rounds):
        chosen = rng.integers(0, len(members), len(members))
        sample = np.concatenate([members[index] for index in chosen])
        if len(set(labels[sample])) == 2:
            scores.append(roc_auc_score(labels[sample], probabilities[sample]))
    return {
        "low": float(np.percentile(scores, 2.5)),
        "high": float(np.percentile(scores, 97.5)),
        "rounds": len(scores),
        "groups": len(members),
    }


def per_generator_accuracy(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Accuracy per generator. For fakes this is the detection rate, for `real` the specificity."""
    correct = (frame["probability"] >= threshold).astype(int) == frame["label"]
    table = correct.groupby(frame["generator_name"]).agg(["mean", "size"])
    return table.rename(columns={"mean": "accuracy", "size": "samples"}).reset_index()
