import numpy as np
import pandas as pd
import pytest

from ml.common import metrics


def test_classification_metrics_counts_each_outcome():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.3, 0.9])
    result = metrics.classification_metrics(labels, probabilities, 0.5)
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]


def test_classification_metrics_skips_auc_for_one_class():
    assert metrics.classification_metrics(np.array([1, 1]), np.array([0.2, 0.9]), 0.5)["roc_auc"] is None


def test_ece_is_small_when_probabilities_are_honest():
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(size=200_000)
    labels = (rng.uniform(size=probabilities.size) < probabilities).astype(int)
    assert metrics.expected_calibration_error(labels, probabilities) < 0.01


def test_ece_is_large_when_the_model_is_overconfident():
    labels = np.array([0, 1] * 500)
    assert metrics.expected_calibration_error(labels, np.full(1000, 0.99)) == pytest.approx(0.49)


def test_reliability_bins_cover_all_samples():
    probabilities = np.array([0.0, 0.5, 1.0])
    rows = metrics.reliability_bins(np.array([0, 1, 1]), probabilities)
    assert sum(row["count"] for row in rows) == 3


def honest_sample(seed: int, size: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2, size)
    return logits, (rng.uniform(size=size) < metrics.sigmoid(logits)).astype(int)


def test_calibration_recovers_known_scale_and_shift():
    logits, labels = honest_sample(1)
    fitted = metrics.fit_calibration(labels, logits * 2.5 - 2.5)
    assert fitted["temperature"] == pytest.approx(2.5, abs=0.1)
    assert fitted["bias"] == pytest.approx(1.0, abs=0.05)


def test_calibration_fixes_a_shift():
    logits, labels = honest_sample(2)
    shifted = logits - 1.5
    fitted = metrics.fit_calibration(labels, shifted)
    before = metrics.expected_calibration_error(labels, metrics.sigmoid(shifted))
    after = metrics.expected_calibration_error(labels, metrics.calibrated_probability(shifted, fitted))
    assert after < 0.01
    assert after < before / 5


def test_calibration_keeps_the_ranking():
    logits, labels = honest_sample(3, 1000)
    fitted = metrics.fit_calibration(labels, logits * 3)
    calibrated = metrics.calibrated_probability(logits * 3, fitted)
    assert (np.argsort(calibrated) == np.argsort(logits)).all()


def test_calibration_stays_bounded_for_a_model_worse_than_chance():
    logits, labels = honest_sample(5, 2000)
    fitted = metrics.fit_calibration(labels, -logits)
    assert fitted["temperature"] <= metrics.TEMPERATURE_BOUNDS[1]
    calibrated = metrics.calibrated_probability(-logits, fitted)
    assert len(np.unique(calibrated)) == len(logits)
    assert metrics.expected_calibration_error(labels, calibrated) > 0.1


def test_calibration_preserves_auc_on_a_broken_model():
    logits, labels = honest_sample(6, 2000)
    flipped = (-logits).astype(np.float32)
    fitted = metrics.fit_calibration(labels, flipped)
    calibrated = metrics.calibrated_probability(flipped, fitted)
    before = metrics.classification_metrics(labels, metrics.sigmoid(flipped.astype(np.float64)), 0.5)["roc_auc"]
    assert metrics.classification_metrics(labels, calibrated, 0.5)["roc_auc"] == pytest.approx(before, abs=1e-6)


def test_quality_warning_flags_only_chance_level_models():
    assert metrics.quality_warnings(0.43)
    assert metrics.quality_warnings(0.55)
    assert metrics.quality_warnings(0.90) == []


def test_calibration_handles_float32_logits():
    logits, labels = honest_sample(4, 5000)
    fitted = metrics.fit_calibration(labels, (logits * 2).astype(np.float32))
    assert fitted["temperature"] == pytest.approx(2.0, abs=0.2)


def test_threshold_keeps_the_highest_recall_at_the_target():
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.6, 0.5, 0.7, 0.8, 0.9])
    choice = metrics.threshold_for_precision(labels, probabilities, 0.75)
    assert choice["reached"] is True
    assert choice["threshold"] == 0.5
    assert choice["recall"] == 1.0
    assert choice["precision"] == pytest.approx(0.8)


def test_threshold_flags_an_unreachable_target():
    labels = np.array([1, 0])
    choice = metrics.threshold_for_precision(labels, np.array([0.4, 0.9]), 0.95)
    assert choice["reached"] is False
    assert choice["precision"] < 0.95


def test_grouped_interval_is_wider_than_treating_frames_as_independent():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(40), 25)
    group_effect = np.repeat(rng.normal(0, 1, 40), 25)
    labels = np.repeat(np.array([0, 1] * 20), 25)
    probabilities = metrics.sigmoid(labels + group_effect + rng.normal(0, 0.1, groups.size))
    grouped = metrics.grouped_auc_interval(labels, probabilities, groups, seed=0, rounds=200)
    independent = metrics.grouped_auc_interval(labels, probabilities, np.arange(groups.size), seed=0, rounds=200)
    assert grouped["high"] - grouped["low"] > 2 * (independent["high"] - independent["low"])
    assert grouped["groups"] == 40


def test_per_generator_accuracy_treats_real_as_specificity():
    frame = pd.DataFrame({
        "generator_name": ["real", "real", "gan", "gan"],
        "label": [0, 0, 1, 1],
        "probability": [0.1, 0.9, 0.8, 0.2],
    })
    table = metrics.per_generator_accuracy(frame, 0.5).set_index("generator_name")
    assert table.loc["real", "accuracy"] == 0.5
    assert table.loc["gan", "samples"] == 2
