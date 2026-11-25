"""Calibration and verdicts shared by the image and audio pipelines.

Training fits calibration as sigmoid(logit / temperature + bias) (ml/common/metrics.py),
so serving must apply the same formula with the values from the model's metrics.json.
"""

from typing import Any

import numpy as np

UNCERTAIN_BAND = 0.15


def calibrated_probability(logits: Any, info: dict[str, Any]) -> Any:
    temperature = float(info.get("temperature", 1.0))
    bias = float(info.get("bias", 0.0))
    probability = 1.0 / (1.0 + np.exp(-(np.asarray(logits, dtype=np.float64) / temperature + bias)))
    return float(probability) if np.ndim(probability) == 0 else probability


def verdict(probability: float, threshold: float) -> str:
    """Below the threshold, a band of uncertainty; never wider than half the threshold, so small validated thresholds still allow "likely_real"."""
    if probability >= threshold:
        return "likely_fake"
    if probability < max(threshold - UNCERTAIN_BAND, threshold / 2):
        return "likely_real"
    return "uncertain"
