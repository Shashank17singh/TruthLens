"""Serving must apply calibration the way training fitted it: sigmoid(logit / temperature + bias)."""

import math

import numpy as np

from backend.app.inference.calibration import calibrated_probability, verdict


def test_matches_training_formula():
    info = {"temperature": 0.3868, "bias": -1.3786}
    expected = 1 / (1 + math.exp(-(2.0 / 0.3868 - 1.3786)))
    assert math.isclose(calibrated_probability(2.0, info), expected, rel_tol=1e-9)


def test_works_on_arrays_and_defaults_to_identity():
    logits = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(calibrated_probability(logits, {}), 1 / (1 + np.exp(-logits)))


def test_small_validated_threshold_still_allows_likely_real():
    assert verdict(0.0, 0.0005) == "likely_real"
    assert verdict(0.0003, 0.0005) == "uncertain"
    assert verdict(0.0005, 0.0005) == "likely_fake"


def test_default_threshold_keeps_the_old_band():
    assert verdict(0.30, 0.5) == "likely_real"
    assert verdict(0.40, 0.5) == "uncertain"
    assert verdict(0.50, 0.5) == "likely_fake"
