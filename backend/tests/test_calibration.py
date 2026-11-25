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


def test_audio_score_is_calibrated(monkeypatch):
    import io

    import soundfile as sf

    from backend.app.inference import audio as audio_module

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "wave"})()]

        def run(self, outputs, feeds):
            return [np.ones(len(feeds["wave"]), dtype=np.float32)]

    info = {"onnx_file": "x.onnx", "temperature": 0.5, "bias": 1.0, "decision_threshold": 0.5}
    monkeypatch.setattr(audio_module.registry, "get_audio_session", lambda: (Session(), info))
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(16000 * 5, dtype=np.float32), 16000, format="WAV")

    result, _ = audio_module.run_audio_inference(buffer.getvalue())

    assert math.isclose(result["probability"], 1 / (1 + math.exp(-3.0)), rel_tol=1e-5)
