"""Integration tests with real ONNX models."""

import io

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from backend.app.core.config import settings
from backend.app.inference.audio import run_audio_inference
from backend.app.inference.image import run_image_inference


def generate_test_image_bytes(size: int = 224) -> bytes:
    """Generate valid RGB test image."""
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def generate_test_audio_bytes(duration_seconds: float = 4.0, sample_rate: int = 16000) -> bytes:
    """Generate 16 kHz synthetic speech-like test audio."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    buf = io.BytesIO()
    sf.write(buf, wave.astype(np.float32), sample_rate, format="WAV")
    return buf.getvalue()


def test_real_image_onnx_inference():
    image_onnx = settings.repo_root / "checkpoints/onnx_benchmark/bench-convnext_tiny-freq.onnx"
    if not image_onnx.exists():
        pytest.skip("Image ONNX model not found on disk.")

    image_bytes = generate_test_image_bytes(224)
    result_data, artefacts = run_image_inference(image_bytes)

    assert "probability" in result_data
    assert 0.0 <= result_data["probability"] <= 1.0
    assert result_data["verdict"] in ("likely_fake", "uncertain", "likely_real")
    assert result_data["model_name"] == "image-convnext-tiny-freq"
    assert result_data["model_version"] == "1.0.0"
    assert result_data["latency_ms"] >= 0

    assert "heatmap.png" in artefacts
    assert "spectrum.png" in artefacts
    assert len(artefacts["heatmap.png"]) > 0
    assert len(artefacts["spectrum.png"]) > 0


def test_real_audio_onnx_inference():
    audio_onnx = settings.repo_root / "checkpoints/onnx_benchmark/bench-audio-lcnn-lfcc.onnx"
    if not audio_onnx.exists():
        pytest.skip("Audio ONNX model not found on disk.")

    audio_bytes = generate_test_audio_bytes(4.0, 16000)
    result_data, artefacts = run_audio_inference(audio_bytes)

    assert "probability" in result_data
    assert 0.0 <= result_data["probability"] <= 1.0
    assert result_data["verdict"] in ("likely_fake", "uncertain", "likely_real")
    assert result_data["model_name"] == "audio-lcnn-lfcc"
    assert result_data["model_version"] == "1.0.0"
    assert result_data["latency_ms"] >= 0

    assert "saliency.png" in artefacts
    assert len(artefacts["saliency.png"]) > 0
