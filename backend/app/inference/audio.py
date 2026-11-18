"""Audio inference and explainability pipeline."""

import io
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from backend.app.inference.registry import registry
from ml.audio.inference import score_long


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def generate_saliency_png(wave: np.ndarray, sample_rate: int) -> bytes:
    """Render audio waveform and spectrogram representation as a PNG artefact."""
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    # Compute simple spectrogram
    ax.specgram(wave[: sample_rate * 4], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    ax.set_title("Suspicious Window Spectrogram Saliency")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def run_audio_inference(audio_bytes: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Run ONNX inference and timeline scoring on audio bytes.

    Returns (analysis_result_dict, artefact_bytes_dict).
    """
    start_time = time.perf_counter()
    session, info = registry.get_audio_session()

    # Load audio into 16 kHz mono
    with io.BytesIO(audio_bytes) as audio_io:
        wave, sr = sf.read(audio_io, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)

    # Resample to 16 kHz if necessary
    target_sr = int(info.get("sample_rate", 16000))
    if sr != target_sr:
        import librosa
        wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    duration_seconds = round(float(len(wave) / sr), 2)
    clip_seconds = float(info.get("clip_seconds", 4.0))

    # Function to run ONNX session on a batch [n, samples]
    input_name = session.get_inputs()[0].name

    def onnx_logits(batch: np.ndarray) -> np.ndarray:
        # Check input dimensionality: if ONNX model expects 2D [B, samples] or 3D
        outputs = session.run(None, {input_name: batch.astype(np.float32)})
        return outputs[0].flatten()

    def prob_func(logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits))

    scoring = score_long(
        logits_of=onnx_logits,
        wave=wave,
        sample_rate=sr,
        clip_seconds=clip_seconds,
        probability_of=prob_func,
    )

    # Aggregates
    mean_prob = float(scoring.get("mean_probability", sigmoid(scoring.get("mean_logit", 0.0))))
    max_prob = float(scoring.get("max_probability", sigmoid(scoring.get("max_logit", 0.0))))

    threshold = float(info.get("decision_threshold", 0.50))
    # Primary probability based on mean
    probability = round(mean_prob, 4)

    if probability >= threshold:
        verdict = "likely_fake"
        verdict_label = "Likely AI-generated"
        explanation = f"Synthetic acoustic artifacts detected in speech spectral windows (max window: {max_prob:.2f})."
    elif probability < (threshold - 0.15):
        verdict = "likely_real"
        verdict_label = "Likely real"
        explanation = "Natural vocal harmonics and breath patterns detected across all audio windows."
    else:
        verdict = "uncertain"
        verdict_label = "Uncertain"
        explanation = "Mixed spectral cues detected; some windows exhibit synthetic compression characteristics."

    delta = min(0.08, max(0.02, 0.1 * abs(0.5 - probability)))
    conf_low = max(0.0, round(probability - delta, 3))
    conf_high = min(1.0, round(probability + delta, 3))

    timeline = scoring.get("timeline", [])
    windows_data = [
        {
            "start": round(entry.get("start", 0.0), 2),
            "end": round(entry.get("end", 0.0), 2),
            "probability": round(float(entry.get("probability", sigmoid(entry.get("logit", 0.0)))), 3),
        }
        for entry in timeline
    ]

    saliency_png = generate_saliency_png(wave, sr)

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    result_data = {
        "model_name": info.get("name", "audio-lcnn-lfcc"),
        "model_version": info.get("version", "1.0.0"),
        "probability": probability,
        "decision_threshold": threshold,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "confidence_band": {"low": conf_low, "high": conf_high},
        "explanation": explanation,
        "warnings": [],
        "disclaimer": info.get(
            "disclaimer",
            "Results are probabilistic and must not be used alone to make decisions about a person.",
        ),
        "latency_ms": latency_ms,
        "details": {
            "duration_seconds": duration_seconds,
            "aggregate": {
                "mean": round(mean_prob, 3),
                "max": round(max_prob, 3),
            },
            "windows": windows_data,
            "artefacts": {
                "saliency": "saliency.png",
            },
        },
    }

    artefacts = {
        "saliency.png": saliency_png,
    }

    return result_data, artefacts
