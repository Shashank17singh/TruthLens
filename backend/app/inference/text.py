"""Text inference pipeline combining statistical features and ensemble classifier."""

import time
from typing import Any

from backend.app.inference.registry import registry
from ml.text.detector import DISCLAIMER, TextDetector


def run_text_inference(text: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Run text AI-generation detection.

    Returns (analysis_result_dict, artefact_bytes_dict).
    """
    start_time = time.perf_counter()
    meta, pair, info = registry.get_text_components()

    version = str(info.get("version", "1.0.0"))
    detector = TextDetector(pair=pair, meta=meta, version=version)
    analysis = detector.analyze(text)

    probability = float(analysis["probability"])
    label = analysis["label"]  # 'likely human', 'mixed', 'likely AI'

    # Map label to standard database verdict
    verdict_map = {
        "likely human": "likely_real",
        "mixed": "uncertain",
        "likely AI": "likely_fake",
    }
    verdict = verdict_map.get(label, "uncertain")

    verdict_label_map = {
        "likely human": "Likely human",
        "mixed": "Mixed",
        "likely AI": "Likely AI",
    }
    verdict_label = verdict_label_map.get(label, "Mixed")

    if verdict == "likely_fake":
        explanation = "Uniform perplexity and low burstiness characteristic of large language models."
    elif verdict == "likely_real":
        explanation = "High syntactic variance, dynamic burstiness, and natural human vocabulary distribution."
    else:
        explanation = "Mixed linguistic patterns indicating human-edited AI text or varied composition style."

    threshold = float(info.get("decision_threshold", 0.50))
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    result_data = {
        "model_name": info.get("name", "text-stats-ensemble"),
        "model_version": version,
        "probability": round(probability, 4),
        "decision_threshold": threshold,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "confidence_band": analysis.get("confidence_band", {"low": 0.0, "high": 1.0}),
        "explanation": explanation,
        "warnings": analysis.get("warnings", []),
        "disclaimer": analysis.get("disclaimer", DISCLAIMER),
        "latency_ms": latency_ms,
        "details": {
            "word_count": analysis.get("words", len(text.split())),
            "sentences": analysis.get("sentences", []),
            "artefacts": {},
        },
    }

    return result_data, {}
