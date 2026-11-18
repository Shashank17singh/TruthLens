"""Tests for end-to-end analysis endpoints with mocked inference pipelines."""

import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

MOCK_IMAGE_RESULT = (
    {
        "model_name": "image-convnext-tiny-freq",
        "model_version": "1.0.0",
        "probability": 0.88,
        "decision_threshold": 0.50,
        "verdict": "likely_fake",
        "verdict_label": "Likely AI-generated",
        "confidence_band": {"low": 0.82, "high": 0.94},
        "explanation": "Unusual texture detected around the mouth and hairline.",
        "warnings": [],
        "disclaimer": "Results are probabilistic.",
        "latency_ms": 120,
        "details": {
            "face_detected": True,
            "regions": [{"box": [10, 10, 50, 50], "score": 0.88, "part": "face"}],
            "artefacts": {},
        },
    },
    {"heatmap.png": b"PNG_HEATMAP_DATA", "spectrum.png": b"PNG_SPECTRUM_DATA"},
)

MOCK_AUDIO_RESULT = (
    {
        "model_name": "audio-lcnn-lfcc",
        "model_version": "1.0.0",
        "probability": 0.12,
        "decision_threshold": 0.50,
        "verdict": "likely_real",
        "verdict_label": "Likely real",
        "confidence_band": {"low": 0.08, "high": 0.18},
        "explanation": "Natural vocal harmonics and breath patterns detected across all audio windows.",
        "warnings": [],
        "disclaimer": "Results are probabilistic.",
        "latency_ms": 350,
        "details": {
            "duration_seconds": 4.0,
            "aggregate": {"mean": 0.12, "max": 0.18},
            "windows": [{"start": 0.0, "end": 4.0, "probability": 0.12}],
            "artefacts": {},
        },
    },
    {"saliency.png": b"PNG_SALIENCY_DATA"},
)

MOCK_TEXT_RESULT = (
    {
        "model_name": "text-stats-ensemble",
        "model_version": "1.0.0",
        "probability": 0.92,
        "decision_threshold": 0.50,
        "verdict": "likely_fake",
        "verdict_label": "Likely AI",
        "confidence_band": {"low": 0.85, "high": 0.98},
        "explanation": "Uniform perplexity and low burstiness characteristic of large language models.",
        "warnings": [],
        "disclaimer": "This result is probabilistic.",
        "latency_ms": 45,
        "details": {
            "word_count": 120,
            "sentences": [{"start": 0, "end": 60, "text": "Sample AI text.", "probability": 0.92}],
            "artefacts": {},
        },
    },
    {},
)


def create_test_image_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_test_wav_bytes() -> bytes:
    # Minimal 1-second 16kHz WAV header + silence
    header = (
        b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00"
    )
    return header + b"\x00" * 2048


@patch("backend.app.worker.tasks.run_image_inference", return_value=MOCK_IMAGE_RESULT)
def test_submit_image_analysis(mock_run, client: TestClient, auth_headers):
    image_data = create_test_image_bytes()
    response = client.post(
        "/api/v1/analyze/image",
        headers=auth_headers,
        files={"file": ("test.jpg", image_data, "image/jpeg")},
    )
    assert response.status_code == 202
    data = response.json()
    assert "analysis_id" in data
    assert data["status"] == "queued"
    analysis_id = data["analysis_id"]

    # Verify GET /analyses/{id}
    res_get = client.get(f"/api/v1/analyses/{analysis_id}", headers=auth_headers)
    assert res_get.status_code == 200
    analysis = res_get.json()
    assert analysis["status"] == "completed"
    assert analysis["modality"] == "image"
    assert analysis["result"]["verdict"] == "likely_fake"
    assert analysis["result"]["probability"] == 0.88


@patch("backend.app.worker.tasks.run_audio_inference", return_value=MOCK_AUDIO_RESULT)
def test_submit_audio_analysis(mock_run, client: TestClient, api_key_headers):
    audio_data = create_test_wav_bytes()
    response = client.post(
        "/api/v1/analyze/audio",
        headers=api_key_headers,
        files={"file": ("test.wav", audio_data, "audio/wav")},
    )
    assert response.status_code == 202
    data = response.json()
    analysis_id = data["analysis_id"]

    res_get = client.get(f"/api/v1/analyses/{analysis_id}", headers=api_key_headers)
    assert res_get.status_code == 200
    analysis = res_get.json()
    assert analysis["status"] == "completed"
    assert analysis["modality"] == "audio"
    assert analysis["result"]["verdict"] == "likely_real"


@patch("backend.app.worker.tasks.run_text_inference", return_value=MOCK_TEXT_RESULT)
def test_submit_text_analysis(mock_run, client: TestClient, auth_headers):
    text_content = "This is a detailed paragraph of text submitted to TruthLens for AI-generated detection evaluation."
    response = client.post(
        "/api/v1/analyze/text",
        headers=auth_headers,
        json={"text": text_content},
    )
    assert response.status_code == 202
    analysis_id = response.json()["analysis_id"]

    res_get = client.get(f"/api/v1/analyses/{analysis_id}", headers=auth_headers)
    assert res_get.status_code == 200
    analysis = res_get.json()
    assert analysis["status"] == "completed"
    assert analysis["result"]["verdict_label"] == "Likely AI"


@patch("backend.app.worker.tasks.run_text_inference", return_value=MOCK_TEXT_RESULT)
def test_feedback_and_history_flow(mock_run, client: TestClient, auth_headers):
    # Create analysis
    response = client.post(
        "/api/v1/analyze/text",
        headers=auth_headers,
        json={"text": "Another test text for history verification."},
    )
    analysis_id = response.json()["analysis_id"]

    # Test history listing
    list_res = client.get("/api/v1/analyses?page=1&page_size=10", headers=auth_headers)
    assert list_res.status_code == 200
    history = list_res.json()
    assert history["total"] >= 1
    assert any(item["id"] == analysis_id for item in history["items"])

    # Test save analysis
    save_res = client.post(f"/api/v1/analyses/{analysis_id}/save", headers=auth_headers)
    assert save_res.status_code == 200
    assert save_res.json()["retain_media"] is True

    # Test feedback submission
    feedback_res = client.post(
        f"/api/v1/analyses/{analysis_id}/feedback",
        headers=auth_headers,
        json={"is_correct": True, "reported_label": "fake", "comment": "Spot-on detection."},
    )
    assert feedback_res.status_code == 201
    assert feedback_res.json()["is_correct"] is True

    # Duplicate feedback must be rejected with 409
    dup_res = client.post(
        f"/api/v1/analyses/{analysis_id}/feedback",
        headers=auth_headers,
        json={"is_correct": False},
    )
    assert dup_res.status_code == 409
    assert dup_res.json()["error"]["code"] == "feedback_exists"


@patch("backend.app.worker.tasks.run_text_inference", return_value=MOCK_TEXT_RESULT)
def test_report_and_verify_flow(mock_run, client: TestClient, auth_headers):
    response = client.post(
        "/api/v1/analyze/text",
        headers=auth_headers,
        json={"text": "Text for PDF report and HMAC verification test."},
    )
    analysis_id = response.json()["analysis_id"]

    # Request report
    rep_res = client.post(f"/api/v1/analyses/{analysis_id}/report", headers=auth_headers)
    assert rep_res.status_code in (200, 202)
    report_id = rep_res.json()["report_id"]

    # Download report PDF
    pdf_res = client.get(f"/api/v1/reports/{report_id}.pdf", headers=auth_headers)
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    pdf_bytes = pdf_res.content
    assert pdf_bytes.startswith(b"%PDF")

    # Verify genuine PDF
    verify_res = client.post(
        "/api/v1/verify",
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
    )
    assert verify_res.status_code == 200
    assert verify_res.json()["genuine"] is True
    assert verify_res.json()["report_id"] == report_id

    # Verify tampered PDF fails
    tampered_bytes = pdf_bytes[:-10] + b"TAMPERED!!"
    tamper_res = client.post(
        "/api/v1/verify",
        files={"file": ("report.pdf", tampered_bytes, "application/pdf")},
    )
    assert tamper_res.status_code == 200
    assert tamper_res.json()["genuine"] is False
