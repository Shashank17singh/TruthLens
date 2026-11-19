"""Unit and integration tests for PDF report generation and HMAC verification."""

import hashlib
import io
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from backend.app.core.security import generate_uuidv7
from backend.app.db.models import Analysis, AnalysisResult, Report, User
from backend.app.reports.generator import (
    generate_qr_bytes,
    generate_qr_data_uri,
    render_pdf_bytes,
    render_report_task,
    sign_report,
    verify_report_hmac,
)
from backend.app.storage.minio_client import storage


def test_qr_generation():
    """Verify QR code PNG bytes and base64 URI generation."""
    test_url = "https://truthlens.ai/verify?report_id=rep_12345"
    raw_bytes = generate_qr_bytes(test_url)
    assert raw_bytes.startswith(b"\x89PNG")

    data_uri = generate_qr_data_uri(test_url)
    assert data_uri.startswith("data:image/png;base64,")


def test_hmac_signing_and_tamper_detection():
    """Verify HMAC signature generation and constant-time tampering detection."""
    report_id = "rep_019318b8-0000-7000-8000-000000000001"
    pdf_content = b"%PDF-1.4 Fake test PDF content for cryptographic test"
    pdf_sha256 = hashlib.sha256(pdf_content).hexdigest()

    signature = sign_report(report_id, pdf_sha256)
    assert isinstance(signature, str)
    assert len(signature) == 64  # SHA-256 hex string

    # 1. Correct signature matches
    assert verify_report_hmac(report_id, pdf_sha256, signature) is True

    # 2. Tampered content fails
    tampered_content = pdf_content + b" modified"
    tampered_sha256 = hashlib.sha256(tampered_content).hexdigest()
    assert verify_report_hmac(report_id, tampered_sha256, signature) is False

    # 3. Mismatched report ID fails
    assert verify_report_hmac("rep_different_id", pdf_sha256, signature) is False


def test_render_pdf_bytes_all_modalities(db_session):
    """Verify PDF rendering engine generates valid PDF bytes across image, audio, and text."""
    user = User(id=generate_uuidv7(), email="reporter@truthlens.ai", password_hash="hash")
    db_session.add(user)

    # 1. Image analysis
    img_analysis = Analysis(
        id=generate_uuidv7(),
        user_id=user.id,
        modality="image",
        status="completed",
        file_name="portrait_deepfake.png",
        sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        completed_at=datetime.now(UTC),
    )
    img_result = AnalysisResult(
        analysis_id=img_analysis.id,
        probability=0.88,
        decision_threshold=0.5,
        confidence_low=0.82,
        confidence_high=0.94,
        verdict="likely_fake",
        explanation="ConvNeXt and 2D DCT frequency analysis detected grid upsampling artifacts.",
        model_name="bench-convnext_tiny-freq",
        model_version="1.0.0",
        details={
            "regions": [
                {"part": "focal_face", "box": [0.1, 0.2, 0.7, 0.8], "score": 0.91}
            ]
        },
    )
    pdf_bytes_img = render_pdf_bytes("rep_img_01", img_analysis, img_result, "pub_tok_01")
    assert pdf_bytes_img.startswith(b"%PDF")
    assert len(pdf_bytes_img) > 500

    # 2. Audio analysis
    audio_analysis = Analysis(
        id=generate_uuidv7(),
        user_id=user.id,
        modality="audio",
        status="completed",
        file_name="speech_sample.wav",
        completed_at=datetime.now(UTC),
    )
    audio_result = AnalysisResult(
        analysis_id=audio_analysis.id,
        probability=0.74,
        decision_threshold=0.5,
        confidence_low=0.68,
        confidence_high=0.80,
        verdict="likely_fake",
        explanation="LCNN detected phase discontinuities in vocoder synthesis across temporal windows.",
        model_name="bench-audio-lcnn-lfcc",
        model_version="1.0.0",
        details={
            "windows": [
                {"start": 0.0, "end": 4.0, "probability": 0.85},
                {"start": 2.0, "end": 6.0, "probability": 0.63},
            ]
        },
    )
    pdf_bytes_audio = render_pdf_bytes("rep_audio_01", audio_analysis, audio_result, "pub_tok_02")
    assert pdf_bytes_audio.startswith(b"%PDF")
    assert len(pdf_bytes_audio) > 500

    # 3. Text analysis
    text_analysis = Analysis(
        id=generate_uuidv7(),
        user_id=user.id,
        modality="text",
        status="completed",
        file_name="essay_input.txt",
        completed_at=datetime.now(UTC),
    )
    text_result = AnalysisResult(
        analysis_id=text_analysis.id,
        probability=0.25,
        decision_threshold=0.5,
        confidence_low=0.18,
        confidence_high=0.32,
        verdict="likely_real",
        explanation="Stylometric perplexity and burstiness indicate human organic variance.",
        model_name="bench-text-meta",
        model_version="1.0.0",
        details={
            "word_count": 120,
            "sentences": [
                {"text": "Human authors write with natural perplexity variance.", "probability": 0.15},
                {"text": "Consequently, this sentence is genuine.", "probability": 0.35},
            ],
        },
    )
    pdf_bytes_text = render_pdf_bytes("rep_text_01", text_analysis, text_result, "pub_tok_03")
    assert pdf_bytes_text.startswith(b"%PDF")
    assert len(pdf_bytes_text) > 500


def test_render_report_celery_task_and_verification_api(client: TestClient, db_session, monkeypatch):
    """Verify Celery task renders, caches in storage, and passes cryptographic /verify checks."""
    user = User(id=generate_uuidv7(), email="auditor@truthlens.ai", password_hash="hash")
    db_session.add(user)

    analysis = Analysis(
        id=generate_uuidv7(),
        user_id=user.id,
        modality="image",
        status="completed",
        file_name="sample_verify.png",
        sha256="1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
        completed_at=datetime.now(UTC),
    )
    db_session.add(analysis)

    result = AnalysisResult(
        analysis_id=analysis.id,
        probability=0.92,
        decision_threshold=0.5,
        confidence_low=0.86,
        confidence_high=0.98,
        verdict="likely_fake",
        explanation="ConvNeXt frequency artifact confirmed.",
        model_name="bench-convnext_tiny-freq",
        model_version="1.0.0",
    )
    db_session.add(result)

    report = Report(
        id=generate_uuidv7(),
        analysis_id=analysis.id,
        status="queued",
        public_token="token_audit_99",
    )
    db_session.add(report)
    db_session.commit()

    # Execute report rendering task
    render_report_task(report.id)

    db_session.refresh(report)
    assert report.status == "ready"
    assert report.sha256 is not None
    assert report.hmac is not None
    assert report.object_key == f"reports/{report.id}.pdf"

    # Verify cached file in storage
    pdf_data = storage.get_object(report.object_key)
    assert pdf_data.startswith(b"%PDF")
    assert hashlib.sha256(pdf_data).hexdigest() == report.sha256

    # Test POST /api/v1/verify with valid PDF file
    upload_file = io.BytesIO(pdf_data)
    res = client.post(
        "/api/v1/verify",
        files={"file": ("report.pdf", upload_file, "application/pdf")},
    )
    assert res.status_code == 200
    vdata = res.json()
    assert vdata["genuine"] is True
    assert vdata["report_id"] == report.id
    assert vdata["sha256"] == report.sha256

    # Test POST /api/v1/verify with report_id form field
    res_id = client.post(
        "/api/v1/verify",
        data={"report_id": report.id},
    )
    assert res_id.status_code == 200
    assert res_id.json()["genuine"] is True

    # Test POST /api/v1/verify with tampered PDF file
    tampered_bytes = pdf_data + b"\n%tampered_payload_annotation"
    tampered_file = io.BytesIO(tampered_bytes)
    res_tampered = client.post(
        "/api/v1/verify",
        files={"file": ("tampered_report.pdf", tampered_file, "application/pdf")},
    )
    assert res_tampered.status_code == 200
    assert res_tampered.json()["genuine"] is False

    # Test POST /api/v1/verify with nonexistent report ID
    res_nonexistent = client.post(
        "/api/v1/verify",
        data={"report_id": "rep_nonexistent_000"},
    )
    assert res_nonexistent.status_code == 200
    assert res_nonexistent.json()["genuine"] is False
