"""Celery asynchronous tasks for media inference, URL download, and explanation."""

import hashlib
import json
import logging
from datetime import UTC, datetime

from backend.app.core.config import settings
from backend.app.db.models import Analysis, AnalysisResult
from backend.app.db.session import SessionLocal
from backend.app.inference.audio import run_audio_inference
from backend.app.inference.image import run_image_inference
from backend.app.inference.text import run_text_inference
from backend.app.storage.minio_client import storage
from backend.app.validation.antivirus import scan_bytes
from backend.app.validation.exif import strip_exif
from backend.app.validation.file_type import (
    detect_audio_type,
    detect_image_type,
)
from backend.app.validation.url_fetch import download_media_from_url
from backend.app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


def publish_event(analysis_id: str, payload: dict) -> None:
    """Publish event to Redis pub/sub channel for WebSocket broadcast."""
    try:
        from backend.app.core.rate_limit import get_redis_client
        r = get_redis_client()
        if r is None:
            return
        channel = f"analysis:{analysis_id}"
        r.publish(channel, json.dumps(payload))
    except Exception as exc:
        logger.debug("Redis pub/sub publish skipped: %s", exc)


@celery_app.task(name="backend.app.worker.tasks.analyze_image_task")
def analyze_image_task(analysis_id: str) -> None:
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            return

        analysis.status = "processing"
        analysis.progress = 20
        analysis.started_at = datetime.now(UTC)
        db.commit()
        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "validating", "progress": 20})

        # Load image bytes from storage
        if not analysis.object_key:
            raise ValueError("Media file not found in storage.")
        image_bytes = storage.get_object(analysis.object_key)

        # Antivirus scan
        if not scan_bytes(image_bytes):
            raise ValueError("File failed security check: potential threat detected.")

        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "inference", "progress": 50})

        # Run inference and explanation
        result_data, artefacts = run_image_inference(image_bytes)

        analysis.progress = 80
        db.commit()
        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "saving", "progress": 80})

        # Store artefacts in storage
        artefact_keys = {}
        for filename, data in artefacts.items():
            key = f"artefacts/{analysis_id}/{filename}"
            storage.put_object(key, data, content_type="image/png")
            artefact_keys[filename.split(".")[0]] = key

        result_data["details"]["artefacts"] = artefact_keys

        # Save AnalysisResult
        result_row = AnalysisResult(
            analysis_id=analysis_id,
            model_name=result_data["model_name"],
            model_version=result_data["model_version"],
            probability=result_data["probability"],
            decision_threshold=result_data["decision_threshold"],
            verdict=result_data["verdict"],
            confidence_low=result_data["confidence_band"]["low"],
            confidence_high=result_data["confidence_band"]["high"],
            explanation=result_data["explanation"],
            details=result_data["details"],
            artefacts=artefact_keys,
            latency_ms=result_data["latency_ms"],
        )
        db.add(result_row)

        analysis.status = "completed"
        analysis.progress = 100
        analysis.completed_at = datetime.now(UTC)
        db.commit()

        publish_event(analysis_id, {"type": "completed", "analysis_id": analysis_id})
    except Exception as exc:
        logger.exception("Error in analyze_image_task for %s: %s", analysis_id, exc)
        if analysis:
            analysis.status = "failed"
            analysis.error_code = "processing_error"
            analysis.error_message = str(exc)
            db.commit()
        publish_event(analysis_id, {"type": "failed", "analysis_id": analysis_id, "error": {"code": "processing_error", "message": str(exc)}})
    finally:
        db.close()


@celery_app.task(name="backend.app.worker.tasks.analyze_audio_task")
def analyze_audio_task(analysis_id: str) -> None:
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            return

        analysis.status = "processing"
        analysis.progress = 20
        analysis.started_at = datetime.now(UTC)
        db.commit()
        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "validating", "progress": 20})

        if not analysis.object_key:
            raise ValueError("Audio file not found in storage.")
        audio_bytes = storage.get_object(analysis.object_key)

        if not scan_bytes(audio_bytes):
            raise ValueError("File failed security check: potential threat detected.")

        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "inference", "progress": 50})

        result_data, artefacts = run_audio_inference(audio_bytes)

        analysis.progress = 80
        db.commit()
        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "saving", "progress": 80})

        # Store artefacts
        artefact_keys = {}
        for filename, data in artefacts.items():
            key = f"artefacts/{analysis_id}/{filename}"
            storage.put_object(key, data, content_type="image/png")
            artefact_keys[filename.split(".")[0]] = key

        result_data["details"]["artefacts"] = artefact_keys

        result_row = AnalysisResult(
            analysis_id=analysis_id,
            model_name=result_data["model_name"],
            model_version=result_data["model_version"],
            probability=result_data["probability"],
            decision_threshold=result_data["decision_threshold"],
            verdict=result_data["verdict"],
            confidence_low=result_data["confidence_band"]["low"],
            confidence_high=result_data["confidence_band"]["high"],
            explanation=result_data["explanation"],
            details=result_data["details"],
            artefacts=artefact_keys,
            latency_ms=result_data["latency_ms"],
        )
        db.add(result_row)

        analysis.status = "completed"
        analysis.progress = 100
        analysis.completed_at = datetime.now(UTC)
        db.commit()

        publish_event(analysis_id, {"type": "completed", "analysis_id": analysis_id})
    except Exception as exc:
        logger.exception("Error in analyze_audio_task for %s: %s", analysis_id, exc)
        if analysis:
            analysis.status = "failed"
            analysis.error_code = "processing_error"
            analysis.error_message = str(exc)
            db.commit()
        publish_event(analysis_id, {"type": "failed", "analysis_id": analysis_id, "error": {"code": "processing_error", "message": str(exc)}})
    finally:
        db.close()


@celery_app.task(name="backend.app.worker.tasks.analyze_text_task")
def analyze_text_task(analysis_id: str) -> None:
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            return

        analysis.status = "processing"
        analysis.progress = 20
        analysis.started_at = datetime.now(UTC)
        db.commit()
        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "validating", "progress": 20})

        if not analysis.object_key:
            raise ValueError("Text payload not found in storage.")
        text_bytes = storage.get_object(analysis.object_key)
        text_content = text_bytes.decode("utf-8")

        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "inference", "progress": 50})

        result_data, _ = run_text_inference(text_content)

        analysis.progress = 80
        db.commit()
        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "saving", "progress": 80})

        result_row = AnalysisResult(
            analysis_id=analysis_id,
            model_name=result_data["model_name"],
            model_version=result_data["model_version"],
            probability=result_data["probability"],
            decision_threshold=result_data["decision_threshold"],
            verdict=result_data["verdict"],
            confidence_low=result_data["confidence_band"]["low"],
            confidence_high=result_data["confidence_band"]["high"],
            explanation=result_data["explanation"],
            details=result_data["details"],
            artefacts={},
            latency_ms=result_data["latency_ms"],
        )
        db.add(result_row)

        analysis.status = "completed"
        analysis.progress = 100
        analysis.completed_at = datetime.now(UTC)
        db.commit()

        publish_event(analysis_id, {"type": "completed", "analysis_id": analysis_id})
    except Exception as exc:
        logger.exception("Error in analyze_text_task for %s: %s", analysis_id, exc)
        if analysis:
            analysis.status = "failed"
            analysis.error_code = "processing_error"
            analysis.error_message = str(exc)
            db.commit()
        publish_event(analysis_id, {"type": "failed", "analysis_id": analysis_id, "error": {"code": "processing_error", "message": str(exc)}})
    finally:
        db.close()


@celery_app.task(name="backend.app.worker.tasks.analyze_url_task")
def analyze_url_task(analysis_id: str) -> None:
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis or not analysis.source_url:
            return

        analysis.status = "processing"
        analysis.progress = 10
        analysis.started_at = datetime.now(UTC)
        db.commit()
        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "fetching", "progress": 10})

        # Download from URL with SSRF checks
        # Max bytes allows up to 25 MB (max of image/audio)
        max_bytes = max(settings.MAX_IMAGE_BYTES, settings.MAX_AUDIO_BYTES)
        data, final_url = download_media_from_url(analysis.source_url, max_bytes=max_bytes)

        analysis.progress = 25
        db.commit()
        publish_event(analysis_id, {"type": "progress", "analysis_id": analysis_id, "stage": "validating", "progress": 25})

        # Antivirus
        if not scan_bytes(data):
            raise ValueError("File failed security check: potential threat detected.")

        # Detect modality
        image_mime = detect_image_type(data)
        audio_mime = detect_audio_type(data)

        if image_mime:
            modality = "image"
            mime_type = image_mime
            if len(data) > settings.MAX_IMAGE_BYTES:
                raise ValueError(f"Image exceeds size limit of {settings.MAX_IMAGE_BYTES} bytes.")
            clean_bytes = strip_exif(data, mime_type)
        elif audio_mime:
            modality = "audio"
            mime_type = audio_mime
            if len(data) > settings.MAX_AUDIO_BYTES:
                raise ValueError(f"Audio exceeds size limit of {settings.MAX_AUDIO_BYTES} bytes.")
            clean_bytes = data
        else:
            raise ValueError("Downloaded file is not a supported image or audio format.")

        sha256_hash = hashlib.sha256(data).hexdigest()
        ext = mime_type.split("/")[-1]
        object_key = f"uploads/{analysis_id}/downloaded.{ext}"
        storage.put_object(object_key, clean_bytes, content_type=mime_type)

        analysis.modality = modality
        analysis.mime_type = mime_type
        analysis.size_bytes = len(data)
        analysis.sha256 = sha256_hash
        analysis.object_key = object_key
        db.commit()

        # Run corresponding inference
        if modality == "image":
            analyze_image_task(analysis_id)
        else:
            analyze_audio_task(analysis_id)

    except Exception as exc:
        logger.exception("Error in analyze_url_task for %s: %s", analysis_id, exc)
        if analysis:
            analysis.status = "failed"
            analysis.error_code = "url_download_failed"
            analysis.error_message = str(exc)
            db.commit()
        publish_event(analysis_id, {"type": "failed", "analysis_id": analysis_id, "error": {"code": "url_download_failed", "message": str(exc)}})
    finally:
        db.close()
