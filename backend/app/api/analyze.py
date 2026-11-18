"""Analysis submission endpoints for image, audio, text, and URL."""

import hashlib
import io
from datetime import UTC, datetime, timedelta

import soundfile as sf
from backend.app.core.auth import AuthenticatedClient, get_authenticated_client
from backend.app.core.config import settings
from backend.app.core.errors import APIError
from backend.app.db.models import Analysis
from backend.app.db.session import get_db
from backend.app.storage.minio_client import storage
from backend.app.validation.antivirus import scan_bytes
from backend.app.validation.exif import strip_exif
from backend.app.validation.file_type import detect_audio_type, detect_image_type
from backend.app.worker.tasks import (
    analyze_audio_task,
    analyze_image_task,
    analyze_text_task,
    analyze_url_task,
)
from fastapi import APIRouter, Depends, File, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

router = APIRouter(prefix="/analyze", tags=["analyze"])


class SubmissionResponse(BaseModel):
    analysis_id: str
    status: str
    status_url: str


class TextAnalysisRequest(BaseModel):
    text: str


class UrlAnalysisRequest(BaseModel):
    url: str
    modality_hint: str | None = None


@router.post("/image", status_code=status.HTTP_202_ACCEPTED, response_model=SubmissionResponse)
async def analyze_image(
    file: UploadFile = File(...),
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    content = await file.read()
    if len(content) > settings.MAX_IMAGE_BYTES:
        raise APIError(
            status_code=413,
            code="file_too_large",
            message=f"Images must be {settings.MAX_IMAGE_BYTES // (1024 * 1024)} MB or smaller.",
            details={"max_bytes": settings.MAX_IMAGE_BYTES, "received_bytes": len(content)},
        )

    mime_type = detect_image_type(content)
    if not mime_type:
        raise APIError(
            status_code=415,
            code="unsupported_type",
            message="Unsupported image format. Allowed formats: JPEG, PNG, WebP.",
        )

    # Strip EXIF and verify image decodes
    try:
        clean_content = strip_exif(content, mime_type)
    except ValueError as exc:
        raise APIError(status_code=422, code="corrupt_media", message=str(exc)) from exc

    if not scan_bytes(clean_content):
        raise APIError(status_code=422, code="threat_detected", message="Security scanner rejected uploaded file.")

    sha256_hash = hashlib.sha256(content).hexdigest()
    now = datetime.now(UTC)
    media_expires = now + timedelta(hours=24)

    # Create Analysis record
    analysis = Analysis(
        user_id=client.user.id,
        api_key_id=client.api_key.id if client.api_key else None,
        modality="image",
        source_type="upload",
        status="queued",
        progress=0,
        file_name=file.filename or "uploaded_image",
        mime_type=mime_type,
        size_bytes=len(content),
        sha256=sha256_hash,
        retain_media=False,
        media_expires_at=media_expires,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # Store in storage
    ext = mime_type.split("/")[-1]
    object_key = f"uploads/{analysis.id}/original.{ext}"
    storage.put_object(object_key, clean_content, content_type=mime_type)

    analysis.object_key = object_key
    db.commit()

    # Enqueue task
    analyze_image_task.delay(analysis.id)

    return SubmissionResponse(
        analysis_id=analysis.id,
        status="queued",
        status_url=f"/api/v1/analyses/{analysis.id}",
    )


@router.post("/audio", status_code=status.HTTP_202_ACCEPTED, response_model=SubmissionResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    content = await file.read()
    if len(content) > settings.MAX_AUDIO_BYTES:
        raise APIError(
            status_code=413,
            code="file_too_large",
            message=f"Audio files must be {settings.MAX_AUDIO_BYTES // (1024 * 1024)} MB or smaller.",
            details={"max_bytes": settings.MAX_AUDIO_BYTES, "received_bytes": len(content)},
        )

    mime_type = detect_audio_type(content)
    if not mime_type:
        raise APIError(
            status_code=415,
            code="unsupported_type",
            message="Unsupported audio format. Allowed formats: WAV, MP3, FLAC, OGG, M4A, MP4.",
        )

    # Decode audio to check duration
    try:
        with io.BytesIO(content) as buf:
            with sf.SoundFile(buf) as sf_file:
                duration = len(sf_file) / sf_file.samplerate
                if duration > settings.MAX_AUDIO_DURATION_SECONDS:
                    raise APIError(
                        status_code=413,
                        code="audio_too_long",
                        message=f"Audio exceeds maximum duration of {int(settings.MAX_AUDIO_DURATION_SECONDS)} seconds.",
                        details={"max_duration_seconds": settings.MAX_AUDIO_DURATION_SECONDS, "audio_duration": duration},
                    )
    except APIError:
        raise
    except Exception as exc:
        raise APIError(status_code=422, code="corrupt_media", message=f"Corrupt or unreadable audio: {exc}") from exc

    if not scan_bytes(content):
        raise APIError(status_code=422, code="threat_detected", message="Security scanner rejected uploaded file.")

    sha256_hash = hashlib.sha256(content).hexdigest()
    now = datetime.now(UTC)
    media_expires = now + timedelta(hours=24)

    analysis = Analysis(
        user_id=client.user.id,
        api_key_id=client.api_key.id if client.api_key else None,
        modality="audio",
        source_type="upload",
        status="queued",
        progress=0,
        file_name=file.filename or "uploaded_audio",
        mime_type=mime_type,
        size_bytes=len(content),
        sha256=sha256_hash,
        retain_media=False,
        media_expires_at=media_expires,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    ext = mime_type.split("/")[-1]
    object_key = f"uploads/{analysis.id}/audio.{ext}"
    storage.put_object(object_key, content, content_type=mime_type)

    analysis.object_key = object_key
    db.commit()

    analyze_audio_task.delay(analysis.id)

    return SubmissionResponse(
        analysis_id=analysis.id,
        status="queued",
        status_url=f"/api/v1/analyses/{analysis.id}",
    )


@router.post("/text", status_code=status.HTTP_202_ACCEPTED, response_model=SubmissionResponse)
def analyze_text(
    req: TextAnalysisRequest,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    text_content = req.text.strip()
    if not text_content:
        raise APIError(status_code=422, code="validation_error", message="Text content cannot be empty.")

    if len(text_content) > settings.MAX_TEXT_CHARACTERS:
        raise APIError(
            status_code=413,
            code="text_too_long",
            message=f"Text exceeds limit of {settings.MAX_TEXT_CHARACTERS} characters.",
            details={"max_characters": settings.MAX_TEXT_CHARACTERS, "received": len(text_content)},
        )

    text_bytes = text_content.encode("utf-8")
    sha256_hash = hashlib.sha256(text_bytes).hexdigest()
    now = datetime.now(UTC)
    media_expires = now + timedelta(hours=24)

    analysis = Analysis(
        user_id=client.user.id,
        api_key_id=client.api_key.id if client.api_key else None,
        modality="text",
        source_type="upload",
        status="queued",
        progress=0,
        file_name="text_snippet.txt",
        mime_type="text/plain",
        size_bytes=len(text_bytes),
        sha256=sha256_hash,
        retain_media=False,
        media_expires_at=media_expires,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    object_key = f"uploads/{analysis.id}/text.txt"
    storage.put_object(object_key, text_bytes, content_type="text/plain; charset=utf-8")

    analysis.object_key = object_key
    db.commit()

    analyze_text_task.delay(analysis.id)

    return SubmissionResponse(
        analysis_id=analysis.id,
        status="queued",
        status_url=f"/api/v1/analyses/{analysis.id}",
    )


@router.post("/url", status_code=status.HTTP_202_ACCEPTED, response_model=SubmissionResponse)
def analyze_url(
    req: UrlAnalysisRequest,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    url_str = req.url.strip()
    if not url_str.startswith(("http://", "https://")):
        raise APIError(status_code=422, code="invalid_url", message="URL must start with http:// or https://")

    now = datetime.now(UTC)
    media_expires = now + timedelta(hours=24)

    analysis = Analysis(
        user_id=client.user.id,
        api_key_id=client.api_key.id if client.api_key else None,
        modality=req.modality_hint,
        source_type="url",
        source_url=url_str,
        status="queued",
        progress=0,
        retain_media=False,
        media_expires_at=media_expires,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    analyze_url_task.delay(analysis.id)

    return SubmissionResponse(
        analysis_id=analysis.id,
        status="queued",
        status_url=f"/api/v1/analyses/{analysis.id}",
    )
