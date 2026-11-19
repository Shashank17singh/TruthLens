"""Endpoints for retrieving, updating, saving, deleting analyses and submitting feedback."""

import secrets
import time
from typing import Any

from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import Response as RawResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.core.auth import AuthenticatedClient, get_authenticated_client
from backend.app.core.errors import APIError
from backend.app.db.models import Analysis, AnalysisResult, Feedback
from backend.app.db.session import get_db
from backend.app.storage.minio_client import storage

router = APIRouter(prefix="/analyses", tags=["analyses"])

# Cache for 30-second WebSocket tickets: ticket -> (analysis_id, expires_at)
ws_tickets: dict[str, tuple[str, float]] = {}


class FeedbackRequest(BaseModel):
    is_correct: bool
    reported_label: str | None = Field(default=None, pattern="^(real|fake|unsure)$")
    comment: str | None = None


def format_analysis_response(analysis: Analysis, result: AnalysisResult | None = None) -> dict[str, Any]:
    """Format analysis object to match api.md schema."""
    result_dict = None
    if result:
        if analysis.modality == "text":
            label_map = {"likely_fake": "Likely AI", "likely_real": "Likely human", "uncertain": "Mixed"}
        else:
            label_map = {"likely_fake": "Likely AI-generated", "likely_real": "Likely real", "uncertain": "Uncertain"}
        verdict_label = label_map.get(result.verdict, "Uncertain")

        result_dict = {
            "probability": result.probability,
            "confidence_band": {
                "low": result.confidence_low,
                "high": result.confidence_high,
            },
            "verdict": result.verdict,
            "verdict_label": verdict_label,
            "explanation": result.explanation,
            "model_name": result.model_name,
            "model_version": result.model_version,
            "warnings": [],
            "disclaimer": "Results are probabilistic and must not be used alone to make decisions about a person.",
            "details": result.details,
        }

    error_dict = None
    if analysis.status == "failed" and analysis.error_code:
        error_dict = {
            "code": analysis.error_code,
            "message": analysis.error_message or "Analysis failed.",
        }

    return {
        "id": analysis.id,
        "modality": analysis.modality,
        "status": analysis.status,
        "progress": analysis.progress,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
        "media_expires_at": analysis.media_expires_at.isoformat() if analysis.media_expires_at else None,
        "retain_media": analysis.retain_media,
        "file_name": analysis.file_name,
        "result": result_dict,
        "error": error_dict,
    }


@router.get("/{id}")
def get_analysis(
    id: str,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    analysis = db.query(Analysis).filter(Analysis.id == id).first()
    if not analysis:
        raise APIError(status_code=404, code="not_found", message="Analysis not found.")

    latest_result = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.analysis_id == id)
        .order_by(desc(AnalysisResult.created_at))
        .first()
    )

    return format_analysis_response(analysis, latest_result)


@router.get("")
def list_analyses(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    modality: str | None = None,
    status: str | None = None,
    q: str | None = None,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    query = db.query(Analysis).filter(Analysis.user_id == client.user.id)

    if modality:
        query = query.filter(Analysis.modality == modality)
    if status:
        query = query.filter(Analysis.status == status)
    if q:
        query = query.filter(Analysis.file_name.ilike(f"%{q}%"))

    total = query.count()
    analyses = (
        query.order_by(desc(Analysis.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = []
    for a in analyses:
        res = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.analysis_id == a.id)
            .order_by(desc(AnalysisResult.created_at))
            .first()
        )
        items.append(format_analysis_response(a, res))

    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
    }


@router.post("/{id}/save")
def save_analysis(
    id: str,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    analysis = db.query(Analysis).filter(Analysis.id == id).first()
    if not analysis:
        raise APIError(status_code=404, code="not_found", message="Analysis not found.")

    if analysis.media_deleted_at is not None:
        raise APIError(
            status_code=410,
            code="media_expired",
            message="Media has already been purged under the 24-hour privacy policy.",
        )

    analysis.retain_media = True
    db.commit()

    latest_result = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.analysis_id == id)
        .order_by(desc(AnalysisResult.created_at))
        .first()
    )
    return format_analysis_response(analysis, latest_result)


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(
    id: str,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    analysis = db.query(Analysis).filter(Analysis.id == id).first()
    if not analysis:
        raise APIError(status_code=404, code="not_found", message="Analysis not found.")

    if client.user.role != "admin" and analysis.user_id != client.user.id:
        raise APIError(status_code=403, code="forbidden", message="You do not have permission to delete this analysis.")

    if analysis.object_key:
        try:
            storage.delete_object(analysis.object_key)
        except Exception:
            pass

    results = db.query(AnalysisResult).filter(AnalysisResult.analysis_id == id).all()
    for r in results:
        if r.artefacts and isinstance(r.artefacts, dict):
            for art_key in r.artefacts.values():
                try:
                    storage.delete_object(art_key)
                except Exception:
                    pass

    db.delete(analysis)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def guess_media_type(analysis: Analysis) -> str:
    """Infer suitable HTTP Content-Type for the original analysis media."""
    if analysis.mime_type and analysis.mime_type != "application/octet-stream":
        return analysis.mime_type
    if analysis.modality == "audio":
        return "audio/wav"
    if analysis.file_name:
        fn = analysis.file_name.lower()
        if fn.endswith(".jpg") or fn.endswith(".jpeg"):
            return "image/jpeg"
        if fn.endswith(".webp"):
            return "image/webp"
        if fn.endswith(".gif"):
            return "image/gif"
        if fn.endswith(".mp3"):
            return "audio/mpeg"
        if fn.endswith(".wav"):
            return "audio/wav"
        if fn.endswith(".ogg"):
            return "audio/ogg"
    return "image/png"


@router.get("/{id}/artefacts/{name}")
def get_artefact(
    id: str,
    name: str,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    analysis = db.query(Analysis).filter(Analysis.id == id).first()
    if not analysis:
        raise APIError(status_code=404, code="not_found", message="Analysis not found.")

    if analysis.media_deleted_at is not None:
        raise APIError(status_code=410, code="media_expired", message="Media was purged after 24 hours.")

    # 1. Check if requesting original uploaded input media
    clean_name = name.lower()
    base_name = clean_name.split(".")[0]
    if base_name in ("input", "original", "source"):
        if not analysis.object_key:
            raise APIError(status_code=404, code="not_found", message="Original media not found.")
        try:
            data = storage.get_object(analysis.object_key)
            media_type = guess_media_type(analysis)
            return RawResponse(content=data, media_type=media_type)
        except Exception as exc:
            raise APIError(status_code=404, code="not_found", message="Original media file not found.") from exc

    # 2. Check model explainability artefacts
    result = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.analysis_id == id)
        .order_by(desc(AnalysisResult.created_at))
        .first()
    )
    if not result or not result.artefacts:
        raise APIError(status_code=404, code="not_found", message="Artefact not found.")

    # Match name (e.g. 'heatmap' or 'heatmap.png')
    artefact_key = None
    for k, v in result.artefacts.items():
        k_base = k.lower().split(".")[0]
        if k == name or f"{k}.png" == name or k_base == base_name or v.endswith(name) or v.endswith(f"/{base_name}.png"):
            artefact_key = v
            break

    if not artefact_key:
        raise APIError(status_code=404, code="not_found", message="Artefact not found.")

    try:
        data = storage.get_object(artefact_key)
        return RawResponse(content=data, media_type="image/png")
    except Exception as exc:
        raise APIError(status_code=404, code="not_found", message="Artefact file not found.") from exc


@router.post("/{id}/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(
    id: str,
    req: FeedbackRequest,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    query = db.query(Analysis).filter(Analysis.id == id)
    if client.user.role != "admin":
        query = query.filter(Analysis.user_id == client.user.id)
    analysis = query.first()
    if not analysis:
        raise APIError(status_code=404, code="not_found", message="Analysis not found.")

    if analysis.status != "completed":
        raise APIError(
            status_code=409,
            code="analysis_not_complete",
            message="Feedback can only be submitted for completed analyses.",
        )

    existing = db.query(Feedback).filter(Feedback.analysis_id == id, Feedback.user_id == client.user.id).first()
    if existing:
        raise APIError(status_code=409, code="feedback_exists", message="Feedback has already been submitted.")

    feedback = Feedback(
        analysis_id=id,
        user_id=client.user.id,
        is_correct=req.is_correct,
        reported_label=req.reported_label,
        comment=req.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return {
        "id": feedback.id,
        "analysis_id": feedback.analysis_id,
        "is_correct": feedback.is_correct,
        "reported_label": feedback.reported_label,
        "comment": feedback.comment,
        "created_at": feedback.created_at.isoformat(),
    }


@router.post("/{id}/ws-ticket", status_code=status.HTTP_201_CREATED)
def create_websocket_ticket(
    id: str,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    query = db.query(Analysis).filter(Analysis.id == id)
    if client.user.role != "admin":
        query = query.filter(Analysis.user_id == client.user.id)
    analysis = query.first()
    if not analysis:
        raise APIError(status_code=404, code="not_found", message="Analysis not found.")

    ticket = secrets.token_urlsafe(24)
    ws_tickets[ticket] = (id, time.time() + 30.0)

    return {"ticket": ticket, "expires_in": 30}
