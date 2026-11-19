"""Report generation, PDF download, public verdict view, and tamper verification."""

import hashlib
import secrets

from backend.app.core.auth import (
    AuthenticatedClient,
    get_authenticated_client,
    rate_limit_unauthenticated,
)
from backend.app.core.config import settings
from backend.app.core.errors import APIError
from backend.app.db.models import Analysis, AnalysisResult, Report
from backend.app.db.session import get_db
from backend.app.reports.generator import render_report_task, verify_report_hmac
from backend.app.storage.minio_client import storage
from fastapi import APIRouter, Depends, File, Form, Response, UploadFile, status
from fastapi.responses import Response as RawResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

router = APIRouter(tags=["reports"])


class ReportCreateRequest(BaseModel):
    analysis_id: str
    format: str = "pdf"


def _request_report_for_analysis(analysis_id: str, client: AuthenticatedClient, db: Session):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id, Analysis.user_id == client.user.id).first()
    if not analysis:
        raise APIError(status_code=404, code="not_found", message="Analysis not found.")

    if analysis.status != "completed":
        raise APIError(status_code=409, code="analysis_not_complete", message="Analysis is still processing.")

    if analysis.media_deleted_at is not None:
        raise APIError(status_code=410, code="media_expired", message="Media was purged after 24 hours.")

    report = db.query(Report).filter(Report.analysis_id == analysis_id).first()
    if not report:
        report = Report(
            analysis_id=analysis_id,
            status="queued",
            public_token=secrets.token_urlsafe(16),
        )
        db.add(report)
        db.commit()
        db.refresh(report)

    if report.status == "ready":
        return {
            "report_id": report.id,
            "status": "ready",
            "pdf_url": f"/api/v1/reports/{report.id}.pdf",
            "public_url": f"/api/v1/public/reports/{report.public_token}",
        }

    render_report_task.delay(report.id)

    return {
        "report_id": report.id,
        "status": "queued",
        "pdf_url": f"/api/v1/reports/{report.id}.pdf",
        "public_url": f"/api/v1/public/reports/{report.public_token}",
    }


@router.post("/reports", status_code=status.HTTP_202_ACCEPTED)
def create_report(
    req: ReportCreateRequest,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    """Create or retrieve a forensic report for an analysis."""
    return _request_report_for_analysis(req.analysis_id, client, db)


@router.post("/analyses/{id}/report", status_code=status.HTTP_202_ACCEPTED)
def request_report_by_analysis_id(
    id: str,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    """Legacy/convenience endpoint to request a report by analysis ID."""
    return _request_report_for_analysis(id, client, db)


@router.get("/reports/{id}.pdf")
def get_report_pdf(
    id: str,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    """Download the raw signed PDF report."""
    clean_id = id[:-4] if id.endswith(".pdf") else id
    report = db.query(Report).filter(Report.id == clean_id).first()
    if not report:
        raise APIError(status_code=404, code="not_found", message="Report not found.")

    analysis = db.query(Analysis).filter(Analysis.id == report.analysis_id, Analysis.user_id == client.user.id).first()
    if not analysis:
        raise APIError(status_code=404, code="not_found", message="Report not found.")

    if report.status in ("queued", "rendering"):
        headers = {"Retry-After": "5"}
        return Response(status_code=status.HTTP_202_ACCEPTED, headers=headers)

    if report.status == "expired":
        raise APIError(status_code=410, code="media_expired", message="Report media expired and was deleted.")

    if report.status != "ready" or not report.object_key:
        raise APIError(status_code=404, code="not_found", message="Report PDF not available.")

    try:
        pdf_bytes = storage.get_object(report.object_key)
        return RawResponse(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="truthlens_report_{clean_id}.pdf"'},
        )
    except Exception as exc:
        raise APIError(status_code=404, code="not_found", message="Report file not found.") from exc


@router.get("/reports/{id}")
def get_report_status(
    id: str,
    client: AuthenticatedClient = Depends(get_authenticated_client),
    db: Session = Depends(get_db),
):
    """Get metadata and rendering status of a report."""
    if id.endswith(".pdf"):
        return get_report_pdf(id=id, client=client, db=db)

    report = db.query(Report).filter(Report.id == id).first()
    if not report:
        raise APIError(status_code=404, code="not_found", message="Report not found.")

    analysis = db.query(Analysis).filter(Analysis.id == report.analysis_id, Analysis.user_id == client.user.id).first()
    if not analysis:
        raise APIError(status_code=404, code="not_found", message="Report not found.")

    return {
        "report_id": report.id,
        "status": report.status,
        "pdf_url": f"/api/v1/reports/{report.id}.pdf",
        "public_url": f"/api/v1/public/reports/{report.public_token}",
        "sha256": report.sha256,
        "ready_at": report.ready_at.isoformat() if report.ready_at else None,
    }


@router.get("/public/reports/{token}", dependencies=[Depends(rate_limit_unauthenticated)])
def get_public_report(token: str, db: Session = Depends(get_db)):
    """Public read-only report view accessible via QR code."""
    report = db.query(Report).filter(Report.public_token == token).first()
    if not report:
        raise APIError(status_code=404, code="not_found", message="Public report not found.")

    analysis = db.query(Analysis).filter(Analysis.id == report.analysis_id).first()
    if not analysis:
        raise APIError(status_code=404, code="not_found", message="Associated analysis not found.")

    result = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.analysis_id == analysis.id)
        .order_by(AnalysisResult.created_at.desc())
        .first()
    )
    if not result:
        raise APIError(status_code=404, code="not_found", message="Analysis result not found.")

    return {
        "report_id": report.id,
        "date": report.ready_at.isoformat() if report.ready_at else report.created_at.isoformat(),
        "modality": analysis.modality,
        "verdict": result.verdict,
        "probability": result.probability,
        "model_name": result.model_name,
        "model_version": result.model_version,
        "explanation": result.explanation,
        "file_hash": analysis.sha256,
        "disclaimer": "Results are probabilistic and must not be used alone to make decisions about a person.",
    }


@router.post("/verify", dependencies=[Depends(rate_limit_unauthenticated)])
async def verify_report(
    file: UploadFile | None = File(default=None),
    report_id: str | None = Form(default=None),
    sha256: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    """Verify cryptographic authenticity of a TruthLens report via PDF upload, report_id, or SHA-256."""
    target_report = None
    target_sha256 = None

    if file:
        content = await file.read()
        if len(content) > settings.MAX_REPORT_BYTES:
            raise APIError(
                status_code=413,
                code="file_too_large",
                message=f"Report file exceeds maximum allowed size of {settings.MAX_REPORT_BYTES // (1024 * 1024)} MB.",
            )

        if not content.startswith(b"%PDF"):
            raise APIError(status_code=415, code="unsupported_type", message="Uploaded file is not a valid PDF.")

        target_sha256 = hashlib.sha256(content).hexdigest()
        target_report = db.query(Report).filter(Report.sha256 == target_sha256).first()
    elif report_id:
        target_report = db.query(Report).filter(Report.id == report_id.strip()).first()
        if target_report:
            target_sha256 = target_report.sha256
    elif sha256:
        target_sha256 = sha256.strip().lower()
        target_report = db.query(Report).filter(Report.sha256 == target_sha256).first()
    else:
        raise APIError(
            status_code=400,
            code="invalid_request",
            message="Must provide either a PDF file, report_id, or sha256 checksum to verify.",
        )

    if not target_report or not target_report.hmac or not target_sha256:
        return {"genuine": False, "message": "Report signature or hash could not be verified in the TruthLens ledger."}

    # Verify HMAC signature in constant time
    is_valid = verify_report_hmac(target_report.id, target_sha256, target_report.hmac)
    if not is_valid:
        return {"genuine": False, "message": "Report tampering detected: cryptographic HMAC signature mismatch."}

    return {
        "genuine": True,
        "report_id": target_report.id,
        "issued_at": target_report.ready_at.isoformat() if target_report.ready_at else target_report.created_at.isoformat(),
        "sha256": target_sha256,
        "message": "Report successfully verified against the TruthLens cryptographic signing ledger.",
    }
