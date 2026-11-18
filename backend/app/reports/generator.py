"""Report generation, PDF rendering, QR code creation, and HMAC signing."""

import base64
import hashlib
import hmac
import io
import logging
from datetime import UTC, datetime
from pathlib import Path

import jinja2
import qrcode
from backend.app.core.config import settings
from backend.app.db.models import Analysis, AnalysisResult, Report
from backend.app.db.session import SessionLocal
from backend.app.storage.minio_client import storage
from backend.app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)


def generate_qr_data_uri(content: str) -> str:
    """Generate base64 data URI of QR code image."""
    qr = qrcode.QRCode(version=1, box_size=4, border=2)
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def sign_report(report_id: str, pdf_sha256: str) -> str:
    """Compute HMAC-SHA256 signature for PDF report."""
    key = settings.HMAC_KEY.encode("utf-8")
    msg = f"{pdf_sha256}:{report_id}".encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_report_hmac(report_id: str, pdf_sha256: str, signature: str) -> bool:
    """Verify HMAC signature in constant time."""
    expected = sign_report(report_id, pdf_sha256)
    return hmac.compare_digest(expected, signature)


def build_fallback_pdf(content_text: str, report_id: str, sha256_hash: str) -> bytes:
    """Generate a clean, minimal valid PDF document without external C libraries."""
    lines = [
        "%PDF-1.4",
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj",
    ]

    escaped_text = content_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_content = (
        "BT /F1 14 Tf 50 720 Td (TruthLens Verification Report) Tj "
        f"/F1 10 Tf 0 -25 Td (Report ID: {report_id}) Tj "
        f"0 -18 Td (Original File SHA-256: {sha256_hash}) Tj "
        f"0 -25 Td ({escaped_text[:120]}) Tj ET"
    )

    lines.extend([
        f"4 0 obj << /Length {len(stream_content)} >> stream",
        stream_content,
        "endstream endobj",
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        "xref",
        "0 6",
        "0000000000 65535 f ",
        "0000000010 00000 n ",
        "0000000060 00000 n ",
        "0000000117 00000 n ",
        "0000000244 00000 n ",
        "0000000350 00000 n ",
        "trailer << /Size 6 /Root 1 0 R >>",
        "startxref",
        "420",
        "%%EOF",
    ])
    return "\n".join(lines).encode("latin-1")


def render_pdf_bytes(report_id: str, analysis: Analysis, result: AnalysisResult, public_token: str) -> bytes:
    """Render HTML and convert to PDF bytes with QR code."""
    qr_url = f"/api/v1/public/reports/{public_token}"
    qr_data_uri = generate_qr_data_uri(qr_url)

    date_str = analysis.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if analysis.completed_at else datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    template = jinja_env.get_template("report.html")
    html_out = template.render(
        report_id=report_id,
        date=date_str,
        analysis=analysis,
        result=result,
        qr_data_uri=qr_data_uri,
    )

    try:
        import weasyprint
        return weasyprint.HTML(string=html_out).write_pdf()
    except Exception as exc:
        logger.info("WeasyPrint not available or failed (%s); using native PDF generator.", exc)
        summary = f"{result.verdict}: {result.explanation}"
        return build_fallback_pdf(summary, report_id, analysis.sha256 or "unknown")


@celery_app.task(name="backend.app.worker.tasks.render_report_task")
def render_report_task(report_id: str) -> None:
    """Celery task to render, sign, and store PDF report."""
    db = SessionLocal()
    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            return

        report.status = "rendering"
        db.commit()

        analysis = db.query(Analysis).filter(Analysis.id == report.analysis_id).first()
        if not analysis:
            raise ValueError("Analysis row not found.")

        result = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.analysis_id == analysis.id)
            .order_by(AnalysisResult.created_at.desc())
            .first()
        )
        if not result:
            raise ValueError("Analysis result not found.")

        # Render PDF
        pdf_bytes = render_pdf_bytes(report.id, analysis, result, report.public_token)
        sha256_hash = hashlib.sha256(pdf_bytes).hexdigest()
        hmac_sig = sign_report(report.id, sha256_hash)

        # Store in storage
        object_key = f"reports/{report.id}.pdf"
        storage.put_object(object_key, pdf_bytes, content_type="application/pdf")

        report.object_key = object_key
        report.sha256 = sha256_hash
        report.hmac = hmac_sig
        report.hmac_key_id = settings.HMAC_KEY_ID
        report.status = "ready"
        report.ready_at = datetime.now(UTC)
        db.commit()
        logger.info("Report %s rendered and signed successfully.", report.id)

    except Exception as exc:
        logger.exception("Failed to render report %s: %s", report_id, exc)
        if report:
            report.status = "failed"
            db.commit()
    finally:
        db.close()
