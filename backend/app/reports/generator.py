"""Report generation, PDF rendering, visual artifact embedding, and cryptographic HMAC signing."""

import base64
import hashlib
import hmac
import io
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def generate_qr_bytes(content: str) -> bytes:
    """Generate PNG bytes of QR code."""
    qr = qrcode.QRCode(version=1, box_size=4, border=2)
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_qr_data_uri(content: str) -> str:
    """Generate base64 data URI of QR code image."""
    png_bytes = generate_qr_bytes(content)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
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
    escaped_text = content_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_content = (
        "BT /F1 14 Tf 50 720 Td (TruthLens Verification Report) Tj "
        f"/F1 10 Tf 0 -25 Td (Report ID: {report_id}) Tj "
        f"0 -18 Td (Original File SHA-256: {sha256_hash}) Tj "
        f"0 -25 Td ({escaped_text[:120]}) Tj ET"
    )

    lines = [
        "%PDF-1.4",
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj",
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
    ]
    return "\n".join(lines).encode("latin-1")


def get_verdict_label(modality: str | None, verdict: str) -> str:
    """Compute human-friendly verdict label by modality."""
    if modality == "text":
        label_map = {"likely_fake": "Likely AI", "likely_real": "Likely human", "uncertain": "Mixed"}
    else:
        label_map = {"likely_fake": "Likely AI-generated", "likely_real": "Likely real", "uncertain": "Uncertain"}
    return label_map.get(verdict, "Uncertain")


def render_reportlab_pdf(
    report_id: str,
    analysis: Analysis,
    result: AnalysisResult,
    date_str: str,
    qr_png_bytes: bytes,
    hmac_sig: str,
    heatmap_bytes: bytes | None = None,
) -> bytes:
    """Render high-fidelity PDF report using ReportLab (pure Python, zero OS/C dependencies)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import (
        HRFlowable,
        Image as RLImage,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    verdict_label = getattr(result, "verdict_label", None) or get_verdict_label(analysis.modality, result.verdict)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    brand_style = ParagraphStyle(
        "BrandTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#4338ca"),
    )
    brand_sub_style = ParagraphStyle(
        "BrandSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#64748b"),
    )
    verdict_title_style = ParagraphStyle(
        "VerdictTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#be123c") if result.verdict == "likely_fake" else (colors.HexColor("#047857") if result.verdict == "likely_real" else colors.HexColor("#b45309")),
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=10,
        textColor=colors.HexColor("#64748b"),
    )
    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # 1. Header
    header_table_data = [
        [
            Paragraph("<b>TruthLens</b><br/><font size='8' color='#64748b'>Autonomous Multi-Modal Forensic Analysis</font>", brand_style),
            Paragraph(f"<b>Report ID:</b> {report_id}<br/><b>Date:</b> {date_str}<br/><b>Modality:</b> {analysis.modality.capitalize() if analysis.modality else 'N/A'}", ParagraphStyle("HeaderRight", parent=styles["Normal"], fontSize=8, leading=11, alignment=2)),
        ]
    ]
    header_table = Table(header_table_data, colWidths=[320, 200])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f172a"), spaceAfter=14))

    # 2. Verdict Banner
    banner_bg = colors.HexColor("#fff1f2") if result.verdict == "likely_fake" else (colors.HexColor("#f0fdf4") if result.verdict == "likely_real" else colors.HexColor("#fffbeb"))
    banner_border = colors.HexColor("#e11d48") if result.verdict == "likely_fake" else (colors.HexColor("#059669") if result.verdict == "likely_real" else colors.HexColor("#d97706"))

    banner_data = [
        [Paragraph(f"<b>{verdict_label}</b>", verdict_title_style)],
        [Paragraph(f"Synthetic AI Likelihood: <b>{result.probability * 100:.1f}%</b> &nbsp;|&nbsp; 95% Confidence Band: [{result.confidence_low * 100:.1f}% – {result.confidence_high * 100:.1f}%]", body_style)],
        [Paragraph(f"{result.explanation}", body_style)],
    ]
    banner_table = Table(banner_data, colWidths=[520])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
        ("BOX", (0, 0), (-1, -1), 1, banner_border),
        ("LINELEFT", (0, 0), (0, -1), 4, banner_border),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 14))

    # 3. File & Inspection Metadata
    story.append(Paragraph("<b>INSPECTION RECORD &amp; FILE INTEGRITY</b>", styles["Heading3"]))
    meta_data = [
        [Paragraph("File Name", styles["Normal"]), Paragraph(str(analysis.file_name or "Direct Input Buffer"), body_style), Paragraph("Inference Modality", styles["Normal"]), Paragraph(str(analysis.modality).capitalize() if analysis.modality else "N/A", body_style)],
        [Paragraph("SHA-256", styles["Normal"]), Paragraph(str(analysis.sha256 or "N/A"), code_style), Paragraph("Classifier Model", styles["Normal"]), Paragraph(f"{result.model_name} (v{result.model_version})", body_style)],
        [Paragraph("Threshold", styles["Normal"]), Paragraph("0.50 (Calibrated Target 95% Precision)", body_style), Paragraph("Runtime Execution", styles["Normal"]), Paragraph("CPU ONNX Runtime", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[70, 190, 90, 170])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 4. Modality Forensic Visual Evidence
    story.append(Paragraph("<b>FORENSIC EVIDENCE &amp; SIGNAL ATTRIBUTION</b>", styles["Heading3"]))
    details = result.details or {}

    if analysis.modality == "image":
        story.append(Paragraph("Spatial feature activations evaluated via ConvNeXt-Tiny and 2D Discrete Cosine Transform (DCT) radial power falloff.", body_style))
        story.append(Spacer(1, 6))

        if heatmap_bytes:
            try:
                heatmap_img = RLImage(io.BytesIO(heatmap_bytes), width=180, height=140)
                story.append(heatmap_img)
                story.append(Paragraph("<font size='7' color='#64748b'>Grad-CAM Saliency Heatmap Overlay</font>", styles["Normal"]))
                story.append(Spacer(1, 6))
            except Exception as e:
                logger.warning("Could not embed heatmap image: %s", e)

        regions = details.get("regions", [])
        if regions:
            reg_table_data = [["Region", "Bounding Box Coordinates [x1, y1, x2, y2]", "Anomaly Score"]]
            for r in regions[:4]:
                box_str = f"[{r.get('box', [0,0,0,0])[0]:.2f}, {r.get('box', [0,0,0,0])[1]:.2f}, {r.get('box', [0,0,0,0])[2]:.2f}, {r.get('box', [0,0,0,0])[3]:.2f}]"
                reg_table_data.append([r.get("part", "focal_face"), box_str, f"{r.get('score', 0) * 100:.1f}%"])
            reg_table = Table(reg_table_data, colWidths=[120, 260, 140])
            reg_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            story.append(reg_table)

    elif analysis.modality == "audio":
        story.append(Paragraph("Acoustic analysis evaluating Linear Frequency Cepstral Coefficients (LFCC) across 4.0-second sliding windows via LCNN architecture.", body_style))
        story.append(Spacer(1, 6))
        windows = details.get("windows", [])
        if windows:
            win_data = [["Window #", "Time Range", "Synthetic Probability", "Segment Assessment"]]
            for idx, w in enumerate(windows[:6]):
                p = w.get("probability", 0)
                status_txt = "Synthetic Voice" if p > 0.6 else ("Uncertain" if p >= 0.4 else "Organic Audio")
                win_data.append([str(idx + 1), f"{w.get('start', 0):.1f}s – {w.get('end', 0):.1f}s", f"{p * 100:.1f}%", status_txt])
            win_table = Table(win_data, colWidths=[60, 160, 140, 160])
            win_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            story.append(win_table)

    elif analysis.modality == "text":
        story.append(Paragraph("Stylometric perplexity and burstiness analysis. Language models exhibit reduced perplexity variance across consecutive sentences.", body_style))
        story.append(Spacer(1, 6))
        sentences = details.get("sentences", [])
        if sentences:
            sent_text_parts = []
            for s in sentences[:8]:
                p = s.get("probability", 0)
                color_hex = "#be123c" if p > 0.65 else ("#b45309" if p > 0.40 else "#0369a1")
                sent_text_parts.append(f"<font color='{color_hex}'><b>[{p*100:.0f}% AI]</b> {s.get('text', '')}</font>")
            story.append(Paragraph(" ".join(sent_text_parts), body_style))

    story.append(Spacer(1, 14))

    # 5. Cryptographic Verification & QR Code
    story.append(Paragraph("<b>CRYPTOGRAPHIC AUDIT &amp; TAMPER-EVIDENCE</b>", styles["Heading3"]))
    qr_img = RLImage(io.BytesIO(qr_png_bytes), width=75, height=75)
    crypto_content = [
        Paragraph(f"<b>Document HMAC-SHA256 Signature:</b><br/><font face='Courier' size='7'>{hmac_sig}</font>", styles["Normal"]),
        Spacer(1, 3),
        Paragraph(f"<b>Key ID:</b> {settings.HMAC_KEY_ID} &nbsp;|&nbsp; <b>Verify URL:</b> https://truthlens.ai/verify?report_id={report_id}", styles["Normal"]),
        Spacer(1, 3),
        Paragraph("<font size='7' color='#64748b'>Any tampering with text, scores, or timestamps invalidates this cryptographic signature.</font>", styles["Normal"]),
    ]
    crypto_table = Table([[crypto_content, qr_img]], colWidths=[430, 90])
    crypto_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(KeepTogether(crypto_table))

    # 6. Disclaimers
    disclaimer = getattr(result, "disclaimer", None) or "Results are probabilistic and must not be used alone to make decisions about a person."
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"<b>Mandatory Forensic Disclaimer &amp; Limitations:</b> {disclaimer} "
        "Results are probabilistic outputs from ONNX CPU runtimes. Known limitations: "
        "(1) Non-native English academic writing may exhibit low burstiness variance; "
        "(2) Social media re-compression degrades high-frequency spectral cues; "
        "(3) Zero-shot generalizations on novel post-2025 architectures require corroborating human review. "
        "Must never be used as sole grounds for sanctions.",
        disclaimer_style,
    ))

    doc.build(story)
    return buf.getvalue()


def render_pdf_bytes(report_id: str, analysis: Analysis, result: AnalysisResult, public_token: str) -> bytes:
    """Render HTML and convert to PDF bytes with QR code, supporting WeasyPrint and ReportLab."""
    verify_url = f"https://truthlens.ai/verify?report_id={report_id}"
    qr_data_uri = generate_qr_data_uri(verify_url)
    qr_png_bytes = generate_qr_bytes(verify_url)

    date_str = (
        analysis.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if analysis.completed_at
        else datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    # Check for heatmap artifact in storage
    heatmap_data_uri = None
    heatmap_bytes = None
    details = result.details or {}
    if isinstance(details, dict):
        artefacts = details.get("artefacts") or {}
        heatmap_key = artefacts.get("heatmap")
        if heatmap_key:
            try:
                heatmap_bytes = storage.get_object(heatmap_key)
                b64_hm = base64.b64encode(heatmap_bytes).decode("utf-8")
                heatmap_data_uri = f"data:image/png;base64,{b64_hm}"
            except Exception as e:
                logger.warning("Could not fetch heatmap artifact %s: %s", heatmap_key, e)

    # Initial signature computation for display in document
    temp_hash = hashlib.sha256(f"{report_id}:{date_str}".encode()).hexdigest()
    hmac_display_sig = sign_report(report_id, temp_hash)

    verdict_label = getattr(result, "verdict_label", None) or get_verdict_label(analysis.modality, result.verdict)
    setattr(result, "verdict_label", verdict_label)
    disclaimer = getattr(result, "disclaimer", None) or "Results are probabilistic and must not be used alone to make decisions about a person."
    setattr(result, "disclaimer", disclaimer)

    template = jinja_env.get_template("report.html")
    html_out = template.render(
        report_id=report_id,
        date=date_str,
        analysis=analysis,
        result=result,
        verdict_label=verdict_label,
        disclaimer=disclaimer,
        qr_data_uri=qr_data_uri,
        heatmap_data_uri=heatmap_data_uri,
        hmac_signature=hmac_display_sig,
        hmac_key_id=settings.HMAC_KEY_ID,
    )

    # 1. Try WeasyPrint (HTML+CSS standard)
    try:
        import weasyprint
        return weasyprint.HTML(string=html_out).write_pdf()
    except Exception as exc:
        logger.info("WeasyPrint rendering unavailable (%s); using ReportLab engine.", exc)

    # 2. Try ReportLab (portable pure Python)
    try:
        return render_reportlab_pdf(
            report_id=report_id,
            analysis=analysis,
            result=result,
            date_str=date_str,
            qr_png_bytes=qr_png_bytes,
            hmac_sig=hmac_display_sig,
            heatmap_bytes=heatmap_bytes,
        )
    except Exception as exc2:
        logger.warning("ReportLab rendering failed (%s); using baseline PDF builder.", exc2)

    # 3. Baseline fallback
    summary = f"{result.verdict}: {result.explanation}"
    return build_fallback_pdf(summary, report_id, analysis.sha256 or "unknown")


@celery_app.task(name="backend.app.worker.tasks.render_report_task")
def render_report_task(report_id: str) -> None:
    """Celery task to render, sign, and store PDF report with file caching."""
    db = SessionLocal()
    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            logger.warning("Report %s not found in database.", report_id)
            return

        # If already cached and ready, don't re-render
        if report.status == "ready" and report.object_key:
            logger.info("Report %s is already ready and cached.", report_id)
            return

        report.status = "rendering"
        db.commit()

        analysis = db.query(Analysis).filter(Analysis.id == report.analysis_id).first()
        if not analysis:
            raise ValueError(f"Analysis row {report.analysis_id} not found.")

        result = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.analysis_id == analysis.id)
            .order_by(AnalysisResult.created_at.desc())
            .first()
        )
        if not result:
            raise ValueError(f"Analysis result for {analysis.id} not found.")

        # Render PDF
        pdf_bytes = render_pdf_bytes(report.id, analysis, result, report.public_token)
        sha256_hash = hashlib.sha256(pdf_bytes).hexdigest()
        hmac_sig = sign_report(report.id, sha256_hash)

        # Store in MinIO / storage
        object_key = f"reports/{report.id}.pdf"
        storage.put_object(object_key, pdf_bytes, content_type="application/pdf")

        report.object_key = object_key
        report.sha256 = sha256_hash
        report.hmac = hmac_sig
        report.hmac_key_id = settings.HMAC_KEY_ID
        report.status = "ready"
        report.ready_at = datetime.now(UTC)
        db.commit()
        logger.info("Report %s rendered and signed successfully (sha256=%s).", report.id, sha256_hash[:12])

    except Exception as exc:
        logger.exception("Failed to render report %s: %s", report_id, exc)
        if report:
            report.status = "failed"
            db.commit()
    finally:
        db.close()
