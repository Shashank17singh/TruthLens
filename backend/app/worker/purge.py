"""24-hour media purge background task."""

import logging
from datetime import UTC, datetime

from backend.app.db.models import Analysis, AnalysisResult, Report
from backend.app.db.session import SessionLocal
from backend.app.storage.minio_client import storage
from backend.app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="backend.app.worker.purge.purge_expired_media_task")
def purge_expired_media_task() -> int:
    """Purge media files older than 24 hours that were not saved by the user."""
    db = SessionLocal()
    purged_count = 0
    now = datetime.now(UTC)

    try:
        expired_analyses = (
            db.query(Analysis)
            .filter(
                Analysis.media_expires_at <= now,
                Analysis.retain_media == False,
                Analysis.media_deleted_at.is_(None),
            )
            .all()
        )

        for analysis in expired_analyses:
            logger.info("Purging expired media for analysis %s", analysis.id)

            # 1. Delete uploaded media
            if analysis.object_key:
                try:
                    storage.delete_object(analysis.object_key)
                except Exception as exc:
                    logger.warning("Failed to delete upload %s: %s", analysis.object_key, exc)
                analysis.object_key = None

            # 2. Delete artefacts and sanitize text details
            results = db.query(AnalysisResult).filter(AnalysisResult.analysis_id == analysis.id).all()
            for res in results:
                if res.artefacts and isinstance(res.artefacts, dict):
                    for _, art_key in res.artefacts.items():
                        try:
                            storage.delete_object(art_key)
                        except Exception as exc:
                            logger.warning("Failed to delete artefact %s: %s", art_key, exc)
                    res.artefacts = {}

                # Remove stored sentences from text details for privacy
                if res.details and isinstance(res.details, dict) and "sentences" in res.details:
                    sanitized_sentences = []
                    for s in res.details.get("sentences", []):
                        sanitized_sentences.append({
                            "start": s.get("start"),
                            "end": s.get("end"),
                            "probability": s.get("probability"),
                            "text": "[PURGED_AFTER_24H]",
                        })
                    details_copy = dict(res.details)
                    details_copy["sentences"] = sanitized_sentences
                    res.details = details_copy

            # 3. Handle report expiration
            report = db.query(Report).filter(Report.analysis_id == analysis.id).first()
            if report:
                if report.object_key:
                    try:
                        storage.delete_object(report.object_key)
                    except Exception as exc:
                        logger.warning("Failed to delete report PDF %s: %s", report.object_key, exc)
                    report.object_key = None
                if report.status == "ready":
                    report.status = "expired"

            # 4. Mark analysis as deleted
            analysis.media_deleted_at = now
            purged_count += 1

        db.commit()
        logger.info("Purge job finished. Total analyses cleaned: %d", purged_count)
        return purged_count

    except Exception as exc:
        logger.exception("Error executing purge job: %s", exc)
        db.rollback()
        return 0
    finally:
        db.close()
