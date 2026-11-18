"""Admin operational endpoints."""

from backend.app.core.auth import get_current_admin
from backend.app.db.models import Analysis, AnalysisResult, Feedback, User
from backend.app.db.session import get_db
from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats")
def get_admin_stats(admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    total_analyses = db.query(Analysis).count()
    completed_analyses = db.query(Analysis).filter(Analysis.status == "completed").count()

    by_modality = dict(
        db.query(Analysis.modality, func.count(Analysis.id))
        .group_by(Analysis.modality)
        .all()
    )

    total_feedback = db.query(Feedback).count()
    correct_feedback = db.query(Feedback).filter(Feedback.is_correct == True).count()
    feedback_accuracy = round(correct_feedback / total_feedback, 4) if total_feedback > 0 else 1.0

    avg_latency = db.query(func.avg(AnalysisResult.latency_ms)).scalar() or 0

    return {
        "total_analyses": total_analyses,
        "completed_analyses": completed_analyses,
        "modality_breakdown": by_modality,
        "total_feedback": total_feedback,
        "feedback_accuracy": feedback_accuracy,
        "average_latency_ms": round(float(avg_latency), 2),
    }


@router.get("/feedback")
def list_feedback(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    flagged: bool | None = None,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    query = db.query(Feedback)
    if flagged is not None:
        query = query.filter(Feedback.is_correct == (not flagged))

    total = query.count()
    items = (
        query.order_by(desc(Feedback.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": [
            {
                "id": f.id,
                "analysis_id": f.analysis_id,
                "user_id": f.user_id,
                "is_correct": f.is_correct,
                "reported_label": f.reported_label,
                "comment": f.comment,
                "created_at": f.created_at.isoformat(),
            }
            for f in items
        ],
        "page": page,
        "page_size": page_size,
        "total": total,
    }
