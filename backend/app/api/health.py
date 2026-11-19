"""Health and readiness check endpoints."""

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from backend.app.db.session import engine

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz():
    """Liveness probe: returns 200 when the process is alive."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz():
    """Readiness probe: verifies database connectivity."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as exc:
        return Response(
            content=f'{{"status": "not_ready", "error": "{exc!s}"}}',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json",
        )
