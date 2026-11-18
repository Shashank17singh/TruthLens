"""Model catalog endpoint."""

from backend.app.inference.registry import registry
from fastapi import APIRouter

router = APIRouter(tags=["models"])


@router.get("/models")
def get_models():
    """Return active model names, versions, and decision thresholds."""
    return registry.get_catalog()
