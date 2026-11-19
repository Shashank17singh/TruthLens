"""Model catalog endpoint."""

from fastapi import APIRouter

from backend.app.inference.registry import registry

router = APIRouter(tags=["models"])


@router.get("/models")
def get_models():
    """Return active model names, versions, and decision thresholds."""
    return registry.get_catalog()
