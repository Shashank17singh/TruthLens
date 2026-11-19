"""API key management endpoints."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.core.auth import get_current_user
from backend.app.core.errors import APIError
from backend.app.core.security import generate_api_key
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_db

router = APIRouter(prefix="/api-keys", tags=["api_keys"])


class CreateApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    expires_at: datetime | None = None


@router.get("")
def list_api_keys(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    keys = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user.id, ApiKey.revoked_at.is_(None))
        .order_by(desc(ApiKey.created_at))
        .all()
    )
    return [
        {
            "id": k.id,
            "name": k.name,
            "key_prefix": k.key_prefix,
            "rate_limit_per_min": k.rate_limit_per_min,
            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            "expires_at": k.expires_at.isoformat() if k.expires_at else None,
            "created_at": k.created_at.isoformat(),
        }
        for k in keys
    ]


@router.post("", status_code=status.HTTP_201_CREATED)
def create_key(
    req: CreateApiKeyRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    full_key, prefix, key_hash = generate_api_key()

    api_key_row = ApiKey(
        user_id=user.id,
        name=req.name,
        key_prefix=prefix,
        key_hash=key_hash,
        rate_limit_per_min=30,
        expires_at=req.expires_at,
    )
    db.add(api_key_row)
    db.commit()
    db.refresh(api_key_row)

    return {
        "id": api_key_row.id,
        "name": api_key_row.name,
        "key": full_key,
        "key_prefix": prefix,
        "rate_limit_per_min": api_key_row.rate_limit_per_min,
        "expires_at": api_key_row.expires_at.isoformat() if api_key_row.expires_at else None,
        "created_at": api_key_row.created_at.isoformat(),
    }


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(
    id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    key = db.query(ApiKey).filter(ApiKey.id == id, ApiKey.user_id == user.id).first()
    if not key:
        raise APIError(status_code=404, code="not_found", message="API key not found.")

    key.revoked_at = datetime.now(UTC)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
