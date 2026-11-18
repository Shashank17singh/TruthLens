"""Authentication dependencies and helpers for JWT and API keys."""

from datetime import UTC, datetime
from typing import Annotated, NamedTuple

from backend.app.core.config import settings
from backend.app.core.errors import APIError
from backend.app.core.rate_limit import enforce_rate_limit
from backend.app.core.security import decode_token, hash_api_key
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_db
from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session


class AuthenticatedClient(NamedTuple):
    user: User
    api_key: ApiKey | None
    rate_limit_headers: dict[str, str]


def get_client_ip(request: Request) -> str:
    """Extract client IP address considering proxy headers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def rate_limit_unauthenticated(request: Request) -> dict[str, str]:
    """Dependency for endpoints allowing unauthenticated access (10 req/min/IP)."""
    ip = get_client_ip(request)
    return enforce_rate_limit(f"ip:{ip}", settings.RATE_LIMIT_UNAUTHENTICATED)


def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    db: Session = Depends(get_db),
) -> User:
    """Extract and authenticate User from Bearer JWT."""
    if not authorization or not authorization.startswith("Bearer "):
        raise APIError(
            status_code=401,
            code="invalid_token",
            message="Missing or invalid Authorization header.",
        )

    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_token(token)
    except Exception as exc:
        raise APIError(
            status_code=401,
            code="invalid_token",
            message=f"Invalid or expired access token: {exc}",
        ) from exc

    if payload.get("type") != "access":
        raise APIError(status_code=401, code="invalid_token", message="Invalid token type.")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise APIError(status_code=401, code="invalid_token", message="User account not found.")

    return user


def get_current_admin(user: User = Depends(get_current_user)) -> User:
    """Require admin role."""
    if user.role != "admin":
        raise APIError(
            status_code=403,
            code="admin_required",
            message="Administrator privileges required.",
        )
    return user


def get_authenticated_client(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
    db: Session = Depends(get_db),
) -> AuthenticatedClient:
    """Authenticate via either JWT Bearer token or X-API-Key and enforce rate limits."""
    now = datetime.now(UTC)

    # 1. Try API Key first
    if x_api_key:
        if not x_api_key.startswith("tl_"):
            raise APIError(status_code=401, code="invalid_api_key", message="Invalid API key format.")

        key_hash = hash_api_key(x_api_key)
        api_key_row = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()

        if not api_key_row or api_key_row.revoked_at is not None:
            raise APIError(status_code=401, code="invalid_api_key", message="API key is invalid or has been revoked.")

        if api_key_row.expires_at and api_key_row.expires_at < now:
            raise APIError(status_code=401, code="invalid_api_key", message="API key has expired.")

        # Update last used
        api_key_row.last_used_at = now
        db.commit()

        # Enforce rate limit
        limit = api_key_row.rate_limit_per_min or settings.RATE_LIMIT_API_KEY_DEFAULT
        headers = enforce_rate_limit(f"key:{api_key_row.id}", limit)

        return AuthenticatedClient(user=api_key_row.user, api_key=api_key_row, rate_limit_headers=headers)

    # 2. Try JWT Bearer
    if authorization and authorization.startswith("Bearer "):
        user = get_current_user(authorization=authorization, db=db)
        headers = enforce_rate_limit(f"user:{user.id}", settings.RATE_LIMIT_USER)
        return AuthenticatedClient(user=user, api_key=None, rate_limit_headers=headers)

    raise APIError(
        status_code=401,
        code="invalid_credentials",
        message="Authentication required: provide Bearer JWT or X-API-Key.",
    )
