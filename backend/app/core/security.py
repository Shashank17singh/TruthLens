"""Security utilities: password hashing, JWT operations, API key generation, and UUIDv7."""

import hashlib
import secrets
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt
from backend.app.core.config import settings


def generate_uuidv7() -> str:
    """Generate RFC 9562 UUIDv7 in pure Python for time-ordered primary keys."""
    ms = int(time.time() * 1000)
    rand = secrets.randbits(62)
    high = (ms << 16) | 0x7000 | ((rand >> 48) & 0x0FFF)
    low = 0x8000_0000_0000_0000 | (rand & 0x3FFF_FFFF_FFFF_FFFF)
    return str(uuid.UUID(int=(high << 64) | low))


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    """Create JWT access token with 15-minute default expiry."""
    now = datetime.now(UTC)
    expire = now + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode: dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access",
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(subject: str, expires_delta: timedelta | None = None) -> str:
    """Create JWT refresh token with 7-day default expiry."""
    now = datetime.now(UTC)
    expire = now + (expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode: dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "refresh",
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT token."""
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])


def generate_api_key() -> tuple[str, str, str]:
    """Generate API key with prefix. Returns (full_key, prefix, sha256_hash).

    Format: tl_<32-byte-hex>
    """
    raw_token = secrets.token_hex(24)
    full_key = f"tl_{raw_token}"
    prefix = full_key[:8]
    key_hash = hash_api_key(full_key)
    return full_key, prefix, key_hash


def hash_api_key(key: str) -> str:
    """Compute SHA-256 hash of API key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
