"""Rate limiting module using Redis with in-memory fallback."""

import logging
import time
from collections import defaultdict
from typing import NamedTuple

import redis

from backend.app.core.config import settings
from backend.app.core.errors import APIError

logger = logging.getLogger(__name__)

# Fallback in-memory rate tracker: key -> list of timestamps
_in_memory_buckets: dict[str, list[float]] = defaultdict(list)
_redis_available: bool | None = None
_last_redis_check: float = 0.0


class RateLimitResult(NamedTuple):
    allowed: bool
    limit: int
    remaining: int
    reset_after_seconds: int


def get_redis_client() -> redis.Redis | None:
    """Return active Redis client or None if Redis is not running."""
    global _redis_available, _last_redis_check
    now = time.time()
    if _redis_available is False and (now - _last_redis_check) < 10.0:
        return None
    try:
        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=0.1, socket_timeout=0.1)
        r.ping()
        _redis_available = True
        return r
    except Exception:
        _redis_available = False
        _last_redis_check = now
        return None


def check_rate_limit(key: str, limit_per_minute: int) -> RateLimitResult:
    """Check rate limit using sliding window.

    Tries Redis first; falls back to in-memory window if Redis is unavailable.
    """
    now = time.time()
    window_start = now - 60.0
    # Every allowed request is also kept in memory, so a Redis timeout falls back to the full count.
    _in_memory_buckets[key] = [t for t in _in_memory_buckets[key] if t > window_start]
    bucket = _in_memory_buckets[key]

    r = get_redis_client()
    if r is not None:
        try:
            pipe = r.pipeline()
            redis_key = f"rl:{key}"
            pipe.zremrangebyscore(redis_key, 0, window_start)
            pipe.zcard(redis_key)
            pipe.zadd(redis_key, {str(now): now})
            pipe.expire(redis_key, 65)
            results = pipe.execute()
            current_count = results[1]

            if current_count >= limit_per_minute:
                # Query oldest item score in window
                oldest_res = r.zrange(redis_key, 0, 0, withscores=True)
                oldest_ts = oldest_res[0][1] if oldest_res else window_start
                reset_after = max(1, int(oldest_ts + 60.0 - now))
                return RateLimitResult(
                    allowed=False,
                    limit=limit_per_minute,
                    remaining=0,
                    reset_after_seconds=reset_after,
                )

            bucket.append(now)
            remaining = max(0, limit_per_minute - current_count - 1)
            return RateLimitResult(
                allowed=True,
                limit=limit_per_minute,
                remaining=remaining,
                reset_after_seconds=60,
            )
        except Exception as exc:
            logger.debug("Redis rate limiter error (%s); falling back to in-memory.", exc)

    # In-memory fallback
    if len(bucket) >= limit_per_minute:
        oldest = bucket[0]
        reset_after = max(1, int(oldest + 60.0 - now))
        return RateLimitResult(
            allowed=False,
            limit=limit_per_minute,
            remaining=0,
            reset_after_seconds=reset_after,
        )

    bucket.append(now)
    remaining = max(0, limit_per_minute - len(bucket))
    return RateLimitResult(
        allowed=True,
        limit=limit_per_minute,
        remaining=remaining,
        reset_after_seconds=60,
    )


def enforce_rate_limit(key: str, limit_per_minute: int) -> dict[str, str]:
    """Check rate limit and return headers, or raise APIError(429)."""
    result = check_rate_limit(key, limit_per_minute)
    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        "Retry-After": str(result.reset_after_seconds),
    }

    if not result.allowed:
        raise APIError(
            status_code=429,
            code="rate_limited",
            message=f"Rate limit exceeded. Try again in {result.reset_after_seconds} seconds.",
            details={"retry_after": result.reset_after_seconds},
            headers=headers,
        )

    return headers
