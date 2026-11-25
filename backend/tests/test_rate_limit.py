"""Tests for rate limiting enforcement and headers."""

import uuid

import pytest

from backend.app.core.errors import APIError
from backend.app.core.rate_limit import check_rate_limit, enforce_rate_limit


def test_rate_limit_allows_under_limit():
    key = f"test_key_{uuid.uuid4().hex}"
    limit = 5

    for _ in range(limit):
        res = check_rate_limit(key, limit)
        assert res.allowed is True
        assert res.limit == limit

    # The next check must be denied
    res_denied = check_rate_limit(key, limit)
    assert res_denied.allowed is False
    assert res_denied.remaining == 0
    assert res_denied.reset_after_seconds > 0


def test_enforce_rate_limit_raises_429():
    key = f"test_blocked_{uuid.uuid4().hex}"
    limit = 2

    enforce_rate_limit(key, limit)
    enforce_rate_limit(key, limit)

    with pytest.raises(APIError) as exc_info:
        enforce_rate_limit(key, limit)

    assert exc_info.value.status_code == 429
    assert exc_info.value.code == "rate_limited"
    assert "Retry-After" in exc_info.value.headers


class _FlakyRedis:
    """Answers the first calls, then times out, as a loaded Redis does."""

    def __init__(self, working_calls: int):
        self.working_calls = working_calls
        self.store: list[float] = []

    def pipeline(self):
        return self

    def zremrangebyscore(self, *args):
        pass

    def zcard(self, *args):
        pass

    def expire(self, *args):
        pass

    def zadd(self, key, mapping):
        self._pending = list(mapping.values())

    def execute(self):
        if self.working_calls == 0:
            raise TimeoutError("redis timed out")
        self.working_calls -= 1
        count = len(self.store)
        self.store += self._pending
        return [0, count, 1, True]


def test_requests_counted_in_redis_still_count_when_redis_times_out(monkeypatch):
    from backend.app.core import rate_limit

    flaky = _FlakyRedis(working_calls=2)
    monkeypatch.setattr(rate_limit, "get_redis_client", lambda: flaky)
    key = f"test_flaky_{uuid.uuid4().hex}"

    enforce_rate_limit(key, 2)
    enforce_rate_limit(key, 2)
    with pytest.raises(APIError):
        enforce_rate_limit(key, 2)
