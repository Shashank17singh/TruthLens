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
