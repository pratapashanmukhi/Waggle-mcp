import time
from collections import deque

import pytest

from waggle.rate_limit import RateLimiter


@pytest.mark.anyio
async def test_concurrency_key_removed_when_count_reaches_zero():
    limiter = RateLimiter(
        requests_per_minute=10,
        max_concurrent_requests=2,
    )

    async with limiter.concurrency_slot("user1"):
        assert "user1" in limiter._concurrent

    assert "user1" not in limiter._concurrent


@pytest.mark.anyio
async def test_expired_request_window_is_cleaned_up():
    limiter = RateLimiter(
        requests_per_minute=10,
        max_concurrent_requests=2,
    )

    old_time = time.monotonic() - 61
    limiter._request_windows["user1"] = deque([old_time])

    await limiter.check_rate("user1", is_write=False)

    assert "user1" in limiter._request_windows
    assert len(limiter._request_windows["user1"]) == 1
