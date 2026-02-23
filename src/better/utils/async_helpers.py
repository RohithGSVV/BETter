"""Async utilities for polling loops and rate limiting."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone


class RateLimiter:
    """Simple token-bucket rate limiter for API calls."""

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self._semaphore = asyncio.Semaphore(max_calls)
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        now = asyncio.get_event_loop().time()
        # Remove timestamps older than the period
        self._timestamps = [t for t in self._timestamps if now - t < self.period]
        if len(self._timestamps) >= self.max_calls:
            wait_time = self.period - (now - self._timestamps[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        self._timestamps.append(asyncio.get_event_loop().time())


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)
