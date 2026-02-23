"""Async odds feed poller.

Polls The Odds API at configurable intervals, respecting the free tier limit.
Free tier: 500 requests/month ≈ 16/day.
Strategy: 2 polls/day for pre-game odds (morning + pre-game).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from better.data.ingest.odds import OddsClient, parse_odds_response, ingest_current_odds
from better.utils.logging import get_logger

log = get_logger(__name__)


class OddsPoller:
    """Polls The Odds API at scheduled intervals."""

    def __init__(
        self,
        poll_interval_minutes: float = 720,  # 12 hours = 2x/day
        max_daily_requests: int = 16,
    ):
        self.poll_interval = poll_interval_minutes * 60  # Convert to seconds
        self.max_daily_requests = max_daily_requests
        self._running = False
        self._daily_count = 0
        self._last_reset: datetime | None = None

    async def start(self) -> None:
        """Start the odds polling loop."""
        self._running = True
        self._last_reset = datetime.now(timezone.utc)
        log.info("odds_poller_started", interval_min=self.poll_interval / 60)

        while self._running:
            # Reset daily counter at midnight UTC
            now = datetime.now(timezone.utc)
            if self._last_reset and now.date() > self._last_reset.date():
                self._daily_count = 0
                self._last_reset = now

            if self._daily_count >= self.max_daily_requests:
                log.warning("odds_daily_limit_reached", count=self._daily_count)
                await asyncio.sleep(3600)  # Wait an hour and check again
                continue

            try:
                rows = ingest_current_odds()
                self._daily_count += 1
                log.info(
                    "odds_poll_complete",
                    rows=rows,
                    daily_count=self._daily_count,
                )
            except Exception as e:
                log.error("odds_poll_error", error=str(e))

            await asyncio.sleep(self.poll_interval)

        log.info("odds_poller_stopped")

    def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
