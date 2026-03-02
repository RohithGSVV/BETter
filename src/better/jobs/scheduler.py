"""Daily job scheduler for odds fetching and prediction refresh.

Uses APScheduler to run background tasks at configurable intervals.
Started automatically by the API server on startup.
"""

from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler

from better.utils.logging import get_logger

log = get_logger(__name__)


class DailyJobScheduler:
    """Schedules daily odds ingestion and prediction refresh."""

    def __init__(self) -> None:
        self.scheduler = BackgroundScheduler()
        self._running = False

    def start(self) -> None:
        """Start the scheduler with daily jobs."""
        if self._running:
            return

        # Fetch odds and refresh predictions daily at 10:00 AM Eastern
        self.scheduler.add_job(
            self._daily_refresh,
            "cron",
            hour=10,
            minute=0,
            timezone="US/Eastern",
            id="daily_odds_predictions",
            replace_existing=True,
        )

        # Second refresh at 5:00 PM Eastern (before evening games)
        self.scheduler.add_job(
            self._daily_refresh,
            "cron",
            hour=17,
            minute=0,
            timezone="US/Eastern",
            id="evening_odds_refresh",
            replace_existing=True,
        )

        self.scheduler.start()
        self._running = True
        log.info("scheduler_started", jobs=["10:00 AM ET", "5:00 PM ET"])

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            log.info("scheduler_stopped")

    @staticmethod
    def _daily_refresh() -> None:
        """Fetch current odds and refresh cached predictions."""
        try:
            from better.data.ingest.odds import ingest_current_odds

            rows = ingest_current_odds()
            log.info("daily_odds_ingested", rows=rows)
        except Exception as exc:
            log.error("daily_odds_failed", error=str(exc))

        try:
            from better.api.services import get_prediction_service

            svc = get_prediction_service()
            svc.refresh_predictions()
            log.info("daily_predictions_refreshed")
        except Exception as exc:
            log.error("daily_refresh_failed", error=str(exc))
