"""Date and season utility functions."""

from __future__ import annotations

from datetime import date, timedelta


def season_date_range(year: int) -> tuple[date, date]:
    """Return approximate (opening_day, last_day) for an MLB season.

    Regular season typically runs late March through early October.
    These are approximate; exact dates vary year to year.
    """
    # Approximate opening day and season end
    opening_day = date(year, 3, 28)
    season_end = date(year, 10, 1)
    return opening_day, season_end


def iter_date_range(start: date, end: date) -> list[date]:
    """Generate a list of dates from start to end (inclusive)."""
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def chunk_date_range(
    start: date, end: date, chunk_days: int = 5
) -> list[tuple[date, date]]:
    """Split a date range into chunks for API queries.

    Statcast queries are limited to ~25K rows, so we chunk into small windows.
    """
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def game_date_to_season(game_date: date) -> int:
    """Determine the MLB season year for a given game date.

    Games in March belong to that year's season.
    Postseason games in October/November belong to that year's season.
    """
    return game_date.year


def days_between(d1: date, d2: date) -> int:
    """Return the number of days between two dates (absolute value)."""
    return abs((d2 - d1).days)
