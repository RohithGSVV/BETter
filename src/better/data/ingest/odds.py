"""The Odds API wrapper for fetching pre-game betting odds.

Source: https://the-odds-api.com/
Free tier: 500 requests/month (~16/day).
Strategy: 2 requests/day (morning snapshot + pre-game update).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd

from better.config import settings
from better.data.db import get_connection
from better.utils.logging import get_logger
from better.utils.stats import implied_probability_from_american, remove_vig

log = get_logger(__name__)

SPORT_KEY = "baseball_mlb"
MARKETS = "h2h"  # Moneyline (head-to-head)


class OddsClient:
    """Client for The Odds API v4."""

    def __init__(self, api_key: str | None = None, timeout: float = 30):
        self.api_key = api_key or settings.odds_api_key
        self.base_url = settings.odds_api_base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def get_odds(
        self,
        regions: str = "us",
        markets: str = MARKETS,
        odds_format: str = "american",
    ) -> list[dict]:
        """Fetch current pre-game odds for all upcoming MLB games.

        Returns a list of game dicts with odds from multiple bookmakers.
        """
        if not self.api_key:
            log.warning("odds_api_key_not_set")
            return []

        response = self.client.get(
            f"{self.base_url}/sports/{SPORT_KEY}/odds",
            params={
                "apiKey": self.api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
            },
        )
        response.raise_for_status()

        # Log remaining requests
        remaining = response.headers.get("x-requests-remaining", "?")
        log.info("odds_fetched", games=len(response.json()), remaining=remaining)

        return response.json()

    def close(self) -> None:
        self.client.close()


def parse_odds_response(games: list[dict]) -> pd.DataFrame:
    """Parse The Odds API response into a flat DataFrame of odds snapshots.

    Each row = one bookmaker's odds for one game.
    """
    rows = []
    now = datetime.now(timezone.utc)

    for game in games:
        game_id = game.get("id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence = game.get("commence_time", "")

        for bookmaker in game.get("bookmakers", []):
            bk_name = bookmaker.get("title", bookmaker.get("key", ""))

            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                home_odds = outcomes.get(home_team)
                away_odds = outcomes.get(away_team)

                if home_odds is not None and away_odds is not None:
                    home_implied = implied_probability_from_american(home_odds)
                    away_implied = implied_probability_from_american(away_odds)
                    home_fair, away_fair = remove_vig(home_implied, away_implied)
                    overround = home_implied + away_implied - 1.0

                    rows.append(
                        {
                            "game_pk": hash(game_id) % (2**31),
                            "captured_at": now,
                            "bookmaker": bk_name,
                            "market": "h2h",
                            "home_odds_american": home_odds,
                            "away_odds_american": away_odds,
                            "home_implied_prob": round(home_implied, 4),
                            "away_implied_prob": round(away_implied, 4),
                            "home_fair_prob": round(home_fair, 4),
                            "away_fair_prob": round(away_fair, 4),
                            "overround": round(overround, 4),
                        }
                    )

    return pd.DataFrame(rows)


def ingest_current_odds() -> int:
    """Fetch and store current MLB odds snapshot.

    Returns the number of odds rows stored.
    """
    client = OddsClient()
    try:
        games = client.get_odds()
        if not games:
            return 0

        df = parse_odds_response(games)
        if df.empty:
            return 0

        # Add auto-incrementing IDs
        conn = get_connection()
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM odds_snapshots").fetchone()[0]
        df.insert(0, "id", range(next_id, next_id + len(df)))

        conn.execute("INSERT INTO odds_snapshots SELECT * FROM df")
        log.info("odds_stored", rows=len(df))
        return len(df)
    finally:
        client.close()


def get_consensus_odds(game_pk: int) -> dict | None:
    """Get consensus (median) odds across bookmakers for a game.

    Returns dict with home_fair_prob, away_fair_prob, or None if not available.
    """
    conn = get_connection()
    result = conn.execute(
        """
        SELECT
            MEDIAN(home_fair_prob) as home_fair_prob,
            MEDIAN(away_fair_prob) as away_fair_prob,
            COUNT(*) as num_bookmakers
        FROM odds_snapshots
        WHERE game_pk = ?
        AND captured_at = (
            SELECT MAX(captured_at) FROM odds_snapshots WHERE game_pk = ?
        )
        """,
        [game_pk, game_pk],
    ).fetchone()

    if result and result[2] > 0:
        return {
            "home_fair_prob": result[0],
            "away_fair_prob": result[1],
            "num_bookmakers": result[2],
        }
    return None
