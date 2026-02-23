"""MLB Stats API wrapper for real-time data.

Fetches schedules, probable pitchers, lineups, rosters, and live game feeds.
Base URL: https://statsapi.mlb.com/api/v1/ (no API key required)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import httpx

from better.config import settings
from better.utils.logging import get_logger

log = get_logger(__name__)


class MLBStatsClient:
    """Client for the MLB Stats API."""

    def __init__(self, base_url: str | None = None, timeout: float = 30):
        self.base_url = (base_url or settings.mlb_api_base_url).rstrip("/")
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    def _get(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        """Make a GET request to the MLB Stats API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_schedule(self, game_date: date) -> list[dict]:
        """Get all games scheduled for a given date.

        Returns a list of game dicts with game_pk, teams, status, etc.
        """
        data = self._get(
            "/schedule",
            params={
                "date": game_date.isoformat(),
                "sportId": 1,  # MLB
                "hydrate": "probablePitcher,team,linescore",
            },
        )
        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                games.append(_parse_schedule_game(game))
        return games

    def get_game_feed(self, game_pk: int) -> dict:
        """Get live game feed for a specific game.

        Returns detailed play-by-play, linescore, and boxscore data.
        """
        # Live feed uses v1.1 endpoint
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        response = self.client.get(url)
        response.raise_for_status()
        return response.json()

    def get_game_boxscore(self, game_pk: int) -> dict:
        """Get boxscore for a completed game."""
        return self._get(f"/game/{game_pk}/boxscore")

    def get_roster(self, team_id: int, season: int | None = None) -> list[dict]:
        """Get active roster for a team."""
        params = {"rosterType": "active"}
        if season:
            params["season"] = season
        data = self._get(f"/teams/{team_id}/roster", params=params)
        return [
            {
                "player_id": p["person"]["id"],
                "name": p["person"]["fullName"],
                "position": p["position"]["abbreviation"],
                "jersey": p.get("jerseyNumber", ""),
                "status": p.get("status", {}).get("description", "Active"),
            }
            for p in data.get("roster", [])
        ]

    def get_probable_pitchers(self, game_date: date) -> list[dict]:
        """Get probable starting pitchers for all games on a date."""
        games = self.get_schedule(game_date)
        return [
            {
                "game_pk": g["game_pk"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_sp_id": g.get("home_sp_id"),
                "home_sp_name": g.get("home_sp_name"),
                "away_sp_id": g.get("away_sp_id"),
                "away_sp_name": g.get("away_sp_name"),
            }
            for g in games
        ]

    def get_lineup(self, game_pk: int) -> dict[str, list[dict]] | None:
        """Get confirmed lineups for a game (available ~4h before first pitch).

        Returns {"home": [...], "away": [...]} or None if not yet available.
        """
        try:
            feed = self.get_game_feed(game_pk)
            lineups = {"home": [], "away": []}

            game_data = feed.get("gameData", {})
            live_data = feed.get("liveData", {})
            boxscore = live_data.get("boxscore", {})

            for side in ["home", "away"]:
                team_data = boxscore.get("teams", {}).get(side, {})
                batting_order = team_data.get("battingOrder", [])
                players = team_data.get("players", {})

                for player_id in batting_order:
                    key = f"ID{player_id}"
                    player = players.get(key, {})
                    lineups[side].append(
                        {
                            "player_id": player_id,
                            "name": player.get("person", {}).get("fullName", ""),
                            "position": player.get("position", {}).get(
                                "abbreviation", ""
                            ),
                            "bat_side": player.get("batSide", {}).get("code", ""),
                        }
                    )

            if not lineups["home"] and not lineups["away"]:
                return None
            return lineups
        except Exception:
            return None

    def close(self) -> None:
        self.client.close()


def _parse_schedule_game(game: dict) -> dict:
    """Parse a game entry from the schedule API response."""
    home = game.get("teams", {}).get("home", {})
    away = game.get("teams", {}).get("away", {})

    home_sp = home.get("probablePitcher", {})
    away_sp = away.get("probablePitcher", {})

    return {
        "game_pk": game["gamePk"],
        "game_date": game.get("officialDate", game.get("gameDate", "")[:10]),
        "status": game.get("status", {}).get("detailedState", ""),
        "home_team": home.get("team", {}).get("abbreviation", ""),
        "away_team": away.get("team", {}).get("abbreviation", ""),
        "home_score": home.get("score"),
        "away_score": away.get("score"),
        "home_sp_id": home_sp.get("id"),
        "home_sp_name": home_sp.get("fullName", ""),
        "away_sp_id": away_sp.get("id"),
        "away_sp_name": away_sp.get("fullName", ""),
        "venue": game.get("venue", {}).get("name", ""),
        "game_type": game.get("gameType", "R"),
    }


def parse_game_state(feed: dict) -> dict:
    """Parse live feed into a game state dict for win probability updates.

    Returns dict with: inning, half, outs, runners, home_score, away_score,
    current_pitcher_id, current_batter_id, pitch_count, etc.
    """
    live_data = feed.get("liveData", {})
    linescore = live_data.get("linescore", {})
    plays = live_data.get("plays", {})
    current_play = plays.get("currentPlay", {})
    matchup = current_play.get("matchup", {})

    runners_on = 0
    offense = linescore.get("offense", {})
    if offense.get("first"):
        runners_on |= 0b100
    if offense.get("second"):
        runners_on |= 0b010
    if offense.get("third"):
        runners_on |= 0b001

    return {
        "inning": linescore.get("currentInning", 1),
        "half": "top" if linescore.get("inningHalf", "Top") == "Top" else "bot",
        "outs": linescore.get("outs", 0),
        "runners": runners_on,
        "home_score": linescore.get("teams", {}).get("home", {}).get("runs", 0),
        "away_score": linescore.get("teams", {}).get("away", {}).get("runs", 0),
        "current_pitcher_id": matchup.get("pitcher", {}).get("id"),
        "current_batter_id": matchup.get("batter", {}).get("id"),
        "balls": current_play.get("count", {}).get("balls", 0),
        "strikes": current_play.get("count", {}).get("strikes", 0),
    }
