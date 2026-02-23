"""Async MLB live game feed poller.

Polls the MLB Stats API live feed every 10 seconds for active games,
detects state changes, and emits events for win probability updates.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import httpx

from better.data.ingest.mlb_api import MLBStatsClient, parse_game_state
from better.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class GameState:
    """Represents the current state of a live game."""

    game_pk: int
    inning: int = 1
    half: str = "top"  # "top" or "bot"
    outs: int = 0
    runners: int = 0  # 3-bit encoded: 1B=0b100, 2B=0b010, 3B=0b001
    home_score: int = 0
    away_score: int = 0
    current_pitcher_id: int | None = None
    current_batter_id: int | None = None
    balls: int = 0
    strikes: int = 0

    @property
    def score_diff(self) -> int:
        """Home score minus away score."""
        return self.home_score - self.away_score

    @property
    def state_key(self) -> tuple:
        """Hashable key for WE table lookup."""
        return (self.inning, self.half, self.outs, self.runners, self.score_diff)

    def has_changed(self, other: GameState) -> bool:
        """Check if the game state has meaningfully changed."""
        return self.state_key != other.state_key


@dataclass
class LiveGamePoller:
    """Polls live game feeds and emits state change events."""

    game_pk: int
    poll_interval: float = 10.0  # seconds
    on_state_change: Callable[[GameState], Any] | None = None
    _running: bool = False
    _last_state: GameState | None = None

    async def start(self) -> None:
        """Start polling the live feed."""
        self._running = True
        log.info("live_poller_started", game_pk=self.game_pk)

        async with httpx.AsyncClient(timeout=30) as client:
            while self._running:
                try:
                    url = f"https://statsapi.mlb.com/api/v1.1/game/{self.game_pk}/feed/live"
                    response = await client.get(url)
                    response.raise_for_status()
                    feed = response.json()

                    state_dict = parse_game_state(feed)
                    state = GameState(game_pk=self.game_pk, **state_dict)

                    # Check if game is over
                    game_status = (
                        feed.get("gameData", {})
                        .get("status", {})
                        .get("detailedState", "")
                    )
                    if game_status in ("Final", "Game Over", "Completed Early"):
                        log.info("game_finished", game_pk=self.game_pk, status=game_status)
                        if self.on_state_change:
                            self.on_state_change(state)
                        break

                    # Emit event if state changed
                    if self._last_state is None or state.has_changed(self._last_state):
                        if self.on_state_change:
                            self.on_state_change(state)
                        self._last_state = state

                except httpx.HTTPError as e:
                    log.warning("live_poll_error", game_pk=self.game_pk, error=str(e))

                await asyncio.sleep(self.poll_interval)

        log.info("live_poller_stopped", game_pk=self.game_pk)

    def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
