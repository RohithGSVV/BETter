"""Live game manager — orchestrates live feeds, predictions, and edge detection.

Ties together:
1. LiveGamePoller (MLB Stats API, every 3s for live state)
2. KalshiWebSocketFeed (push-based market prices, near-instant)
3. LiveWinProbModel (WE table lookup + pre-game blend, O(1))

On every state change (new inning, out, run, runner movement):
- Immediately computes updated P(home_win) from WE model
- Compares against latest Kalshi market price → live edge
- Stores snapshot for the API/dashboard to read

Latency budget:
    MLB poll → parse → WE lookup → edge calc = <50ms per cycle
    Kalshi WS push → store = <10ms per message
    API read → response = <5ms (in-memory)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from better.config import settings
from better.data.live.game_feed import GameState, LiveGamePoller
from better.models.ingame.win_expectancy import LiveWinProbModel
from better.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LiveGameSnapshot:
    """Point-in-time snapshot of a live game with predictions and market odds."""

    game_pk: int
    home_team: str
    away_team: str
    timestamp: datetime

    # Game state
    inning: int
    half: str
    outs: int
    runners: int
    home_score: int
    away_score: int

    # Model predictions
    win_prob: float             # blended in-game probability
    we_prob: float              # pure WE table probability
    pregame_prob: float | None  # pre-game model prior
    we_weight: float            # how much WE dominates (0-1)

    # Market
    market_prob: float | None = None  # Kalshi mid-price as probability
    edge: float | None = None         # win_prob - market_prob
    market_source: str = ""           # "kalshi" or "odds_api"

    # Status
    status: str = "live"        # "live", "final", "pre"

    def to_dict(self) -> dict:
        return {
            "game_pk": self.game_pk,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "timestamp": self.timestamp.isoformat(),
            "inning": self.inning,
            "half": self.half,
            "outs": self.outs,
            "runners": self.runners,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "win_prob": self.win_prob,
            "we_prob": self.we_prob,
            "pregame_prob": self.pregame_prob,
            "we_weight": self.we_weight,
            "market_prob": self.market_prob,
            "edge": self.edge,
            "market_source": self.market_source,
            "status": self.status,
        }


class LiveGameManager:
    """Manages all live game feeds and predictions.

    Provides a single entry point to:
    - Start/stop tracking all active games
    - Get the latest snapshot for any game (O(1), in-memory)
    - Get all active game snapshots
    """

    def __init__(self) -> None:
        self._we_model = LiveWinProbModel()
        self._pollers: dict[int, LiveGamePoller] = {}     # game_pk → poller
        self._snapshots: dict[int, LiveGameSnapshot] = {} # game_pk → latest
        self._game_info: dict[int, dict] = {}             # game_pk → schedule info
        self._pregame_probs: dict[int, float] = {}        # game_pk → pre-game P(home)
        self._market_probs: dict[int, float] = {}         # game_pk → latest market P(home)
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self.on_update: Callable[[LiveGameSnapshot], Any] | None = None

    def load_model(self) -> bool:
        """Load the WE table into memory. Must be called before start()."""
        return self._we_model.load()

    def get_snapshot(self, game_pk: int) -> LiveGameSnapshot | None:
        """Get the latest live snapshot for a game (O(1) read)."""
        return self._snapshots.get(game_pk)

    def get_all_snapshots(self) -> list[LiveGameSnapshot]:
        """Get all active game snapshots."""
        return list(self._snapshots.values())

    def update_market_prob(self, game_pk: int, prob: float, source: str = "kalshi") -> None:
        """Called by Kalshi feed when a new market price arrives.

        Immediately recalculates edge against our model prediction.
        """
        self._market_probs[game_pk] = prob

        # Update existing snapshot with new market data
        snap = self._snapshots.get(game_pk)
        if snap is not None:
            snap.market_prob = prob
            snap.market_source = source
            snap.edge = round(snap.win_prob - prob, 4)
            snap.timestamp = datetime.now(timezone.utc)

            if self.on_update:
                self.on_update(snap)

    async def start(
        self,
        games: list[dict],
        pregame_probs: dict[int, float] | None = None,
        poll_interval: float = 3.0,
    ) -> None:
        """Start tracking all active games.

        Args:
            games: Schedule entries [{game_pk, home_team, away_team, ...}]
            pregame_probs: {game_pk: pre-game P(home_win)} from PredictionService
            poll_interval: Seconds between MLB API polls (default 3s)
        """
        self._running = True

        if pregame_probs:
            self._pregame_probs = pregame_probs

        for game in games:
            gpk = game.get("game_pk")
            if not gpk:
                continue

            self._game_info[gpk] = game

            # Initialize with pre-game snapshot
            pregame = self._pregame_probs.get(gpk)
            self._snapshots[gpk] = LiveGameSnapshot(
                game_pk=gpk,
                home_team=game.get("home_team", ""),
                away_team=game.get("away_team", ""),
                timestamp=datetime.now(timezone.utc),
                inning=1, half="top", outs=0, runners=0,
                home_score=0, away_score=0,
                win_prob=pregame or 0.535,
                we_prob=0.535,
                pregame_prob=pregame,
                we_weight=0.0,
                status="pre",
            )

            # Create poller with aggressive interval
            poller = LiveGamePoller(
                game_pk=gpk,
                poll_interval=poll_interval,
                on_state_change=lambda state, gp=gpk: self._on_state_change(gp, state),
            )
            self._pollers[gpk] = poller
            task = asyncio.create_task(poller.start())
            self._tasks.append(task)

        log.info(
            "live_manager_started",
            games=len(self._pollers),
            poll_interval=poll_interval,
        )

    def _on_state_change(self, game_pk: int, state: GameState) -> None:
        """Handle a game state change — recalculate win prob immediately.

        This is the hot path. Must be FAST:
        - WE table lookup: O(1) dict access
        - Blend calculation: 2 multiplies + 1 add
        - Edge calculation: 1 subtract
        - Total: <1ms

        For finished games (state.status == "final"), bypass the WE model
        entirely and use the actual final score to determine win_prob.
        """
        game_info = self._game_info.get(game_pk, {})
        market = self._market_probs.get(game_pk)
        pregame = self._pregame_probs.get(game_pk)

        if state.status == "final":
            # Game is over — use actual score, not the WE model
            if state.home_score > state.away_score:
                win_prob = 1.0
            elif state.away_score > state.home_score:
                win_prob = 0.0
            else:
                win_prob = 0.5  # suspended/called with tie
            result = {
                "win_prob": win_prob,
                "we_prob": win_prob,
                "pregame_prob": round(pregame, 4) if pregame else None,
                "we_weight": 1.0,
                "inning": state.inning,
                "half": state.half,
                "outs": state.outs,
                "runners": state.runners,
                "home_score": state.home_score,
                "away_score": state.away_score,
            }
        else:
            result = self._we_model.predict(state, pregame_prob=pregame)

        snap = LiveGameSnapshot(
            game_pk=game_pk,
            home_team=game_info.get("home_team", ""),
            away_team=game_info.get("away_team", ""),
            timestamp=datetime.now(timezone.utc),
            inning=result["inning"],
            half=result["half"],
            outs=result["outs"],
            runners=result["runners"],
            home_score=result["home_score"],
            away_score=result["away_score"],
            win_prob=result["win_prob"],
            we_prob=result["we_prob"],
            pregame_prob=result["pregame_prob"],
            we_weight=result["we_weight"],
            market_prob=market,
            edge=round(result["win_prob"] - market, 4) if market else None,
            market_source="kalshi" if market else "",
            status=state.status,
        )

        self._snapshots[game_pk] = snap

        if self.on_update:
            self.on_update(snap)

        log.info(
            "live_prediction_updated",
            game_pk=game_pk,
            inning=f"{state.half[0].upper()}{state.inning}",
            score=f"{state.away_score}-{state.home_score}",
            win_prob=result["win_prob"],
            edge=snap.edge,
            status=state.status,
        )

    def stop(self) -> None:
        """Stop all pollers gracefully."""
        self._running = False
        for poller in self._pollers.values():
            poller.stop()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        log.info("live_manager_stopped")
