"""Bankroll management and bet tracking.

Tracks bankroll state, records bets, computes P&L, ROI, max drawdown,
and other performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from better.betting.engine import BetRecommendation
from better.utils.logging import get_logger
from better.utils.stats import american_to_decimal

log = get_logger(__name__)


@dataclass
class BetRecord:
    """A settled bet with outcome."""

    game_date: date | None
    home_team: str
    away_team: str
    bet_side: str
    model_prob: float
    market_prob: float
    edge: float
    odds_american: int
    bet_amount: float
    bankroll_before: float
    outcome: bool  # True = bet won
    pnl: float
    bankroll_after: float


@dataclass
class BankrollStats:
    """Aggregate performance metrics."""

    initial_bankroll: float
    current_bankroll: float
    total_bets: int
    bets_won: int
    bets_lost: int
    win_rate: float
    total_wagered: float
    total_pnl: float
    roi_pct: float  # total_pnl / initial_bankroll * 100
    yield_pct: float  # total_pnl / total_wagered * 100
    max_drawdown_pct: float
    avg_edge: float
    avg_odds: float
    best_bet_pnl: float
    worst_bet_pnl: float
    longest_losing_streak: int
    sharpe_ratio: float


class BankrollManager:
    """Manages bankroll state and bet history.

    Parameters
    ----------
    initial_bankroll : float
        Starting bankroll amount. Default $1000.
    """

    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self._peak_bankroll = initial_bankroll
        self._max_drawdown = 0.0
        self.history: list[BetRecord] = []

    def place_and_settle(
        self,
        rec: BetRecommendation,
        outcome: bool,
    ) -> BetRecord:
        """Place a bet and immediately settle it (for backtesting).

        Parameters
        ----------
        rec : BetRecommendation
            The bet recommendation from the engine.
        outcome : bool
            True if the bet won (bet side won the game).

        Returns
        -------
        BetRecord with P&L details.
        """
        # Scale bet amount to current bankroll (not the bankroll at time of recommendation)
        bet_fraction = rec.kelly_fraction
        if bet_fraction <= 0:
            bet_fraction = rec.bet_amount / max(self.current_bankroll, 1.0)
        bet_amount = min(
            self.current_bankroll * bet_fraction,
            self.current_bankroll * 0.05,  # hard cap
        )
        bet_amount = round(max(bet_amount, 0.0), 2)

        bankroll_before = self.current_bankroll

        if outcome:
            # Won: profit = bet_amount * (decimal_odds - 1)
            decimal_odds = american_to_decimal(rec.odds_american)
            pnl = round(bet_amount * (decimal_odds - 1), 2)
        else:
            # Lost: lose the bet amount
            pnl = -bet_amount

        self.current_bankroll = round(bankroll_before + pnl, 2)

        # Track peak and drawdown
        if self.current_bankroll > self._peak_bankroll:
            self._peak_bankroll = self.current_bankroll
        drawdown = (self._peak_bankroll - self.current_bankroll) / self._peak_bankroll
        self._max_drawdown = max(self._max_drawdown, drawdown)

        record = BetRecord(
            game_date=rec.game_date,
            home_team=rec.home_team,
            away_team=rec.away_team,
            bet_side=rec.bet_side,
            model_prob=rec.model_prob,
            market_prob=rec.market_prob,
            edge=rec.edge,
            odds_american=rec.odds_american,
            bet_amount=bet_amount,
            bankroll_before=bankroll_before,
            outcome=outcome,
            pnl=pnl,
            bankroll_after=self.current_bankroll,
        )
        self.history.append(record)
        return record

    def get_stats(self) -> BankrollStats:
        """Compute aggregate performance statistics."""
        if not self.history:
            return BankrollStats(
                initial_bankroll=self.initial_bankroll,
                current_bankroll=self.current_bankroll,
                total_bets=0,
                bets_won=0,
                bets_lost=0,
                win_rate=0.0,
                total_wagered=0.0,
                total_pnl=0.0,
                roi_pct=0.0,
                yield_pct=0.0,
                max_drawdown_pct=0.0,
                avg_edge=0.0,
                avg_odds=0.0,
                best_bet_pnl=0.0,
                worst_bet_pnl=0.0,
                longest_losing_streak=0,
                sharpe_ratio=0.0,
            )

        wins = [r for r in self.history if r.outcome]
        losses = [r for r in self.history if not r.outcome]
        pnls = [r.pnl for r in self.history]
        total_wagered = sum(r.bet_amount for r in self.history)
        total_pnl = sum(pnls)

        # Longest losing streak
        max_streak = 0
        current_streak = 0
        for r in self.history:
            if not r.outcome:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Sharpe ratio (annualized, assuming ~2400 games/year)
        pnl_array = np.array(pnls)
        if len(pnl_array) > 1 and pnl_array.std() > 0:
            sharpe = float(pnl_array.mean() / pnl_array.std() * np.sqrt(2400))
        else:
            sharpe = 0.0

        return BankrollStats(
            initial_bankroll=self.initial_bankroll,
            current_bankroll=self.current_bankroll,
            total_bets=len(self.history),
            bets_won=len(wins),
            bets_lost=len(losses),
            win_rate=round(len(wins) / len(self.history), 4) if self.history else 0.0,
            total_wagered=round(total_wagered, 2),
            total_pnl=round(total_pnl, 2),
            roi_pct=round(total_pnl / self.initial_bankroll * 100, 2),
            yield_pct=round(total_pnl / total_wagered * 100, 2) if total_wagered > 0 else 0.0,
            max_drawdown_pct=round(self._max_drawdown * 100, 2),
            avg_edge=round(np.mean([r.edge for r in self.history]), 4),
            avg_odds=round(np.mean([r.odds_american for r in self.history]), 1),
            best_bet_pnl=round(max(pnls), 2),
            worst_bet_pnl=round(min(pnls), 2),
            longest_losing_streak=max_streak,
            sharpe_ratio=round(sharpe, 2),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert bet history to a DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "game_date": r.game_date,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "bet_side": r.bet_side,
                "model_prob": r.model_prob,
                "market_prob": r.market_prob,
                "edge": r.edge,
                "odds_american": r.odds_american,
                "bet_amount": r.bet_amount,
                "bankroll_before": r.bankroll_before,
                "outcome": r.outcome,
                "pnl": r.pnl,
                "bankroll_after": r.bankroll_after,
            }
            for r in self.history
        ])
