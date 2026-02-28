"""Historical backtester for the betting engine.

Uses out-of-fold predictions (leakage-free) from walk-forward training
and Elo-based synthetic market probabilities to simulate historical betting.

Usage::

    from better.betting.backtest import Backtester
    result = Backtester().run()
    result.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from better.betting.bankroll import BankrollManager, BankrollStats
from better.betting.engine import BettingEngine
from better.config import settings
from better.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    """Results from a historical backtest run."""

    stats: BankrollStats
    bet_history: pd.DataFrame
    bankroll_curve: pd.DataFrame  # date, bankroll
    by_year: pd.DataFrame  # year, bets, wins, roi
    by_confidence: pd.DataFrame  # confidence, bets, wins, roi
    edge_threshold: float
    kelly_fraction: float
    model_used: str

    def print_summary(self) -> None:
        """Print a formatted summary to console."""
        s = self.stats
        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS — {self.model_used}")
        print(f"  Edge threshold: {self.edge_threshold:.1%}  |  Kelly: {self.kelly_fraction:.0%}")
        print(f"{'='*60}")
        print(f"  Initial bankroll:    ${s.initial_bankroll:,.2f}")
        print(f"  Final bankroll:      ${s.current_bankroll:,.2f}")
        print(f"  Total P&L:           ${s.total_pnl:+,.2f}")
        print(f"  ROI:                 {s.roi_pct:+.1f}%")
        print(f"  Yield:               {s.yield_pct:+.1f}%")
        print(f"  Total bets:          {s.total_bets}")
        print(f"  Win rate:            {s.win_rate:.1%}")
        print(f"  Avg edge:            {s.avg_edge:.1%}")
        print(f"  Max drawdown:        {s.max_drawdown_pct:.1f}%")
        print(f"  Longest losing run:  {s.longest_losing_streak}")
        print(f"  Sharpe ratio:        {s.sharpe_ratio:.2f}")
        print(f"{'='*60}")

        if not self.by_year.empty:
            print("\n  BY YEAR:")
            print(self.by_year.to_string(index=False))

        if not self.by_confidence.empty:
            print("\n  BY CONFIDENCE:")
            print(self.by_confidence.to_string(index=False))
        print()


class Backtester:
    """Simulates historical betting using OOF predictions + Elo market proxy.

    Parameters
    ----------
    initial_bankroll : float
        Starting bankroll. Default $1000.
    min_edge : float
        Minimum edge to place a bet. Default from config (0.03).
    kelly_fraction : float
        Fraction of full Kelly. Default from config (0.25).
    max_bet_pct : float
        Max bet as fraction of bankroll. Default from config (0.05).
    model : str
        Which model's predictions to use. Default "meta_learner".
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        min_edge: float | None = None,
        kelly_fraction: float | None = None,
        max_bet_pct: float | None = None,
        model: str = "meta_learner",
    ):
        self.initial_bankroll = initial_bankroll
        self.min_edge = min_edge if min_edge is not None else settings.min_edge_threshold
        self.kelly_fraction = (
            kelly_fraction if kelly_fraction is not None else settings.kelly_fraction
        )
        self.max_bet_pct = (
            max_bet_pct if max_bet_pct is not None else settings.max_bet_pct
        )
        self.model = model

    def run(
        self,
        oof_path: Path | None = None,
    ) -> BacktestResult:
        """Execute historical backtest.

        Loads OOF details CSV, iterates through games chronologically,
        and simulates betting decisions.
        """
        oof_path = oof_path or (settings.project_root / "results" / "oof_details.csv")
        oof = self._load_oof_data(oof_path)

        engine = BettingEngine(
            bankroll=self.initial_bankroll,
            min_edge=self.min_edge,
            kelly_fraction=self.kelly_fraction,
            max_bet_pct=self.max_bet_pct,
        )
        bankroll_mgr = BankrollManager(initial_bankroll=self.initial_bankroll)

        model_col = f"{self.model}_prob"
        if model_col not in oof.columns:
            raise ValueError(
                f"Model '{self.model}' not found in OOF data. "
                f"Available: {[c for c in oof.columns if c.endswith('_prob')]}"
            )

        # Process games chronologically
        bankroll_points = []
        for _, row in oof.iterrows():
            model_prob = row[model_col]
            market_prob = row["elo_home_win_prob"]
            home_win = bool(row["home_win"])

            if pd.isna(model_prob) or pd.isna(market_prob):
                continue

            # Update engine bankroll to current
            engine.bankroll = bankroll_mgr.current_bankroll

            rec = engine.evaluate_game_fair_prob(
                model_home_prob=model_prob,
                market_home_fair_prob=market_prob,
                home_team=str(row.get("home_team", "")),
                away_team=str(row.get("away_team", "")),
                game_date=row.get("game_date"),
                game_pk=row.get("game_pk"),
            )

            if rec is not None:
                # Determine outcome: did the bet side win?
                if rec.bet_side == "HOME":
                    outcome = home_win
                else:
                    outcome = not home_win

                bankroll_mgr.place_and_settle(rec, outcome)

                bankroll_points.append({
                    "game_date": row.get("game_date"),
                    "season": row.get("season"),
                    "bankroll": bankroll_mgr.current_bankroll,
                })

        # Build result
        stats = bankroll_mgr.get_stats()
        history_df = bankroll_mgr.to_dataframe()
        bankroll_curve = pd.DataFrame(bankroll_points) if bankroll_points else pd.DataFrame()

        by_year = self._stats_by_year(history_df)
        by_confidence = self._stats_by_confidence(history_df, oof)

        log.info(
            "backtest_complete",
            model=self.model,
            min_edge=self.min_edge,
            total_bets=stats.total_bets,
            roi_pct=stats.roi_pct,
            final_bankroll=stats.current_bankroll,
        )

        return BacktestResult(
            stats=stats,
            bet_history=history_df,
            bankroll_curve=bankroll_curve,
            by_year=by_year,
            by_confidence=by_confidence,
            edge_threshold=self.min_edge,
            kelly_fraction=self.kelly_fraction,
            model_used=self.model,
        )

    def sweep_edge_thresholds(
        self,
        thresholds: list[float] | None = None,
        oof_path: Path | None = None,
    ) -> pd.DataFrame:
        """Run backtest at multiple edge thresholds to find the optimum.

        Returns a DataFrame comparing ROI, bets, win rate at each threshold.
        """
        thresholds = thresholds or [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
        rows = []

        for threshold in thresholds:
            bt = Backtester(
                initial_bankroll=self.initial_bankroll,
                min_edge=threshold,
                kelly_fraction=self.kelly_fraction,
                max_bet_pct=self.max_bet_pct,
                model=self.model,
            )
            result = bt.run(oof_path)
            s = result.stats
            rows.append({
                "edge_threshold": threshold,
                "total_bets": s.total_bets,
                "win_rate": s.win_rate,
                "total_pnl": s.total_pnl,
                "roi_pct": s.roi_pct,
                "yield_pct": s.yield_pct,
                "max_drawdown_pct": s.max_drawdown_pct,
                "avg_edge": s.avg_edge,
                "final_bankroll": s.current_bankroll,
            })

        return pd.DataFrame(rows)

    def _load_oof_data(self, path: Path) -> pd.DataFrame:
        """Load and validate OOF details CSV."""
        if not path.exists():
            raise FileNotFoundError(
                f"OOF details not found at {path}. "
                "Run `better model train` first to generate predictions."
            )

        oof = pd.read_csv(path)
        required = ["home_win", "elo_home_win_prob"]
        missing = [c for c in required if c not in oof.columns]
        if missing:
            raise ValueError(f"OOF data missing required columns: {missing}")

        # Sort chronologically
        if "game_date" in oof.columns:
            oof = oof.sort_values("game_date").reset_index(drop=True)

        log.info("oof_data_loaded", rows=len(oof), columns=list(oof.columns))
        return oof

    @staticmethod
    def _stats_by_year(history: pd.DataFrame) -> pd.DataFrame:
        """Aggregate bet stats by year."""
        if history.empty or "game_date" not in history.columns:
            return pd.DataFrame()

        history = history.copy()
        history["year"] = pd.to_datetime(history["game_date"]).dt.year

        return (
            history.groupby("year")
            .agg(
                bets=("pnl", "count"),
                wins=("outcome", "sum"),
                total_pnl=("pnl", "sum"),
                avg_edge=("edge", "mean"),
                total_wagered=("bet_amount", "sum"),
            )
            .assign(
                win_rate=lambda d: (d["wins"] / d["bets"]).round(3),
                yield_pct=lambda d: (d["total_pnl"] / d["total_wagered"] * 100).round(1),
            )
            .reset_index()
        )

    @staticmethod
    def _stats_by_confidence(history: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
        """Aggregate bet stats by edge confidence bucket."""
        if history.empty:
            return pd.DataFrame()

        h = history.copy()
        h["confidence"] = pd.cut(
            h["edge"],
            bins=[0, 0.04, 0.07, 1.0],
            labels=["low", "medium", "high"],
        )

        return (
            h.groupby("confidence", observed=True)
            .agg(
                bets=("pnl", "count"),
                wins=("outcome", "sum"),
                total_pnl=("pnl", "sum"),
                avg_edge=("edge", "mean"),
            )
            .assign(win_rate=lambda d: (d["wins"] / d["bets"]).round(3))
            .reset_index()
        )
