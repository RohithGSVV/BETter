"""Tests for BankrollManager."""

from datetime import date

import pytest

from better.betting.bankroll import BankrollManager, BetRecord
from better.betting.engine import BetRecommendation


def _make_rec(
    model_prob: float = 0.60,
    market_prob: float = 0.50,
    odds_american: int = -110,
    bet_side: str = "HOME",
    kelly_fraction: float = 0.02,
    bet_amount: float = 20.0,
) -> BetRecommendation:
    """Helper to create a BetRecommendation for testing."""
    return BetRecommendation(
        game_date=date(2024, 7, 15),
        home_team="NYY",
        away_team="BOS",
        bet_side=bet_side,
        model_prob=model_prob,
        market_prob=market_prob,
        edge=round(model_prob - market_prob, 4),
        expected_value=0.05,
        kelly_fraction=kelly_fraction,
        bet_amount=bet_amount,
        odds_american=odds_american,
        bookmaker="test",
        confidence="medium",
    )


class TestBankrollManager:
    """Tests for bankroll tracking and P&L computation."""

    def test_initial_state(self):
        mgr = BankrollManager(initial_bankroll=1000.0)
        assert mgr.current_bankroll == 1000.0
        assert len(mgr.history) == 0

    def test_winning_bet_increases_bankroll(self):
        mgr = BankrollManager(initial_bankroll=1000.0)
        rec = _make_rec(odds_american=-110, kelly_fraction=0.02)
        record = mgr.place_and_settle(rec, outcome=True)

        assert record.outcome is True
        assert record.pnl > 0
        assert mgr.current_bankroll > 1000.0

    def test_losing_bet_decreases_bankroll(self):
        mgr = BankrollManager(initial_bankroll=1000.0)
        rec = _make_rec(kelly_fraction=0.02)
        record = mgr.place_and_settle(rec, outcome=False)

        assert record.outcome is False
        assert record.pnl < 0
        assert mgr.current_bankroll < 1000.0

    def test_pnl_calculation_at_minus_110(self):
        """At -110 odds, winning $20 bet returns ~$18.18 profit."""
        mgr = BankrollManager(initial_bankroll=1000.0)
        rec = _make_rec(odds_american=-110, kelly_fraction=0.02)
        record = mgr.place_and_settle(rec, outcome=True)

        expected_profit = round(20.0 * (100 / 110), 2)
        assert abs(record.pnl - expected_profit) < 0.1

    def test_pnl_calculation_at_plus_150(self):
        """At +150 odds, winning $20 bet returns $30 profit."""
        mgr = BankrollManager(initial_bankroll=1000.0)
        rec = _make_rec(odds_american=150, kelly_fraction=0.02)
        record = mgr.place_and_settle(rec, outcome=True)

        expected_profit = round(20.0 * 1.5, 2)
        assert abs(record.pnl - expected_profit) < 0.1

    def test_sequential_bets_track_correctly(self):
        """Multiple bets update bankroll sequentially."""
        mgr = BankrollManager(initial_bankroll=1000.0)
        rec = _make_rec(kelly_fraction=0.02)

        # Win, lose, win
        mgr.place_and_settle(rec, outcome=True)
        mgr.place_and_settle(rec, outcome=False)
        mgr.place_and_settle(rec, outcome=True)

        assert len(mgr.history) == 3
        assert mgr.history[0].bankroll_before == 1000.0
        assert mgr.history[1].bankroll_before == mgr.history[0].bankroll_after
        assert mgr.history[2].bankroll_before == mgr.history[1].bankroll_after

    def test_max_drawdown_tracked(self):
        """Max drawdown is recorded when bankroll drops from peak."""
        mgr = BankrollManager(initial_bankroll=1000.0)
        rec = _make_rec(kelly_fraction=0.03)

        # Lose 3 in a row then win
        mgr.place_and_settle(rec, outcome=False)
        mgr.place_and_settle(rec, outcome=False)
        mgr.place_and_settle(rec, outcome=False)
        mgr.place_and_settle(rec, outcome=True)

        stats = mgr.get_stats()
        assert stats.max_drawdown_pct > 0

    def test_stats_with_no_bets(self):
        """Stats are valid even with no bets."""
        mgr = BankrollManager(initial_bankroll=1000.0)
        stats = mgr.get_stats()
        assert stats.total_bets == 0
        assert stats.roi_pct == 0.0

    def test_stats_win_rate(self):
        """Win rate is calculated correctly."""
        mgr = BankrollManager(initial_bankroll=1000.0)
        rec = _make_rec(kelly_fraction=0.02)

        mgr.place_and_settle(rec, outcome=True)
        mgr.place_and_settle(rec, outcome=True)
        mgr.place_and_settle(rec, outcome=False)

        stats = mgr.get_stats()
        assert abs(stats.win_rate - 2 / 3) < 0.01

    def test_to_dataframe(self):
        """History can be exported as DataFrame."""
        mgr = BankrollManager(initial_bankroll=1000.0)
        rec = _make_rec(kelly_fraction=0.02)
        mgr.place_and_settle(rec, outcome=True)
        mgr.place_and_settle(rec, outcome=False)

        df = mgr.to_dataframe()
        assert len(df) == 2
        assert "pnl" in df.columns
        assert "bankroll_after" in df.columns

    def test_longest_losing_streak(self):
        """Longest losing streak is tracked correctly."""
        mgr = BankrollManager(initial_bankroll=1000.0)
        rec = _make_rec(kelly_fraction=0.01)

        outcomes = [True, False, False, False, True, False, False]
        for o in outcomes:
            mgr.place_and_settle(rec, outcome=o)

        stats = mgr.get_stats()
        assert stats.longest_losing_streak == 3
