"""Tests for the BettingEngine core."""

from datetime import date

import pytest

from better.betting.engine import BetRecommendation, BettingEngine, _fair_prob_to_american


class TestBettingEngine:
    """Tests for BettingEngine edge detection and Kelly sizing."""

    def setup_method(self):
        self.engine = BettingEngine(
            bankroll=1000.0,
            min_edge=0.03,
            kelly_fraction=0.25,
            max_bet_pct=0.05,
        )

    def test_no_edge_returns_none(self):
        """When model agrees with market, no bet recommended."""
        # Model says 55% home, market says 55% home (roughly -122/+102)
        result = self.engine.evaluate_game(
            model_home_prob=0.55,
            home_odds_american=-122,
            away_odds_american=102,
        )
        assert result is None

    def test_positive_edge_returns_recommendation(self):
        """When model has significant edge, returns a bet."""
        # Model says 65% home, market says 55% (odds imply ~55% after vig removal)
        result = self.engine.evaluate_game(
            model_home_prob=0.65,
            home_odds_american=-120,
            away_odds_american=100,
            home_team="NYY",
            away_team="BOS",
        )
        assert result is not None
        assert result.bet_side == "HOME"
        assert result.edge > 0.03
        assert result.model_prob == 0.65

    def test_away_edge(self):
        """When model favors the away team more than market, bets away."""
        # Model says 40% home (60% away), market says 55% home
        result = self.engine.evaluate_game(
            model_home_prob=0.40,
            home_odds_american=-120,
            away_odds_american=100,
            home_team="NYY",
            away_team="BOS",
        )
        assert result is not None
        assert result.bet_side == "AWAY"

    def test_max_bet_cap(self):
        """Bet amount never exceeds max_bet_pct of bankroll."""
        result = self.engine.evaluate_game(
            model_home_prob=0.90,  # Huge edge
            home_odds_american=-110,
            away_odds_american=-110,
            home_team="NYY",
            away_team="BOS",
        )
        assert result is not None
        assert result.bet_amount <= 1000.0 * 0.05  # max_bet_pct

    def test_kelly_fraction_applied(self):
        """Kelly fraction reduces bet size from full Kelly."""
        # With quarter Kelly, bet should be 25% of full Kelly
        full_kelly_engine = BettingEngine(
            bankroll=1000.0,
            min_edge=0.01,
            kelly_fraction=1.0,  # Full Kelly
            max_bet_pct=1.0,  # No cap
        )
        quarter_kelly_engine = BettingEngine(
            bankroll=1000.0,
            min_edge=0.01,
            kelly_fraction=0.25,  # Quarter Kelly
            max_bet_pct=1.0,  # No cap
        )

        full = full_kelly_engine.evaluate_game(
            model_home_prob=0.60,
            home_odds_american=-110,
            away_odds_american=-110,
        )
        quarter = quarter_kelly_engine.evaluate_game(
            model_home_prob=0.60,
            home_odds_american=-110,
            away_odds_american=-110,
        )
        assert full is not None and quarter is not None
        assert abs(quarter.kelly_fraction - full.kelly_fraction * 0.25) < 0.01

    def test_confidence_buckets(self):
        """Edge size maps to correct confidence level."""
        # Low edge: fair prob at -110/-110 is 50%, so 0.535 gives edge ~3.5%
        low = self.engine.evaluate_game(
            model_home_prob=0.535,
            home_odds_american=-110,
            away_odds_american=-110,
        )
        if low:
            assert low.confidence == "low"

        # High edge: 0.59 at 50% fair prob gives edge ~9% -> high
        high = self.engine.evaluate_game(
            model_home_prob=0.59,
            home_odds_american=-110,
            away_odds_american=-110,
        )
        if high:
            assert high.confidence == "high"

    def test_evaluate_game_fair_prob(self):
        """Fair probability evaluation (for backtesting) works correctly."""
        result = self.engine.evaluate_game_fair_prob(
            model_home_prob=0.65,
            market_home_fair_prob=0.55,
            home_team="NYY",
            away_team="BOS",
        )
        assert result is not None
        assert result.edge == round(0.65 - 0.55, 4)
        assert result.bet_side == "HOME"
        assert result.bookmaker == "synthetic_elo"

    def test_evaluate_game_fair_prob_no_edge(self):
        """Fair prob evaluation returns None when edge is below threshold."""
        result = self.engine.evaluate_game_fair_prob(
            model_home_prob=0.56,
            market_home_fair_prob=0.55,
        )
        assert result is None


class TestBestLine:
    """Tests for line shopping across bookmakers."""

    def test_picks_best_positive_odds(self):
        """Selects the bookmaker with the highest positive odds."""
        odds = {"DraftKings": 115, "FanDuel": 120, "BetMGM": 110}
        book, best = BettingEngine.best_line(odds)
        assert book == "FanDuel"
        assert best == 120

    def test_picks_best_negative_odds(self):
        """Selects the bookmaker with the least negative odds (closest to even)."""
        odds = {"DraftKings": -115, "FanDuel": -105, "BetMGM": -110}
        book, best = BettingEngine.best_line(odds)
        assert book == "FanDuel"
        assert best == -105

    def test_empty_fallback(self):
        """Returns fallback when no bookmakers provided."""
        book, odds = BettingEngine.best_line({})
        assert book == "unknown"
        assert odds == -110


class TestFairProbToAmerican:
    """Tests for probability to American odds conversion."""

    def test_favorite(self):
        """60% probability -> negative American odds."""
        odds = _fair_prob_to_american(0.60)
        assert odds < 0  # Should be around -150

    def test_underdog(self):
        """40% probability -> positive American odds."""
        odds = _fair_prob_to_american(0.40)
        assert odds > 0  # Should be around +150

    def test_even(self):
        """50% probability -> even money."""
        odds = _fair_prob_to_american(0.50)
        assert odds == -100 or odds == 100  # Boundary case
