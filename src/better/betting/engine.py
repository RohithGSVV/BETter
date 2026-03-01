"""Betting engine: edge detection, Kelly sizing, and bet recommendations.

Given model predictions and market odds, identifies +EV bets and
computes optimal position sizes using fractional Kelly criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from better.config import settings
from better.utils.logging import get_logger
from better.utils.stats import (
    american_to_decimal,
    implied_probability_from_american,
    kelly_criterion,
    remove_vig,
)

log = get_logger(__name__)


@dataclass
class BetRecommendation:
    """A single actionable bet recommendation."""

    game_date: date | None
    home_team: str
    away_team: str
    bet_side: str  # "HOME" or "AWAY"
    model_prob: float  # Our model's P(bet side wins)
    market_prob: float  # Market fair P(bet side wins) after vig removal
    edge: float  # model_prob - market_prob
    expected_value: float  # EV per $1 wagered
    kelly_fraction: float  # Optimal bankroll fraction (before caps)
    bet_amount: float  # Actual $ bet (after caps)
    odds_american: int  # Best available line
    bookmaker: str  # Which book has the best line
    confidence: str  # "low" / "medium" / "high"
    game_pk: int | None = None


class BettingEngine:
    """Core betting decision-maker.

    Takes model predictions + market odds → computes edges → recommends bets.

    Parameters
    ----------
    bankroll : float
        Current bankroll balance.
    min_edge : float
        Minimum edge (model_prob - market_prob) to place a bet. Default 0.03 (3%).
    kelly_fraction : float
        Fraction of full Kelly to use (0.25 = quarter-Kelly). Default 0.25.
    max_bet_pct : float
        Maximum bet as fraction of bankroll. Default 0.05 (5%).
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        min_edge: float | None = None,
        kelly_fraction: float | None = None,
        max_bet_pct: float | None = None,
    ):
        self.bankroll = bankroll
        self.min_edge = min_edge if min_edge is not None else settings.min_edge_threshold
        self.kelly_fraction = (
            kelly_fraction if kelly_fraction is not None else settings.kelly_fraction
        )
        self.max_bet_pct = (
            max_bet_pct if max_bet_pct is not None else settings.max_bet_pct
        )

    def evaluate_game(
        self,
        model_home_prob: float,
        home_odds_american: int,
        away_odds_american: int,
        home_team: str = "",
        away_team: str = "",
        game_date: date | None = None,
        bookmaker: str = "consensus",
        game_pk: int | None = None,
    ) -> BetRecommendation | None:
        """Evaluate a single game for betting opportunities.

        Returns a BetRecommendation if edge exceeds threshold, else None.
        """
        # Convert market odds to fair probabilities (vig removed)
        home_implied = implied_probability_from_american(home_odds_american)
        away_implied = implied_probability_from_american(away_odds_american)
        home_fair, away_fair = remove_vig(home_implied, away_implied)

        model_away_prob = 1.0 - model_home_prob

        # Check both sides for edge
        home_edge = model_home_prob - home_fair
        away_edge = model_away_prob - away_fair

        # Pick the side with more edge (if any)
        if home_edge >= away_edge and home_edge > self.min_edge:
            return self._build_recommendation(
                bet_side="HOME",
                model_prob=model_home_prob,
                market_prob=home_fair,
                odds_american=home_odds_american,
                home_team=home_team,
                away_team=away_team,
                game_date=game_date,
                bookmaker=bookmaker,
                game_pk=game_pk,
            )
        elif away_edge > home_edge and away_edge > self.min_edge:
            return self._build_recommendation(
                bet_side="AWAY",
                model_prob=model_away_prob,
                market_prob=away_fair,
                odds_american=away_odds_american,
                home_team=home_team,
                away_team=away_team,
                game_date=game_date,
                bookmaker=bookmaker,
                game_pk=game_pk,
            )

        return None

    def evaluate_game_fair_prob(
        self,
        model_home_prob: float,
        market_home_fair_prob: float,
        home_team: str = "",
        away_team: str = "",
        game_date: date | None = None,
        game_pk: int | None = None,
    ) -> BetRecommendation | None:
        """Evaluate a game using pre-computed fair probabilities (no raw odds).

        Used by the backtester with synthetic market probabilities (e.g., Elo).
        """
        market_away_fair = 1.0 - market_home_fair_prob
        model_away_prob = 1.0 - model_home_prob

        home_edge = model_home_prob - market_home_fair_prob
        away_edge = model_away_prob - market_away_fair

        # Convert fair prob to synthetic American odds for Kelly calculation
        home_odds = _fair_prob_to_american(market_home_fair_prob)
        away_odds = _fair_prob_to_american(market_away_fair)

        if home_edge >= away_edge and home_edge > self.min_edge:
            return self._build_recommendation(
                bet_side="HOME",
                model_prob=model_home_prob,
                market_prob=market_home_fair_prob,
                odds_american=home_odds,
                home_team=home_team,
                away_team=away_team,
                game_date=game_date,
                bookmaker="synthetic_elo",
                game_pk=game_pk,
            )
        elif away_edge > home_edge and away_edge > self.min_edge:
            return self._build_recommendation(
                bet_side="AWAY",
                model_prob=model_away_prob,
                market_prob=market_away_fair,
                odds_american=away_odds,
                home_team=home_team,
                away_team=away_team,
                game_date=game_date,
                bookmaker="synthetic_elo",
                game_pk=game_pk,
            )

        return None

    def _build_recommendation(
        self,
        bet_side: str,
        model_prob: float,
        market_prob: float,
        odds_american: int,
        home_team: str,
        away_team: str,
        game_date: date | None,
        bookmaker: str,
        game_pk: int | None,
    ) -> BetRecommendation:
        """Construct a BetRecommendation with Kelly sizing."""
        edge = model_prob - market_prob
        decimal_odds = american_to_decimal(odds_american)
        ev = model_prob * (decimal_odds - 1) - (1.0 - model_prob)

        # Kelly criterion (fractional)
        kelly = kelly_criterion(model_prob, decimal_odds, self.kelly_fraction)

        # Cap at max_bet_pct of bankroll
        bet_fraction = min(kelly, self.max_bet_pct)
        bet_amount = round(self.bankroll * bet_fraction, 2)

        # Confidence bucket
        if edge >= 0.07:
            confidence = "high"
        elif edge >= 0.04:
            confidence = "medium"
        else:
            confidence = "low"

        return BetRecommendation(
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            bet_side=bet_side,
            model_prob=round(model_prob, 4),
            market_prob=round(market_prob, 4),
            edge=round(edge, 4),
            expected_value=round(ev, 4),
            kelly_fraction=round(kelly, 4),
            bet_amount=bet_amount,
            odds_american=odds_american,
            bookmaker=bookmaker,
            confidence=confidence,
            game_pk=game_pk,
        )

    @staticmethod
    def best_line(
        odds_by_bookmaker: dict[str, int],
    ) -> tuple[str, int]:
        """Find the best (highest) odds across bookmakers for a given side.

        Parameters
        ----------
        odds_by_bookmaker : dict
            Mapping of bookmaker name → American odds for one side.

        Returns
        -------
        (bookmaker, best_odds) — the book with the best line.
        """
        if not odds_by_bookmaker:
            return ("unknown", -110)

        # For American odds: higher is better for the bettor
        # -105 is better than -110; +120 is better than +115
        best_book = max(odds_by_bookmaker, key=odds_by_bookmaker.get)  # type: ignore[arg-type]
        return best_book, odds_by_bookmaker[best_book]


def _fair_prob_to_american(prob: float) -> int:
    """Convert a fair probability to American odds (no vig)."""
    if prob <= 0 or prob >= 1:
        return -110  # fallback
    if prob >= 0.5:
        # Favorite: negative odds
        return int(-100 * prob / (1.0 - prob))
    else:
        # Underdog: positive odds
        return int(100 * (1.0 - prob) / prob)
