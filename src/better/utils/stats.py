"""Statistical helper functions for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pythagorean_win_pct(
    runs_scored: float, runs_allowed: float, exponent: float = 1.83
) -> float:
    """Compute Pythagorean win percentage.

    Formula: RS^exp / (RS^exp + RA^exp)
    This predicts a team's "true" winning percentage better than actual record.
    """
    if runs_scored <= 0 and runs_allowed <= 0:
        return 0.5
    rs_exp = runs_scored**exponent
    ra_exp = runs_allowed**exponent
    denom = rs_exp + ra_exp
    if denom == 0:
        return 0.5
    return rs_exp / denom


def ewma(values: pd.Series, span: int) -> pd.Series:
    """Compute exponentially weighted moving average.

    Args:
        values: Time-ordered series of values.
        span: Span parameter (larger = slower decay).
    """
    return values.ewm(span=span, adjust=False).mean()


def rolling_mean(values: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """Compute rolling mean with configurable minimum periods."""
    return values.rolling(window=window, min_periods=min_periods).mean()


def rolling_sum(values: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """Compute rolling sum with configurable minimum periods."""
    return values.rolling(window=window, min_periods=min_periods).sum()


def game_score(
    ip: float,
    hits: int,
    runs: int,
    earned_runs: int,
    walks: int,
    strikeouts: int,
    home_runs: int,
) -> float:
    """Compute Bill James' Game Score for a pitcher's outing.

    Starts at 50, rewards strikeouts and innings, penalizes hits/runs/walks.
    """
    score = 50.0
    score += ip * 3  # 3 points per out (approx via IP)
    score += strikeouts
    score -= hits * 2
    score -= earned_runs * 4
    score -= (runs - earned_runs) * 2
    score -= walks * 2
    score -= home_runs * 6
    return score


def normalize_to_range(
    values: pd.Series, low: float = 0.0, high: float = 1.0
) -> pd.Series:
    """Min-max normalize a series to [low, high]."""
    vmin = values.min()
    vmax = values.max()
    if vmax == vmin:
        return pd.Series([(low + high) / 2] * len(values), index=values.index)
    return low + (values - vmin) * (high - low) / (vmax - vmin)


def implied_probability_from_american(odds: int) -> float:
    """Convert American odds to implied probability.

    Negative odds (favorites): |odds| / (|odds| + 100)
    Positive odds (underdogs): 100 / (odds + 100)
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    else:
        return 0.5


def remove_vig(home_implied: float, away_implied: float) -> tuple[float, float]:
    """Remove bookmaker vig (overround) from implied probabilities.

    Normalizes so probabilities sum to 1.0.
    Returns (home_fair_prob, away_fair_prob).
    """
    total = home_implied + away_implied
    if total == 0:
        return 0.5, 0.5
    return home_implied / total, away_implied / total


def kelly_criterion(
    win_prob: float, decimal_odds: float, fraction: float = 0.25
) -> float:
    """Compute fractional Kelly criterion bet size.

    f* = fraction * (b*p - q) / b
    where b = decimal_odds - 1, p = win_prob, q = 1 - p.

    Returns fraction of bankroll to bet (0 if no edge).
    """
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    q = 1 - win_prob
    edge = b * win_prob - q
    if edge <= 0:
        return 0.0
    return fraction * edge / b


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds < 0:
        return 1 + 100 / abs(odds)
    elif odds > 0:
        return 1 + odds / 100
    else:
        return 2.0
