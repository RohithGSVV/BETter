"""Tests for odds conversion and edge detection."""

from better.utils.stats import (
    american_to_decimal,
    implied_probability_from_american,
    kelly_criterion,
    remove_vig,
)


def test_vig_removal_sums_to_one():
    """Fair probabilities should always sum to 1.0."""
    test_cases = [
        (-150, 130),
        (-200, 170),
        (-110, -110),
        (100, -120),
    ]
    for home, away in test_cases:
        h_imp = implied_probability_from_american(home)
        a_imp = implied_probability_from_american(away)
        h_fair, a_fair = remove_vig(h_imp, a_imp)
        assert abs(h_fair + a_fair - 1.0) < 0.0001, f"Failed for {home}/{away}"


def test_edge_calculation():
    """Model prob minus fair implied should give the edge."""
    model_prob = 0.62
    market_odds = -150  # Implied ~60%
    implied = implied_probability_from_american(market_odds)
    edge = model_prob - implied
    assert edge > 0  # Model sees value


def test_kelly_sizing_proportional_to_edge():
    """Larger edges should produce larger Kelly fractions."""
    small_edge = kelly_criterion(0.55, 2.0, fraction=0.25)
    large_edge = kelly_criterion(0.65, 2.0, fraction=0.25)
    assert large_edge > small_edge
