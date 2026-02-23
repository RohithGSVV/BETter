"""Tests for statistical utility functions."""

import pytest

from better.utils.stats import (
    american_to_decimal,
    implied_probability_from_american,
    kelly_criterion,
    pythagorean_win_pct,
    remove_vig,
)


class TestPythagorean:
    def test_equal_runs(self):
        assert pythagorean_win_pct(100, 100) == pytest.approx(0.5)

    def test_dominant_team(self):
        # Team scoring twice as many runs should win ~77%
        result = pythagorean_win_pct(200, 100)
        assert 0.75 < result < 0.80

    def test_weak_team(self):
        result = pythagorean_win_pct(100, 200)
        assert 0.20 < result < 0.25

    def test_zero_runs(self):
        assert pythagorean_win_pct(0, 0) == 0.5
        assert pythagorean_win_pct(100, 0) == 1.0


class TestOddsConversion:
    def test_negative_american_to_implied(self):
        # -150 favorite: 150/250 = 0.6
        assert implied_probability_from_american(-150) == pytest.approx(0.6)

    def test_positive_american_to_implied(self):
        # +150 underdog: 100/250 = 0.4
        assert implied_probability_from_american(150) == pytest.approx(0.4)

    def test_even_odds(self):
        assert implied_probability_from_american(100) == pytest.approx(0.5)
        assert implied_probability_from_american(-100) == pytest.approx(0.5)

    def test_american_to_decimal(self):
        assert american_to_decimal(-150) == pytest.approx(1.6667, abs=0.001)
        assert american_to_decimal(150) == pytest.approx(2.5)
        assert american_to_decimal(-100) == pytest.approx(2.0)

    def test_remove_vig(self):
        # Typical vig: -150/+130 (overround ~3.5%)
        home_imp = implied_probability_from_american(-150)  # 0.6
        away_imp = implied_probability_from_american(130)  # 0.4348
        home_fair, away_fair = remove_vig(home_imp, away_imp)
        assert home_fair + away_fair == pytest.approx(1.0)
        assert home_fair > 0.5  # Home is still favored


class TestKelly:
    def test_no_edge(self):
        # If model prob equals implied prob, Kelly = 0
        assert kelly_criterion(0.5, 2.0) == 0.0

    def test_positive_edge(self):
        # 60% model prob, +100 odds (decimal 2.0)
        # edge = 1*0.6 - 0.4 = 0.2, f* = 0.25 * 0.2 / 1 = 0.05
        result = kelly_criterion(0.6, 2.0, fraction=0.25)
        assert result == pytest.approx(0.05)

    def test_negative_edge(self):
        # 40% prob on +100 odds = negative edge
        assert kelly_criterion(0.4, 2.0) == 0.0

    def test_heavy_favorite(self):
        # 70% prob, -200 odds (decimal 1.5)
        result = kelly_criterion(0.7, 1.5, fraction=0.25)
        assert result > 0
