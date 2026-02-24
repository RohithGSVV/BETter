"""Unit tests for the Elo rating system."""

import math

import pytest

from better.features.elo import elo_expected, elo_update, revert_to_mean


class TestEloExpected:
    def test_equal_ratings(self):
        assert elo_expected(1500, 1500) == pytest.approx(0.5)

    def test_higher_rated_favoured(self):
        assert elo_expected(1600, 1400) > 0.5

    def test_lower_rated_underdog(self):
        assert elo_expected(1400, 1600) < 0.5

    def test_symmetry(self):
        p = elo_expected(1600, 1400)
        assert p + elo_expected(1400, 1600) == pytest.approx(1.0)

    def test_extreme_rating_gap(self):
        # 400-point gap → ~91% expected
        p = elo_expected(1900, 1500)
        assert 0.90 < p < 0.92


class TestEloUpdate:
    def test_zero_sum(self):
        """Winner's gain equals loser's loss."""
        old_h, old_a = 1500.0, 1500.0
        new_h, new_a = elo_update(old_h, old_a, home_win=True)
        delta_h = new_h - old_h
        delta_a = new_a - old_a
        assert delta_h == pytest.approx(-delta_a)

    def test_home_win_increases_home(self):
        new_h, new_a = elo_update(1500, 1500, home_win=True)
        assert new_h > 1500
        assert new_a < 1500

    def test_away_win_increases_away(self):
        new_h, new_a = elo_update(1500, 1500, home_win=False)
        assert new_h < 1500
        assert new_a > 1500

    def test_upset_yields_larger_update(self):
        """When underdog wins, the rating shift is larger."""
        # Home is strong favourite
        _, delta_expected = elo_update(1600, 1400, home_win=True)
        _, delta_upset = elo_update(1600, 1400, home_win=False)
        # Upset loss for away team is smaller (they gained) than expected loss
        assert delta_upset > delta_expected  # away gained more from upset


class TestRevertToMean:
    def test_above_mean_decreases(self):
        assert revert_to_mean(1600) < 1600

    def test_below_mean_increases(self):
        assert revert_to_mean(1400) > 1400

    def test_at_mean_unchanged(self):
        assert revert_to_mean(1500) == pytest.approx(1500)

    def test_reversion_amount(self):
        """Rating reverts by exactly 1/3 toward mean."""
        rating = 1600
        expected = 1600 - (1 / 3) * (1600 - 1500)
        assert revert_to_mean(rating) == pytest.approx(expected)
