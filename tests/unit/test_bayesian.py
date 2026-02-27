"""Tests for Bayesian Kalman team-strength model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from better.models.bayesian.kalman import BayesianKalmanModel, KalmanParams, TeamState


@pytest.fixture
def simple_games() -> tuple[pd.DataFrame, pd.Series]:
    """A small set of games where NYY always wins and BOS always loses."""
    n = 20
    df = pd.DataFrame(
        {
            "game_date": pd.date_range("2023-04-01", periods=n, freq="D"),
            "season": [2023] * n,
            "home_team": ["NYY"] * 10 + ["BOS"] * 10,
            "away_team": ["BOS"] * 10 + ["NYY"] * 10,
        }
    )
    # NYY wins all games (home or away)
    y = pd.Series([1] * 10 + [0] * 10)
    return df, y


class TestBayesianKalman:
    def test_equal_teams_near_home_advantage(self):
        """Two unseen teams should predict ~54% (home advantage)."""
        model = BayesianKalmanModel()
        p = model._predict_game("AAA", "BBB")
        # Home advantage logit ~0.16 → sigmoid(0.16) ≈ 0.54
        assert 0.52 < p < 0.56

    def test_strong_team_predicted_higher(self, simple_games):
        """After fitting with NYY always winning, NYY should be favored."""
        X, y = simple_games
        model = BayesianKalmanModel()
        model.fit(X, y)

        # NYY at home vs BOS should be strongly favored
        p = model._predict_game("NYY", "BOS")
        assert p > 0.60

    def test_winner_strength_increases(self, simple_games):
        """After fitting, the winner's mu should be positive."""
        X, y = simple_games
        model = BayesianKalmanModel()
        model.fit(X, y)

        assert model.team_states["NYY"].mu > 0
        assert model.team_states["BOS"].mu < 0

    def test_variance_decreases_with_games(self):
        """Sigma2 should shrink as more games are observed."""
        model = BayesianKalmanModel()
        initial_var = model._get_state("TEST").sigma2

        # Simulate some games
        for _ in range(10):
            model._update_after_game("TEST", "OPP", True)

        assert model.team_states["TEST"].sigma2 < initial_var

    def test_season_boundary_reverts(self):
        """Strengths should move toward 0 at season boundary."""
        model = BayesianKalmanModel()
        state = model._get_state("TEST")
        state.mu = 0.5  # Strong team

        model._season_boundary()

        # Should be closer to 0 (reverted by 33%)
        assert abs(state.mu) < 0.5
        assert state.mu == pytest.approx(0.5 * (1 - 0.33), abs=0.01)

    def test_save_load_roundtrip(self, simple_games, tmp_path):
        """Saving and loading should preserve team states."""
        X, y = simple_games
        model = BayesianKalmanModel()
        model.fit(X, y)

        model.save(tmp_path / "bayesian")

        loaded = BayesianKalmanModel()
        loaded.load(tmp_path / "bayesian")

        assert loaded.team_states["NYY"].mu == pytest.approx(
            model.team_states["NYY"].mu
        )
