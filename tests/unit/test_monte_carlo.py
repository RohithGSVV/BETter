"""Tests for Monte Carlo run-scoring simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from better.models.monte_carlo.simulator import MCParams, MonteCarloSimulator


@pytest.fixture
def fitted_simulator() -> tuple[MonteCarloSimulator, pd.DataFrame]:
    """Create a fitted simulator with known team rates."""
    sim = MonteCarloSimulator(params=MCParams(n_sims=5000))
    # Manually set team rates instead of fitting from data
    sim._team_run_rates = {
        "NYY": 5.0,  # strong offense
        "BOS": 4.5,  # average offense
        "MIA": 3.5,  # weak offense
    }
    sim._team_run_allowed_rates = {
        "NYY": 3.8,  # good pitching
        "BOS": 4.5,  # average pitching
        "MIA": 5.0,  # bad pitching
    }

    test_df = pd.DataFrame(
        {
            "home_team": ["NYY", "BOS", "MIA"],
            "away_team": ["BOS", "MIA", "NYY"],
        }
    )
    return sim, test_df


class TestMonteCarloSimulator:
    def test_home_advantage_increases_prob(self):
        """Equal teams: home team should win >50% due to home advantage."""
        sim = MonteCarloSimulator(params=MCParams(n_sims=10000))
        sim._team_run_rates = {"A": 4.5, "B": 4.5}
        sim._team_run_allowed_rates = {"A": 4.5, "B": 4.5}

        test = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
        p = sim.predict_proba(test)[0]
        assert p > 0.50

    def test_stronger_team_favored(self, fitted_simulator):
        """NYY (strong) should be favored over BOS (average)."""
        sim, test_df = fitted_simulator
        probs = sim.predict_proba(test_df)
        # NYY at home vs BOS — should be favored
        assert probs[0] > 0.55

    def test_predictions_bounded(self, fitted_simulator):
        """All predictions should be in (0, 1)."""
        sim, test_df = fitted_simulator
        probs = sim.predict_proba(test_df)
        assert np.all(probs > 0) and np.all(probs < 1)

    def test_reproducibility_with_seed(self, fitted_simulator):
        """Same seed should produce same predictions."""
        sim, test_df = fitted_simulator

        sim._rng = np.random.default_rng(123)
        preds1 = sim.predict_proba(test_df)

        sim._rng = np.random.default_rng(123)
        preds2 = sim.predict_proba(test_df)

        np.testing.assert_array_equal(preds1, preds2)

    def test_more_sims_reduces_variance(self):
        """Running with more simulations should give more stable results."""
        sim_few = MonteCarloSimulator(params=MCParams(n_sims=100))
        sim_many = MonteCarloSimulator(params=MCParams(n_sims=10000))

        for sim in [sim_few, sim_many]:
            sim._team_run_rates = {"A": 4.5, "B": 4.5}
            sim._team_run_allowed_rates = {"A": 4.5, "B": 4.5}

        test = pd.DataFrame(
            {"home_team": ["A"] * 20, "away_team": ["B"] * 20}
        )

        preds_few = sim_few.predict_proba(test)
        preds_many = sim_many.predict_proba(test)

        # More sims should have lower variance across repeated matchups
        assert np.std(preds_many) < np.std(preds_few)

    def test_save_load_roundtrip(self, fitted_simulator, tmp_path):
        """Saving and loading should preserve team rates."""
        sim, test_df = fitted_simulator
        sim.save(tmp_path / "mc")

        loaded = MonteCarloSimulator()
        loaded.load(tmp_path / "mc")

        assert loaded._team_run_rates["NYY"] == pytest.approx(5.0)
        assert loaded._team_run_allowed_rates["MIA"] == pytest.approx(5.0)
