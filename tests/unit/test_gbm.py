"""Tests for GBM ensemble models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from better.models.gbm.ensemble import (
    CatBoostModel,
    GBMEnsemble,
    LightGBMModel,
    XGBoostModel,
    _prepare_features,
)


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic dataset with plausible feature columns."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(
        {
            "game_pk": range(n),
            "game_date": pd.date_range("2023-04-01", periods=n, freq="D"),
            "season": [2023] * 150 + [2024] * 150,
            "home_team": rng.choice(["NYY", "BOS", "LAD"], n),
            "away_team": rng.choice(["HOU", "ATL", "SDP"], n),
            "home_win": rng.integers(0, 2, n),
            "home_bayesian_strength": rng.normal(0, 0.3, n),
            "away_bayesian_strength": rng.normal(0, 0.3, n),
            "elo_diff": rng.normal(0, 100, n),
            "strength_diff": rng.normal(0, 0.3, n),
            "sp_fip_diff": rng.normal(0, 1, n),
            "run_diff_momentum": rng.normal(0, 10, n),
            "park_runs_factor": rng.choice([0.92, 1.0, 1.05, 1.15], n),
            "home_pythag_win_pct_30": rng.uniform(0.35, 0.65, n),
            "away_pythag_win_pct_30": rng.uniform(0.35, 0.65, n),
            "home_run_diff_7": rng.normal(0, 5, n),
            "away_run_diff_7": rng.normal(0, 5, n),
            "home_ewma_win_rate_30": rng.uniform(0.3, 0.7, n),
            "away_ewma_win_rate_30": rng.uniform(0.3, 0.7, n),
            "elo_home_win_prob": rng.uniform(0.4, 0.6, n),
            "home_sp_fip": rng.uniform(2.5, 5.5, n),
            "away_sp_fip": rng.uniform(2.5, 5.5, n),
        }
    )
    y = X.pop("home_win").astype(int)
    return X, y


class TestXGBoost:
    def test_fit_predict_shape(self, synthetic_data):
        X, y = synthetic_data
        model = XGBoostModel()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert preds.shape == (len(X),)

    def test_predictions_bounded(self, synthetic_data):
        X, y = synthetic_data
        model = XGBoostModel()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert np.all(preds >= 0) and np.all(preds <= 1)


class TestLightGBM:
    def test_fit_predict_shape(self, synthetic_data):
        X, y = synthetic_data
        model = LightGBMModel()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert preds.shape == (len(X),)


class TestCatBoost:
    def test_fit_predict_shape(self, synthetic_data):
        X, y = synthetic_data
        model = CatBoostModel()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert preds.shape == (len(X),)


class TestGBMEnsemble:
    def test_ensemble_averages_components(self, synthetic_data):
        X, y = synthetic_data
        ensemble = GBMEnsemble()
        ensemble.fit(X, y)
        ensemble_preds = ensemble.predict_proba(X)

        component_preds = np.column_stack(
            [m.predict_proba(X) for m in ensemble.models]
        )
        expected = component_preds.mean(axis=1)
        np.testing.assert_allclose(ensemble_preds, expected, atol=1e-10)

    def test_feature_importance_keys(self, synthetic_data):
        X, y = synthetic_data
        ensemble = GBMEnsemble()
        ensemble.fit(X, y)
        imp = ensemble.feature_importance
        assert len(imp) > 0
        # All importance values should be non-negative
        assert all(v >= 0 for v in imp.values())


class TestPrepareFeatures:
    def test_drops_identifier_cols(self, synthetic_data):
        X, _ = synthetic_data
        Xc = _prepare_features(X)
        assert "game_pk" not in Xc.columns
        assert "home_team" not in Xc.columns
        assert "game_date" not in Xc.columns
