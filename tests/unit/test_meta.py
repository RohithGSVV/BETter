"""Tests for meta-learner stacking ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from better.models.meta.stacker import MetaLearner


@pytest.fixture
def oof_data() -> tuple[dict[str, np.ndarray], pd.Series, pd.DataFrame]:
    """Synthetic OOF predictions from 3 base models."""
    rng = np.random.default_rng(42)
    n = 500
    y = pd.Series(rng.integers(0, 2, n))

    # Base model predictions — slightly correlated with truth
    noise = 0.3
    base_preds = {
        "bayesian_kalman": np.clip(y.values * 0.6 + rng.normal(0, noise, n), 0.05, 0.95),
        "gbm_ensemble": np.clip(y.values * 0.65 + rng.normal(0, noise, n), 0.05, 0.95),
        "monte_carlo": np.clip(y.values * 0.55 + rng.normal(0, noise, n), 0.05, 0.95),
    }

    X_raw = pd.DataFrame(
        {
            "elo_diff": rng.normal(0, 100, n),
            "strength_diff": rng.normal(0, 0.3, n),
            "sp_fip_diff": rng.normal(0, 1, n),
            "run_diff_momentum": rng.normal(0, 10, n),
            "park_runs_factor": rng.choice([0.92, 1.0, 1.05, 1.15], n),
        }
    )
    return base_preds, y, X_raw


class TestMetaLearner:
    def test_predictions_bounded(self, oof_data):
        base_preds, y, X_raw = oof_data
        meta = MetaLearner(include_raw_features=True)
        meta.fit_from_base_predictions(base_preds, y, X_raw)
        preds = meta.predict_from_base_predictions(base_preds, X_raw)
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_output_shape(self, oof_data):
        base_preds, y, X_raw = oof_data
        meta = MetaLearner(include_raw_features=True)
        meta.fit_from_base_predictions(base_preds, y, X_raw)
        preds = meta.predict_from_base_predictions(base_preds, X_raw)
        assert preds.shape == (len(y),)

    def test_raw_features_included(self, oof_data):
        base_preds, y, X_raw = oof_data
        meta = MetaLearner(include_raw_features=True)
        meta.fit_from_base_predictions(base_preds, y, X_raw)
        # 3 base models + 5 raw features = 8 meta-features
        assert len(meta._meta_feature_names) == 8

    def test_without_raw_features(self, oof_data):
        base_preds, y, X_raw = oof_data
        meta = MetaLearner(include_raw_features=False)
        meta.fit_from_base_predictions(base_preds, y)
        assert len(meta._meta_feature_names) == 3

    def test_save_load_roundtrip(self, oof_data, tmp_path):
        base_preds, y, X_raw = oof_data
        meta = MetaLearner(include_raw_features=True)
        meta.fit_from_base_predictions(base_preds, y, X_raw)
        preds_before = meta.predict_from_base_predictions(base_preds, X_raw)

        meta.save(tmp_path / "meta")

        loaded = MetaLearner()
        loaded.load(tmp_path / "meta")
        preds_after = loaded.predict_from_base_predictions(base_preds, X_raw)

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-10)

    def test_fit_raises_not_implemented(self, oof_data):
        meta = MetaLearner()
        with pytest.raises(NotImplementedError):
            meta.fit(pd.DataFrame(), pd.Series(dtype=int))
