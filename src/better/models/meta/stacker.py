"""Meta-learner stacking ensemble with probability calibration.

Takes out-of-fold predictions from the base models (GBM ensemble, Bayesian,
Monte Carlo) as features and trains a second-level model to produce the
final calibrated P(home_win).

CRITICAL: The base model predictions used as meta-features must be
out-of-fold predictions to prevent information leakage.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from better.models import BasePredictor
from better.utils.logging import get_logger

log = get_logger(__name__)


class MetaLearner(BasePredictor):
    """Logistic regression meta-learner with Platt scaling."""

    def __init__(
        self,
        include_raw_features: bool = True,
        raw_feature_cols: list[str] | None = None,
        calibrate: bool = True,
    ):
        self.include_raw_features = include_raw_features
        self.raw_feature_cols = raw_feature_cols or [
            "elo_diff",
            "strength_diff",
            "sp_fip_diff",
            "run_diff_momentum",
            "park_runs_factor",
        ]
        self.calibrate = calibrate
        self._model: LogisticRegression | CalibratedClassifierCV | None = None
        self._meta_feature_names: list[str] = []

    @property
    def name(self) -> str:
        return "meta_learner"

    def _build_meta_features(
        self,
        base_predictions: dict[str, np.ndarray],
        X_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Assemble meta-feature matrix from base model outputs + raw features."""
        cols: list[np.ndarray] = []
        names: list[str] = []

        for model_name in sorted(base_predictions.keys()):
            cols.append(base_predictions[model_name].reshape(-1, 1))
            names.append(f"pred_{model_name}")

        if self.include_raw_features and X_raw is not None:
            for col in self.raw_feature_cols:
                if col in X_raw.columns:
                    vals = X_raw[col].fillna(0).values.reshape(-1, 1)
                    cols.append(vals)
                    names.append(f"raw_{col}")

        self._meta_feature_names = names
        return np.hstack(cols)

    def fit_from_base_predictions(
        self,
        base_predictions: dict[str, np.ndarray],
        y: pd.Series,
        X_raw: pd.DataFrame | None = None,
    ) -> None:
        """Train the meta-learner on out-of-fold base model predictions."""
        meta_X = self._build_meta_features(base_predictions, X_raw)

        if self.calibrate:
            base_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            self._model = CalibratedClassifierCV(
                base_lr, cv=5, method="sigmoid"
            )
        else:
            self._model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        self._model.fit(meta_X, y.astype(int))
        log.info(
            "meta_learner_fitted",
            meta_features=self._meta_feature_names,
            n_samples=len(y),
        )

    def predict_from_base_predictions(
        self,
        base_predictions: dict[str, np.ndarray],
        X_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Predict using base model outputs."""
        meta_X = self._build_meta_features(base_predictions, X_raw)
        return self._model.predict_proba(meta_X)[:, 1]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError(
            "Use fit_from_base_predictions() instead."
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "Use predict_from_base_predictions() instead."
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "meta_feature_names": self._meta_feature_names,
                "raw_feature_cols": self.raw_feature_cols,
                "include_raw_features": self.include_raw_features,
                "calibrate": self.calibrate,
            },
            path / "meta_learner.joblib",
        )

    def load(self, path: Path) -> None:
        data = joblib.load(path / "meta_learner.joblib")
        self._model = data["model"]
        self._meta_feature_names = data["meta_feature_names"]
        self.raw_feature_cols = data["raw_feature_cols"]
        self.include_raw_features = data["include_raw_features"]
        self.calibrate = data["calibrate"]
