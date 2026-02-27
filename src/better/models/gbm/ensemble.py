"""Gradient-boosted ensemble: XGBoost + LightGBM + CatBoost.

Each model is trained independently on the same feature set.  The ensemble
averages their predicted probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from better.models import BasePredictor, DROP_COLS
from better.utils.logging import get_logger

log = get_logger(__name__)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric / identifier columns, return clean feature matrix."""
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes
    return X


class XGBoostModel(BasePredictor):
    """XGBoost binary classifier wrapper."""

    def __init__(self, params: dict[str, Any] | None = None):
        self._params = params or {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
        self._early_stopping_rounds = self._params.pop("early_stopping_rounds", 50)
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple] | None = None,
    ) -> None:
        Xc = _prepare_features(X)
        self._feature_names = list(Xc.columns)
        self._model = xgb.XGBClassifier(
            early_stopping_rounds=self._early_stopping_rounds,
            **self._params,
        )
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if eval_set is not None:
            Xv = _prepare_features(eval_set[0][0])
            fit_kwargs["eval_set"] = [(Xv, eval_set[0][1])]
        else:
            fit_kwargs["eval_set"] = [(Xc, y)]
        self._model.fit(Xc, y, **fit_kwargs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xc = _prepare_features(X)
        return self._model.predict_proba(Xc)[:, 1]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "xgboost_model.joblib")
        joblib.dump(self._feature_names, path / "xgboost_features.joblib")

    def load(self, path: Path) -> None:
        self._model = joblib.load(path / "xgboost_model.joblib")
        self._feature_names = joblib.load(path / "xgboost_features.joblib")

    @property
    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {}
        return dict(zip(self._feature_names, self._model.feature_importances_))


class LightGBMModel(BasePredictor):
    """LightGBM binary classifier wrapper."""

    def __init__(self, params: dict[str, Any] | None = None):
        self._params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self._model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return "lightgbm"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple] | None = None,
    ) -> None:
        Xc = _prepare_features(X)
        self._feature_names = list(Xc.columns)
        self._model = lgb.LGBMClassifier(**self._params)
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            Xv = _prepare_features(eval_set[0][0])
            fit_kwargs["eval_set"] = [(Xv, eval_set[0][1])]
            fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        self._model.fit(Xc, y, **fit_kwargs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xc = _prepare_features(X)
        return self._model.predict_proba(Xc)[:, 1]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "lightgbm_model.joblib")
        joblib.dump(self._feature_names, path / "lightgbm_features.joblib")

    def load(self, path: Path) -> None:
        self._model = joblib.load(path / "lightgbm_model.joblib")
        self._feature_names = joblib.load(path / "lightgbm_features.joblib")

    @property
    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {}
        return dict(zip(self._feature_names, self._model.feature_importances_))


class CatBoostModel(BasePredictor):
    """CatBoost binary classifier wrapper."""

    def __init__(self, params: dict[str, Any] | None = None):
        self._params = params or {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 500,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 0,
            "early_stopping_rounds": 50,
        }
        self._model: cb.CatBoostClassifier | None = None
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return "catboost"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple] | None = None,
    ) -> None:
        Xc = _prepare_features(X)
        self._feature_names = list(Xc.columns)
        self._model = cb.CatBoostClassifier(**self._params)
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            Xv = _prepare_features(eval_set[0][0])
            fit_kwargs["eval_set"] = cb.Pool(Xv, eval_set[0][1])
        self._model.fit(Xc, y, **fit_kwargs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xc = _prepare_features(X)
        return self._model.predict_proba(Xc)[:, 1]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path / "catboost_model.cbm"))
        joblib.dump(self._feature_names, path / "catboost_features.joblib")

    def load(self, path: Path) -> None:
        self._model = cb.CatBoostClassifier()
        self._model.load_model(str(path / "catboost_model.cbm"))
        self._feature_names = joblib.load(path / "catboost_features.joblib")

    @property
    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {}
        return dict(zip(self._feature_names, self._model.get_feature_importance()))


class GBMEnsemble(BasePredictor):
    """Ensemble that averages XGBoost + LightGBM + CatBoost probabilities."""

    def __init__(self) -> None:
        self.models: list[BasePredictor] = [
            XGBoostModel(),
            LightGBMModel(),
            CatBoostModel(),
        ]

    @property
    def name(self) -> str:
        return "gbm_ensemble"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple] | None = None,
    ) -> None:
        for model in self.models:
            log.info("training_gbm_component", model=model.name)
            model.fit(X, y, eval_set=eval_set)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict_proba(X) for m in self.models])
        return preds.mean(axis=1)

    def save(self, path: Path) -> None:
        for model in self.models:
            model.save(path / model.name)

    def load(self, path: Path) -> None:
        for model in self.models:
            model.load(path / model.name)

    @property
    def feature_importance(self) -> dict[str, float]:
        """Average feature importance across all 3 component models."""
        all_imp: dict[str, list[float]] = {}
        for model in self.models:
            for feat, imp in model.feature_importance.items():
                all_imp.setdefault(feat, []).append(imp)
        return {f: float(np.mean(v)) for f, v in all_imp.items()}
