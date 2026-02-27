"""Optuna-based hyperparameter tuning for GBM models.

Each trial proposes hyperparameters, trains on walk-forward training
folds, and evaluates log-loss on test folds.  Optuna minimizes the
average log-loss across all walk-forward folds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from better.models.gbm.ensemble import CatBoostModel, LightGBMModel, XGBoostModel
from better.training.splits import WalkForwardFold
from better.utils.logging import get_logger

log = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _xgboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }


def _lightgbm_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def _catboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }


_SEARCH_SPACES: dict[str, tuple] = {
    "xgboost": (_xgboost_search_space, XGBoostModel),
    "lightgbm": (_lightgbm_search_space, LightGBMModel),
    "catboost": (_catboost_search_space, CatBoostModel),
}


def tune_model(
    model_name: str,
    df: pd.DataFrame,
    folds: list[WalkForwardFold],
    n_trials: int = 50,
) -> dict[str, Any]:
    """Run Optuna tuning for a single GBM model.

    Returns the best hyperparameters found.
    """
    space_fn, model_cls = _SEARCH_SPACES[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        fold_losses: list[float] = []

        for fold in folds:
            X_train = df.loc[fold.train_idx]
            y_train = df.loc[fold.train_idx, "home_win"].astype(int)
            X_test = df.loc[fold.test_idx]
            y_test = df.loc[fold.test_idx, "home_win"].astype(int)

            model = model_cls(params=params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            preds = model.predict_proba(X_test)
            fold_losses.append(log_loss(y_test, preds))

        return float(np.mean(fold_losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(
        "tuning_complete",
        model=model_name,
        best_loss=round(study.best_value, 5),
        best_params=study.best_params,
    )
    return study.best_params
