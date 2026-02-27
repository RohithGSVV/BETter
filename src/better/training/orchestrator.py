"""Training orchestrator: full pipeline from features to trained meta-learner.

Pipeline steps:
  1. Load/build training DataFrame
  2. Generate walk-forward folds
  3. (Optional) Optuna-tune GBM hyperparameters
  4. For each fold — train base models, collect OOF predictions, evaluate
  5. Assemble OOF predictions across all folds
  6. Train meta-learner on assembled OOF predictions (leakage-free)
  7. Train final production models on ALL data
  8. Save everything
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from better.config import settings
from better.features.pipeline import build_training_set
from better.models import TARGET_COL
from better.models.bayesian.kalman import BayesianKalmanModel
from better.models.gbm.ensemble import GBMEnsemble
from better.models.meta.stacker import MetaLearner
from better.models.monte_carlo.simulator import MonteCarloSimulator
from better.training.evaluation import evaluate_predictions, summarize_results
from better.training.persistence import save_model, save_oof_predictions
from better.training.splits import generate_walk_forward_splits
from better.utils.logging import get_logger

log = get_logger(__name__)


def run_full_training_pipeline(
    skip_tuning: bool = False,
    n_tuning_trials: int = 50,
) -> pd.DataFrame:
    """Execute the complete training pipeline.

    Returns a summary DataFrame of evaluation results.
    """
    # 1. Load training data
    log.info("loading_training_data")
    df = build_training_set()
    log.info("training_data_loaded", rows=len(df), columns=len(df.columns))

    # 2. Generate walk-forward folds
    folds = generate_walk_forward_splits(df)
    log.info("walk_forward_folds", n_folds=len(folds))

    # 3. Optional: tune GBM hyperparameters
    if not skip_tuning:
        from better.models.gbm.tuning import tune_model

        for model_name in ["xgboost", "lightgbm", "catboost"]:
            log.info("tuning_model", model=model_name)
            tune_model(model_name, df, folds, n_trials=n_tuning_trials)

    # 4. Walk-forward: train base models, collect OOF predictions
    base_model_names = ["gbm_ensemble", "bayesian_kalman", "monte_carlo"]
    oof_predictions: dict[str, list[np.ndarray]] = {n: [] for n in base_model_names}
    oof_y_true: list[np.ndarray] = []
    oof_X_raw: list[pd.DataFrame] = []
    all_results = []

    for fold in folds:
        X_train = df.loc[fold.train_idx]
        y_train = df.loc[fold.train_idx, TARGET_COL].astype(int)
        X_test = df.loc[fold.test_idx]
        y_test = df.loc[fold.test_idx, TARGET_COL].astype(int)

        log.info(
            "training_fold",
            test_year=fold.test_year,
            train_size=len(X_train),
            test_size=len(X_test),
        )

        # --- GBM Ensemble ---
        gbm = GBMEnsemble()
        # Use last 15% of training data (by date) as early-stopping eval set
        cutoff = int(len(X_train) * 0.85)
        X_tr, X_val = X_train.iloc[:cutoff], X_train.iloc[cutoff:]
        y_tr, y_val = y_train.iloc[:cutoff], y_train.iloc[cutoff:]
        gbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        gbm_preds = gbm.predict_proba(X_test)
        oof_predictions["gbm_ensemble"].append(gbm_preds)
        all_results.append(
            evaluate_predictions(y_test, gbm_preds, "gbm_ensemble", fold.test_year)
        )
        save_model(gbm, fold.test_year)

        # --- Bayesian Kalman ---
        bayesian = BayesianKalmanModel()
        bayesian.fit(X_train, y_train)
        bayesian_preds = bayesian.predict_proba(X_test)
        oof_predictions["bayesian_kalman"].append(bayesian_preds)
        all_results.append(
            evaluate_predictions(
                y_test, bayesian_preds, "bayesian_kalman", fold.test_year
            )
        )
        save_model(bayesian, fold.test_year)

        # --- Monte Carlo ---
        mc = MonteCarloSimulator()
        mc.fit(X_train, y_train)
        mc_preds = mc.predict_proba(X_test)
        oof_predictions["monte_carlo"].append(mc_preds)
        all_results.append(
            evaluate_predictions(y_test, mc_preds, "monte_carlo", fold.test_year)
        )
        save_model(mc, fold.test_year)

        oof_y_true.append(y_test.values)
        oof_X_raw.append(X_test)

    # 5. Assemble OOF predictions for meta-learner training
    all_oof_y = np.concatenate(oof_y_true)
    all_oof_X = pd.concat(oof_X_raw, ignore_index=True)
    all_oof_preds = {
        name: np.concatenate(preds) for name, preds in oof_predictions.items()
    }

    # Save OOF predictions for analysis
    save_oof_predictions(
        {
            name: dict(zip([f.test_year for f in folds], preds))
            for name, preds in oof_predictions.items()
        }
    )

    # 6. Train meta-learner on OOF predictions
    log.info("training_meta_learner")
    meta = MetaLearner(include_raw_features=True)
    meta.fit_from_base_predictions(all_oof_preds, pd.Series(all_oof_y), all_oof_X)

    meta_preds = meta.predict_from_base_predictions(all_oof_preds, all_oof_X)
    all_results.append(evaluate_predictions(all_oof_y, meta_preds, "meta_learner"))
    save_model(meta)

    # 7. Train final production models on ALL data
    log.info("training_final_models")
    y_all = df[TARGET_COL].astype(int)

    final_gbm = GBMEnsemble()
    final_gbm.fit(df, y_all)
    save_model(final_gbm)

    final_bayesian = BayesianKalmanModel()
    final_bayesian.fit(df, y_all)
    save_model(final_bayesian)

    final_mc = MonteCarloSimulator()
    final_mc.fit(df, y_all)
    save_model(final_mc)

    # 8. Summary + persist results to git-tracked results/ directory
    summary = summarize_results(all_results)

    results_dir = settings.project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Per-fold detail (every model × every test year)
    fold_detail = pd.DataFrame([
        {
            "model": r.model_name,
            "fold_year": r.fold_year,
            "log_loss": round(r.log_loss, 5),
            "brier_score": round(r.brier_score, 5),
            "accuracy": round(r.accuracy, 5),
            "n_games": r.n_games,
        }
        for r in all_results
    ])
    fold_detail.to_csv(results_dir / "fold_results.csv", index=False)

    # Aggregate summary
    summary.to_csv(results_dir / "summary.csv", index=False)

    log.info(
        "training_complete",
        results_saved=str(results_dir),
        tuning=not skip_tuning,
    )
    return summary
