"""Model evaluation metrics and calibration analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from better.models import EvaluationResult
from better.utils.logging import get_logger

log = get_logger(__name__)


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    model_name: str,
    fold_year: int | None = None,
) -> EvaluationResult:
    """Compute standard evaluation metrics for binary probability predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_binary = (y_prob >= 0.5).astype(int)

    result = EvaluationResult(
        model_name=model_name,
        log_loss=float(log_loss(y_true, y_prob)),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred_binary)),
        n_games=len(y_true),
        fold_year=fold_year,
    )

    log.info(
        "model_evaluation",
        model=model_name,
        fold_year=fold_year,
        log_loss=round(result.log_loss, 4),
        brier_score=round(result.brier_score, 4),
        accuracy=round(result.accuracy, 4),
        n_games=result.n_games,
    )
    return result


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve (fraction of positives vs mean predicted)."""
    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return fraction_pos, mean_predicted


def summarize_results(results: list[EvaluationResult]) -> pd.DataFrame:
    """Aggregate evaluation results per model across folds."""
    rows = [
        {
            "model": r.model_name,
            "fold_year": r.fold_year,
            "log_loss": r.log_loss,
            "brier_score": r.brier_score,
            "accuracy": r.accuracy,
            "n_games": r.n_games,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)

    summary = (
        df.groupby("model")
        .agg(
            avg_log_loss=("log_loss", "mean"),
            avg_brier=("brier_score", "mean"),
            avg_accuracy=("accuracy", "mean"),
            total_games=("n_games", "sum"),
        )
        .reset_index()
    )
    return summary
