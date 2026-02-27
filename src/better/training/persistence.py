"""Model persistence utilities.

Directory structure::

    models/{model_name}/fold_{year}/   — per-fold checkpoints
    models/{model_name}/final/         — production model trained on all data
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from better.config import settings
from better.utils.logging import get_logger

if TYPE_CHECKING:
    from better.models import BasePredictor

log = get_logger(__name__)


def model_dir(model_name: str, fold_year: int | None = None) -> Path:
    """Return the directory for a model's artifacts."""
    base = settings.models_dir / model_name
    if fold_year is not None:
        return base / f"fold_{fold_year}"
    return base / "final"


def save_model(model: BasePredictor, fold_year: int | None = None) -> Path:
    """Save a model to the standard directory structure."""
    path = model_dir(model.name, fold_year)
    model.save(path)
    log.info("model_saved", model=model.name, path=str(path))
    return path


def load_model(model: BasePredictor, fold_year: int | None = None) -> None:
    """Load a model from the standard directory structure."""
    path = model_dir(model.name, fold_year)
    model.load(path)
    log.info("model_loaded", model=model.name, path=str(path))


def save_oof_predictions(
    predictions: dict[str, dict[int, np.ndarray]],
    path: Path | None = None,
) -> None:
    """Save out-of-fold predictions for all models."""
    path = path or (settings.models_dir / "oof_predictions.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictions, path)
    log.info("oof_predictions_saved", path=str(path))


def load_oof_predictions(
    path: Path | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Load out-of-fold predictions."""
    path = path or (settings.models_dir / "oof_predictions.joblib")
    return joblib.load(path)
