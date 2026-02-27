"""Base model interface and shared types for the BETter prediction system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ModelPrediction:
    """Container for a single model's predictions on a dataset."""

    model_name: str
    probabilities: np.ndarray
    game_pks: np.ndarray | None = None


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""

    model_name: str
    log_loss: float
    brier_score: float
    accuracy: float
    n_games: int
    fold_year: int | None = None


class BasePredictor(ABC):
    """Abstract base class for all pregame prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier (e.g., 'xgboost', 'bayesian_kalman')."""
        ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on features X and binary target y."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(home_win) for each row in X."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artifacts to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model artifacts from disk."""
        ...


# Columns excluded from model features
IDENTIFIER_COLS = ["game_pk", "game_date", "season", "home_team", "away_team"]
TARGET_COL = "home_win"
DROP_COLS = IDENTIFIER_COLS + [TARGET_COL, "home_sp_throws", "away_sp_throws"]
