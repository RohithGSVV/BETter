"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from better.models import EvaluationResult
from better.training.evaluation import (
    compute_calibration_curve,
    evaluate_predictions,
    summarize_results,
)


class TestEvaluatePredictions:
    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.99, 0.01, 0.99, 0.01, 0.99])
        result = evaluate_predictions(y_true, y_prob, "perfect")
        assert result.accuracy == 1.0
        assert result.brier_score < 0.01
        assert result.log_loss < 0.1

    def test_coin_flip_baseline(self):
        rng = np.random.default_rng(42)
        n = 10000
        y_true = rng.integers(0, 2, n)
        y_prob = np.full(n, 0.5)
        result = evaluate_predictions(y_true, y_prob, "coin_flip")
        assert result.accuracy == pytest.approx(0.5, abs=0.05)
        assert result.brier_score == pytest.approx(0.25, abs=0.01)
        assert result.log_loss == pytest.approx(0.693, abs=0.01)

    def test_result_fields(self):
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.8, 0.2, 0.7])
        result = evaluate_predictions(y_true, y_prob, "test_model", fold_year=2023)
        assert result.model_name == "test_model"
        assert result.fold_year == 2023
        assert result.n_games == 3


class TestCalibrationCurve:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 1000)
        y_prob = rng.uniform(0, 1, 1000)
        fraction_pos, mean_pred = compute_calibration_curve(y_true, y_prob, n_bins=5)
        assert len(fraction_pos) <= 5
        assert len(mean_pred) <= 5


class TestSummarizeResults:
    def test_groups_by_model(self):
        results = [
            EvaluationResult("A", 0.68, 0.24, 0.56, 1000, 2020),
            EvaluationResult("A", 0.67, 0.23, 0.57, 1000, 2021),
            EvaluationResult("B", 0.70, 0.25, 0.54, 1000, 2020),
            EvaluationResult("B", 0.69, 0.24, 0.55, 1000, 2021),
        ]
        summary = summarize_results(results)
        assert len(summary) == 2
        assert set(summary["model"]) == {"A", "B"}

    def test_aggregation_correct(self):
        results = [
            EvaluationResult("X", 0.60, 0.20, 0.60, 500, 2020),
            EvaluationResult("X", 0.70, 0.30, 0.50, 500, 2021),
        ]
        summary = summarize_results(results)
        row = summary[summary["model"] == "X"].iloc[0]
        assert row["avg_log_loss"] == pytest.approx(0.65)
        assert row["avg_brier"] == pytest.approx(0.25)
        assert row["avg_accuracy"] == pytest.approx(0.55)
        assert row["total_games"] == 1000
