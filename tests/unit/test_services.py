"""Tests for PredictionService logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPredictionServiceInit:
    """Test PredictionService initialization and model loading."""

    def test_service_starts_unloaded(self):
        from better.api.services import PredictionService

        svc = PredictionService()
        assert svc._models_loaded is False
        assert svc._gbm is None
        assert svc._bayesian is None
        assert svc._mc is None
        assert svc._meta is None

    @patch("better.api.services.PredictionService.load_models")
    def test_get_prediction_service_singleton(self, mock_load):
        """Singleton returns the same instance."""
        import better.api.services as mod

        mod._service = None  # Reset singleton
        mock_load.return_value = {}

        svc1 = mod.get_prediction_service()
        svc2 = mod.get_prediction_service()
        assert svc1 is svc2

        # Clean up
        mod._service = None

    def test_load_models_handles_missing_files(self):
        """load_models() should not raise even if model files don't exist."""
        from better.api.services import PredictionService

        svc = PredictionService()
        results = svc.load_models()

        # All should fail gracefully (no model files in test env)
        assert isinstance(results, dict)
        assert "gbm_ensemble" in results
        assert "bayesian_kalman" in results
        assert "monte_carlo" in results
        assert "meta_learner" in results


class TestPredictionServiceModelStatus:
    """Test model status reporting."""

    def test_model_status_structure(self):
        from better.api.services import PredictionService

        svc = PredictionService()
        status = svc.get_model_status()

        assert "models_loaded" in status
        assert isinstance(status["models_loaded"], dict)
        assert "gbm_ensemble" in status["models_loaded"]
        assert "model_details" in status

    def test_model_status_all_unloaded(self):
        from better.api.services import PredictionService

        svc = PredictionService()
        status = svc.get_model_status()

        for name, loaded in status["models_loaded"].items():
            assert loaded is False


class TestPredictionServicePredictions:
    """Test prediction generation."""

    def test_predict_game_no_models(self):
        """predict_game with no models should return empty dict."""
        from better.api.services import PredictionService

        svc = PredictionService()
        preds = svc.predict_game("NYY", "BOS")
        assert preds == {}

    def test_predict_game_with_mock_bayesian(self):
        """predict_game with mocked Bayesian model."""
        from better.api.services import PredictionService

        svc = PredictionService()
        mock_bayesian = MagicMock()
        mock_bayesian._predict_game.return_value = 0.58
        svc._bayesian = mock_bayesian

        preds = svc.predict_game("NYY", "BOS")
        assert "bayesian_kalman" in preds
        assert preds["bayesian_kalman"] == 0.58
        assert "consensus" in preds

    def test_predict_game_consensus_average(self):
        """Consensus should average all available model predictions."""
        from better.api.services import PredictionService

        svc = PredictionService()

        mock_bayesian = MagicMock()
        mock_bayesian._predict_game.return_value = 0.60

        mock_mc = MagicMock()
        mock_mc._expected_runs.return_value = 4.5
        mock_mc._simulate_runs.return_value = np.array([3, 4, 5, 6, 7])
        mock_mc.params = MagicMock()
        mock_mc.params.n_sims = 5

        svc._bayesian = mock_bayesian
        svc._mc = mock_mc

        preds = svc.predict_game("NYY", "BOS")
        assert "consensus" in preds
        # Consensus should be average of available models
        non_consensus = {k: v for k, v in preds.items() if k != "consensus"}
        expected = round(sum(non_consensus.values()) / len(non_consensus), 4)
        assert preds["consensus"] == expected

    def test_todays_predictions_empty_schedule(self):
        """get_todays_predictions returns empty list when no games."""
        from better.api.services import PredictionService

        svc = PredictionService()
        with patch.object(svc, "get_todays_schedule", return_value=[]):
            preds = svc.get_todays_predictions()
            assert preds == []


class TestPredictionServiceBacktest:
    """Test backtest methods."""

    def test_backtest_summary_structure(self):
        """Backtest summary should have expected keys."""
        from better.api.services import PredictionService

        svc = PredictionService()

        # Mock the backtester to avoid needing OOF data
        mock_stats = MagicMock()
        mock_stats.initial_bankroll = 1000.0
        mock_stats.current_bankroll = 1050.0
        mock_stats.total_bets = 100
        mock_stats.bets_won = 55
        mock_stats.bets_lost = 45
        mock_stats.win_rate = 0.55
        mock_stats.roi_pct = 5.0
        mock_stats.yield_pct = 3.0
        mock_stats.max_drawdown_pct = 8.0
        mock_stats.avg_edge = 0.05
        mock_stats.sharpe_ratio = 1.2
        mock_stats.longest_losing_streak = 5

        mock_result = MagicMock()
        mock_result.stats = mock_stats

        with patch.object(svc, "_run_backtest", return_value=mock_result):
            summary = svc.get_backtest_summary()

        assert summary["total_bets"] == 100
        assert summary["win_rate"] == 0.55
        assert summary["model_used"] == "meta_learner"
        assert "initial_bankroll" in summary
        assert "yield_pct" in summary

    def test_backtest_bankroll_curve_empty(self):
        """Bankroll curve with empty DataFrame."""
        import pandas as pd

        from better.api.services import PredictionService

        svc = PredictionService()
        mock_result = MagicMock()
        mock_result.bankroll_curve = pd.DataFrame()

        with patch.object(svc, "_run_backtest", return_value=mock_result):
            curve = svc.get_backtest_bankroll_curve()
            assert curve == []


class TestPredictionServiceCalibration:
    """Test calibration data retrieval."""

    def test_calibration_missing_oof(self, tmp_path):
        """Should return error dict when OOF file doesn't exist."""
        from better.api.services import PredictionService

        svc = PredictionService()
        with patch("better.api.services.settings") as mock_settings:
            mock_settings.project_root = tmp_path
            result = svc.get_calibration_data()
            assert "error" in result


class TestPredictionServiceCache:
    """Test cache management."""

    def test_refresh_clears_caches(self):
        from better.api.services import PredictionService

        svc = PredictionService()
        svc._cached_predictions = [{"test": True}]
        svc._cached_schedule = [{"test": True}]

        svc.refresh_predictions()

        assert svc._cached_predictions is None
        assert svc._cached_schedule is None
        assert svc._cache_date is None
