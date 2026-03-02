"""Tests for FastAPI endpoints using TestClient."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_service():
    """Create a mock PredictionService."""
    svc = MagicMock()
    svc._models_loaded = True

    svc.get_todays_schedule.return_value = [
        {
            "game_pk": 717001,
            "game_date": "2024-07-15",
            "home_team": "NYY",
            "away_team": "BOS",
            "status": "Scheduled",
            "home_sp_name": "Gerrit Cole",
            "away_sp_name": "Brayan Bello",
            "venue": "Yankee Stadium",
        }
    ]

    svc.get_todays_predictions.return_value = [
        {
            "game_pk": 717001,
            "game_date": "2024-07-15",
            "home_team": "NYY",
            "away_team": "BOS",
            "home_sp_name": "Gerrit Cole",
            "away_sp_name": "Brayan Bello",
            "bayesian_prob": 0.58,
            "gbm_prob": None,
            "monte_carlo_prob": 0.55,
            "meta_prob": None,
            "consensus_prob": 0.565,
            "market_implied_prob": 0.52,
            "edge": 0.045,
            "confidence": "medium",
        }
    ]

    svc.get_bet_recommendations.return_value = [
        {
            "game_pk": 717001,
            "game_date": "2024-07-15",
            "home_team": "NYY",
            "away_team": "BOS",
            "bet_side": "HOME",
            "model_prob": 0.565,
            "market_prob": 0.52,
            "edge": 0.045,
            "expected_value": 0.05,
            "kelly_fraction": 0.02,
            "bet_amount": 20.0,
            "odds_american": -120,
            "bookmaker": "consensus",
            "confidence": "medium",
        }
    ]

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

    svc.get_backtest_summary.return_value = {
        "initial_bankroll": 1000.0,
        "final_bankroll": 1050.0,
        "total_bets": 100,
        "wins": 55,
        "losses": 45,
        "win_rate": 0.55,
        "roi_pct": 5.0,
        "yield_pct": 3.0,
        "max_drawdown_pct": 8.0,
        "avg_edge": 0.05,
        "sharpe_ratio": 1.2,
        "longest_losing_streak": 5,
        "edge_threshold": 0.03,
        "kelly_fraction": 0.25,
        "model_used": "meta_learner",
    }

    svc.get_backtest_bankroll_curve.return_value = [
        {"game_date": "2024-01-01", "season": 2024, "bankroll": 1000.0},
        {"game_date": "2024-06-01", "season": 2024, "bankroll": 1025.0},
    ]

    svc.get_model_status.return_value = {
        "models_loaded": {
            "gbm_ensemble": True,
            "bayesian_kalman": True,
            "monte_carlo": True,
            "meta_learner": True,
        },
        "last_training_date": "2024-07-15 10:00 UTC",
        "model_details": {},
    }

    svc.get_calibration_data.return_value = {
        "calibration": [{"prob_bin": "0.5-0.6", "count": 100, "actual_win_rate": 0.55, "predicted_prob": 0.55}],
        "model_comparison": [],
    }

    return svc


@pytest.fixture
def client(mock_service):
    """Create a TestClient with mocked PredictionService."""
    with patch("better.api.services.get_prediction_service", return_value=mock_service):
        from better.api.app import create_app

        app = create_app()
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "models_loaded" in data

    def test_health_shows_models_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["models_loaded"] is True


class TestGamesEndpoint:
    def test_todays_games_returns_list(self, client):
        response = client.get("/api/games/today")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_game_has_expected_fields(self, client):
        response = client.get("/api/games/today")
        game = response.json()[0]
        assert game["game_pk"] == 717001
        assert game["home_team"] == "NYY"
        assert game["away_team"] == "BOS"
        assert game["home_sp_name"] == "Gerrit Cole"


class TestPredictionsEndpoint:
    def test_predictions_returns_list(self, client):
        response = client.get("/api/predictions/today")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_prediction_has_model_probs(self, client):
        response = client.get("/api/predictions/today")
        pred = response.json()[0]
        assert pred["bayesian_prob"] == 0.58
        assert pred["monte_carlo_prob"] == 0.55
        assert pred["edge"] == 0.045
        assert pred["confidence"] == "medium"


class TestBetsEndpoint:
    def test_recommendations_returns_list(self, client):
        response = client.get("/api/bets/recommendations")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_recommendation_has_kelly_sizing(self, client):
        response = client.get("/api/bets/recommendations")
        rec = response.json()[0]
        assert rec["bet_side"] == "HOME"
        assert rec["edge"] == 0.045
        assert rec["kelly_fraction"] == 0.02
        assert rec["bet_amount"] == 20.0


class TestBacktestEndpoint:
    def test_backtest_summary_returns_stats(self, client):
        response = client.get("/api/backtest/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_bets"] == 100
        assert data["win_rate"] == 0.55
        assert data["model_used"] == "meta_learner"

    def test_bankroll_curve_returns_list(self, client):
        response = client.get("/api/backtest/bankroll-curve")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["bankroll"] == 1000.0

    def test_backtest_accepts_query_params(self, client):
        response = client.get(
            "/api/backtest/summary?edge_threshold=0.05&kelly_fraction=0.5&model=gbm_ensemble"
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_bets" in data


class TestModelsEndpoint:
    def test_model_status_returns_loaded_dict(self, client):
        response = client.get("/api/models/status")
        assert response.status_code == 200
        data = response.json()
        assert data["models_loaded"]["gbm_ensemble"] is True
        assert data["models_loaded"]["meta_learner"] is True

    def test_calibration_returns_data(self, client):
        response = client.get("/api/edge/calibration?model=meta_learner")
        assert response.status_code == 200
        data = response.json()
        assert "calibration" in data
