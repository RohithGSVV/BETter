"""Pydantic response models for the BETter API."""

from __future__ import annotations

from pydantic import BaseModel


class GameResponse(BaseModel):
    """A single scheduled game."""

    game_pk: int
    game_date: str
    home_team: str
    away_team: str
    status: str = ""
    home_sp_name: str = ""
    away_sp_name: str = ""
    venue: str = ""


class PredictionResponse(BaseModel):
    """Model predictions for a single game."""

    game_pk: int
    game_date: str
    home_team: str
    away_team: str
    home_sp_name: str = ""
    away_sp_name: str = ""
    bayesian_prob: float | None = None
    gbm_prob: float | None = None
    monte_carlo_prob: float | None = None
    meta_prob: float | None = None
    consensus_prob: float | None = None
    market_implied_prob: float | None = None
    edge: float | None = None
    confidence: str | None = None


class BetRecommendationResponse(BaseModel):
    """A single bet recommendation."""

    game_pk: int | None = None
    game_date: str | None = None
    home_team: str
    away_team: str
    bet_side: str
    model_prob: float
    market_prob: float
    edge: float
    expected_value: float
    kelly_fraction: float
    bet_amount: float
    odds_american: int
    bookmaker: str
    confidence: str


class BacktestSummaryResponse(BaseModel):
    """Summary statistics from a backtest run."""

    initial_bankroll: float
    final_bankroll: float
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    roi_pct: float
    yield_pct: float
    max_drawdown_pct: float
    avg_edge: float
    sharpe_ratio: float
    longest_losing_streak: int
    edge_threshold: float
    kelly_fraction: float
    model_used: str


class BankrollPointResponse(BaseModel):
    """A single point on the bankroll curve."""

    game_date: str | None = None
    season: int | None = None
    bankroll: float


class ModelStatusResponse(BaseModel):
    """Status of loaded models."""

    models_loaded: dict[str, bool]
    last_training_date: str | None = None
    model_details: dict[str, dict] = {}


class CalibrationBinResponse(BaseModel):
    """A single calibration bin."""

    prob_bin: str
    count: int
    actual_win_rate: float
    predicted_prob: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: bool
    timestamp: str
