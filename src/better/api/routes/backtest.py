"""Backtest endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from better.api.schemas import BacktestSummaryResponse, BankrollPointResponse
from better.api.services import get_prediction_service

router = APIRouter()


@router.get("/backtest/summary", response_model=BacktestSummaryResponse)
def get_backtest_summary(
    edge_threshold: float = Query(0.03, description="Minimum edge to place a bet"),
    kelly_fraction: float = Query(0.25, description="Fraction of Kelly criterion"),
    model: str = Query("meta_learner", description="Model to use"),
) -> dict:
    """Return backtest summary statistics."""
    svc = get_prediction_service()
    return svc.get_backtest_summary(edge_threshold, kelly_fraction, model)


@router.get("/backtest/bankroll-curve", response_model=list[BankrollPointResponse])
def get_bankroll_curve(
    edge_threshold: float = Query(0.03, description="Minimum edge to place a bet"),
    kelly_fraction: float = Query(0.25, description="Fraction of Kelly criterion"),
    model: str = Query("meta_learner", description="Model to use"),
) -> list[dict]:
    """Return bankroll curve data for charting."""
    svc = get_prediction_service()
    return svc.get_backtest_bankroll_curve(edge_threshold, kelly_fraction, model)
