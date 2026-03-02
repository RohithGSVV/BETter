"""Bet recommendations endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from better.api.schemas import BetRecommendationResponse
from better.api.services import get_prediction_service

router = APIRouter()


@router.get("/bets/recommendations", response_model=list[BetRecommendationResponse])
def get_bet_recommendations() -> list[dict]:
    """Return current bet recommendations (positive edge only)."""
    svc = get_prediction_service()
    return svc.get_bet_recommendations()
