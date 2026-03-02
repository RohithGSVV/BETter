"""Predictions endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from better.api.schemas import PredictionResponse
from better.api.services import get_prediction_service

router = APIRouter()


@router.get("/predictions/today", response_model=list[PredictionResponse])
def get_todays_predictions() -> list[dict]:
    """Return model predictions for today's games."""
    svc = get_prediction_service()
    return svc.get_todays_predictions()
