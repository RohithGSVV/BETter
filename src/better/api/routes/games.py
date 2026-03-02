"""Today's games schedule endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from better.api.schemas import GameResponse
from better.api.services import get_prediction_service

router = APIRouter()


@router.get("/games/today", response_model=list[GameResponse])
def get_todays_games() -> list[dict]:
    """Return today's MLB schedule with probable pitchers."""
    svc = get_prediction_service()
    return svc.get_todays_schedule()
