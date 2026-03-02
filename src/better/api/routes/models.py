"""Model status and edge analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from better.api.schemas import ModelStatusResponse
from better.api.services import get_prediction_service

router = APIRouter()


@router.get("/models/status", response_model=ModelStatusResponse)
def get_model_status() -> dict:
    """Return status of loaded prediction models."""
    svc = get_prediction_service()
    return svc.get_model_status()


@router.get("/edge/calibration")
def get_calibration(
    model: str = Query("meta_learner", description="Model to analyze"),
) -> dict:
    """Return calibration and edge analysis data."""
    svc = get_prediction_service()
    return svc.get_calibration_data(model)
