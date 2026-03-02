"""Health check endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from better.api.schemas import HealthResponse
from better.api.services import get_prediction_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> dict:
    """Return API health status."""
    svc = get_prediction_service()
    return {
        "status": "ok",
        "models_loaded": svc._models_loaded,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
