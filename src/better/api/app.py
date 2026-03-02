"""FastAPI application factory for the BETter prediction API.

Start with::

    uv run better api serve
    # or directly:
    uvicorn better.api.app:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from better.api.routes.backtest import router as backtest_router
from better.api.routes.bets import router as bets_router
from better.api.routes.games import router as games_router
from better.api.routes.health import router as health_router
from better.api.routes.models import router as models_router
from better.api.routes.predictions import router as predictions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    from better.api.services import get_prediction_service

    get_prediction_service()  # Triggers model loading
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="BETter MLB Prediction API",
        version="1.0.0",
        description="MLB game prediction and betting optimization API",
        lifespan=lifespan,
    )

    # CORS for local Streamlit/frontend access
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route groups
    application.include_router(health_router, tags=["Health"])
    application.include_router(games_router, prefix="/api", tags=["Games"])
    application.include_router(predictions_router, prefix="/api", tags=["Predictions"])
    application.include_router(bets_router, prefix="/api", tags=["Bets"])
    application.include_router(backtest_router, prefix="/api", tags=["Backtest"])
    application.include_router(models_router, prefix="/api", tags=["Models"])

    return application


app = create_app()
