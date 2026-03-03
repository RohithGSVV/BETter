"""FastAPI application factory for the BETter prediction API.

Start with::

    uv run better api serve
    # or directly:
    uvicorn better.api.app:app --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from better.api.routes.backtest import router as backtest_router
from better.api.routes.bets import router as bets_router
from better.api.routes.games import router as games_router
from better.api.routes.health import router as health_router
from better.api.routes.models import router as models_router
from better.api.routes.predictions import router as predictions_router
from better.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, start Kalshi feed if credentials are set."""
    from better.api.services import get_prediction_service

    get_prediction_service()  # Triggers model loading

    # Start Kalshi WebSocket feed (no-op if credentials not configured)
    kalshi_task = None
    if settings.kalshi_email:
        from better.data.live.kalshi_feed import start_kalshi_feed

        feed = await start_kalshi_feed()
        if feed is not None:
            app.state.kalshi_feed = feed
            kalshi_task = asyncio.create_task(feed.run())

    yield

    # Shutdown: stop Kalshi feed gracefully
    if kalshi_task is not None and not kalshi_task.done():
        if hasattr(app.state, "kalshi_feed"):
            app.state.kalshi_feed.stop()
        kalshi_task.cancel()
        try:
            await kalshi_task
        except asyncio.CancelledError:
            pass


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
