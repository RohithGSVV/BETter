"""Live game prediction endpoints — real-time win probabilities and edges."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/live/games")
async def get_live_games():
    """Get all active live game snapshots with predictions and edges."""
    from better.api.services import get_prediction_service

    svc = get_prediction_service()
    manager = svc.get_live_manager()

    if manager is None:
        return {"games": [], "status": "offline", "message": "Live tracking not started"}

    snapshots = manager.get_all_snapshots()
    return {
        "games": [s.to_dict() for s in snapshots],
        "status": "live",
        "active_count": len([s for s in snapshots if s.status == "live"]),
    }


@router.get("/live/games/{game_pk}")
async def get_live_game(game_pk: int):
    """Get the live snapshot for a specific game."""
    from better.api.services import get_prediction_service

    svc = get_prediction_service()
    manager = svc.get_live_manager()

    if manager is None:
        raise HTTPException(status_code=503, detail="Live tracking not started")

    snap = manager.get_snapshot(game_pk)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"Game {game_pk} not being tracked")

    return snap.to_dict()
