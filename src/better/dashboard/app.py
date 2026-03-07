"""BETter MLB Prediction Dashboard — NiceGUI app.

Run with::

    uv run better dashboard
    # or directly:
    python -m better.dashboard.app
"""

from __future__ import annotations

import asyncio

from nicegui import app, ui

from better.api.services import PredictionService
from better.utils.logging import get_logger

log = get_logger(__name__)

# ── Shared service (loaded once) ────────────────────────────────────────

_svc: PredictionService | None = None
_live_task: asyncio.Task | None = None


def get_service() -> PredictionService:
    global _svc
    if _svc is None:
        _svc = PredictionService()
        _svc.load_models()
    return _svc


async def _start_live_tracking() -> None:
    """Start the LiveGameManager on dashboard startup."""
    global _live_task
    try:
        svc = get_service()
        manager = svc.get_live_manager()

        schedule = svc.get_todays_schedule()
        if not schedule:
            log.info("dashboard_live_no_games", msg="No games on schedule today")
            return

        # Build pre-game probabilities
        pregame_probs: dict[int, float] = {}
        for game in schedule:
            gpk = game.get("game_pk")
            if gpk:
                preds = svc.predict_game(
                    game.get("home_team", ""),
                    game.get("away_team", ""),
                )
                best = (
                    preds.get("meta_learner")
                    or preds.get("consensus")
                    or preds.get("bayesian_kalman")
                )
                if best:
                    pregame_probs[gpk] = best

        _live_task = asyncio.create_task(
            manager.start(schedule, pregame_probs=pregame_probs, poll_interval=3.0)
        )
        log.info(
            "dashboard_live_started",
            games=len(schedule),
            pregame_probs=len(pregame_probs),
        )
    except Exception as exc:
        log.warning("dashboard_live_start_failed", error=str(exc))


def _stop_live_tracking() -> None:
    """Stop the LiveGameManager on dashboard shutdown."""
    global _live_task
    try:
        if _svc is not None:
            _svc.get_live_manager().stop()
    except Exception:
        pass
    if _live_task is not None and not _live_task.done():
        _live_task.cancel()


app.on_startup(_start_live_tracking)
app.on_shutdown(_stop_live_tracking)


# ── Theme ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* Global background */
    body { background: #0a0f1a !important; }

    /* Gradient header */
    .better-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        border-bottom: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Sidebar */
    .better-sidebar {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Glow cards */
    .glow-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 12px;
        transition: all 0.2s ease;
    }
    .glow-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.08);
    }

    /* Metric cards with colored top border */
    .metric-blue { border-top: 3px solid #3b82f6; }
    .metric-green { border-top: 3px solid #22c55e; }
    .metric-amber { border-top: 3px solid #f59e0b; }
    .metric-red { border-top: 3px solid #ef4444; }
    .metric-purple { border-top: 3px solid #a855f7; }

    /* Active nav button glow */
    .nav-active {
        background: linear-gradient(90deg, rgba(59,130,246,0.2) 0%, rgba(59,130,246,0.05) 100%) !important;
        border-left: 3px solid #3b82f6 !important;
    }

    /* Better tables */
    .q-table thead th {
        background: rgba(30, 41, 59, 0.8) !important;
        color: rgba(148, 163, 184, 1) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.05em !important;
    }
    .q-table tbody td {
        border-color: rgba(255,255,255,0.04) !important;
    }
    .q-table tbody tr:hover td {
        background: rgba(59, 130, 246, 0.05) !important;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    /* Schedule table inside cards — no extra border */
    .schedule-table .q-table__top,
    .schedule-table .q-table__bottom {
        padding: 0 !important;
    }
    .schedule-table {
        border: none !important;
    }

    /* Alternating row tint for tables */
    .q-table tbody tr:nth-child(even) td {
        background: rgba(255,255,255,0.015) !important;
    }
</style>
"""


def apply_theme() -> None:
    """Apply consistent dark theme styling."""
    ui.colors(primary="#3b82f6", secondary="#1e40af", accent="#60a5fa")
    ui.dark_mode(True)
    ui.add_head_html(CUSTOM_CSS)


# ── Layout shell ────────────────────────────────────────────────────────


@ui.page("/")
def index():
    apply_theme()
    _build_page("games")


@ui.page("/live")
def live_page():
    apply_theme()
    _build_page("live")


@ui.page("/game/{game_pk}")
def game_detail_page(game_pk: int):
    apply_theme()
    _build_page("game_detail", game_pk=game_pk)


@ui.page("/schedule")
def schedule_page():
    apply_theme()
    _build_page("schedule")


@ui.page("/backtest")
def backtest_page():
    apply_theme()
    _build_page("backtest")


@ui.page("/edge")
def edge_page():
    apply_theme()
    _build_page("edge")


@ui.page("/models")
def models_page():
    apply_theme()
    _build_page("models")


def _build_page(active: str, **kwargs) -> None:
    """Build the page shell with header, sidebar, and content area."""

    # Header
    with ui.header().classes("items-center justify-between px-6 better-header"):
        with ui.row().classes("items-center gap-3"):
            # Baseball emoji as logo
            ui.html('<span style="font-size:1.6rem">&#9918;</span>')
            with ui.column().classes("gap-0"):
                ui.label("BETter").classes(
                    "text-xl font-bold tracking-wide"
                ).style("background: linear-gradient(90deg, #60a5fa, #a78bfa); "
                        "-webkit-background-clip: text; -webkit-text-fill-color: transparent;")
                ui.label("MLB Prediction System").classes("text-[0.65rem] text-gray-500 -mt-1")
        with ui.row().classes("items-center gap-2"):
            ui.html('<span style="font-size:0.8rem; color: #22c55e;">&#9679;</span>')
            ui.label("Models Active").classes("text-xs text-gray-400")

    # Left drawer / sidebar
    with ui.left_drawer(value=True).classes("better-sidebar p-4"):
        ui.label("MENU").classes(
            "text-[0.65rem] uppercase tracking-[0.2em] text-gray-500 mb-4 ml-1"
        )

        nav_items = [
            ("Dashboard", "/", "dashboard", "games"),
            ("Schedule", "/schedule", "event", "schedule"),
            ("Backtest", "/backtest", "show_chart", "backtest"),
            ("Models", "/models", "hub", "models"),
        ]

        for label, href, icon, key in nav_items:
            is_active = key == active
            btn_classes = "w-full justify-start rounded-lg mb-1 "
            if is_active:
                btn_classes += "nav-active text-blue-300"
            else:
                btn_classes += "text-gray-400 hover:text-gray-200 hover:bg-white/5"

            ui.button(
                label,
                icon=icon,
                on_click=lambda h=href: ui.navigate.to(h),
            ).props("flat no-caps align=left").classes(btn_classes)

        # Spacer + footer
        ui.space()
        with ui.column().classes("gap-1 mt-auto"):
            ui.separator().classes("opacity-10")
            with ui.row().classes("items-center gap-2 mt-2 ml-1"):
                ui.html(
                    '<span style="font-size:0.7rem; opacity:0.5;">&#9918;</span>'
                )
                ui.label("v1.0.0").classes("text-[0.65rem] text-gray-600")

    # Main content
    with ui.column().classes("w-full p-6 gap-6"):
        if active == "games":
            from better.dashboard.pages.todays_games import render

            render(get_service())
        elif active == "live":
            from better.dashboard.pages.live_games import render

            render(get_service())
        elif active == "game_detail":
            from better.dashboard.pages.game_detail import render

            render(get_service(), kwargs.get("game_pk", 0))
        elif active == "schedule":
            from better.dashboard.pages.schedule import render

            render(get_service())
        elif active == "backtest":
            from better.dashboard.pages.backtest import render

            render(get_service())
        elif active == "edge":
            from better.dashboard.pages.edge_analysis import render

            render()
        elif active == "models":
            from better.dashboard.pages.model_status import render

            render(get_service())


def run(port: int = 8501) -> None:
    """Start the NiceGUI dashboard server."""
    ui.run(
        host="127.0.0.1",
        port=port,
        title="BETter MLB Predictions",
        favicon="\u26be",
        show=False,
        reload=False,
    )


if __name__ == "__main__":
    run()
