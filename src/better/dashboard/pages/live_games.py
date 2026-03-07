"""Live Games page — real-time in-game predictions with auto-refresh."""

from __future__ import annotations

from datetime import datetime, timezone

from nicegui import ui

from better.api.services import PredictionService

# ── MLB team colors ──────────────────────────────────────────────────────

TEAM_COLORS: dict[str, tuple[str, str]] = {
    "ARI": ("#a71930", "#e3d4ad"), "ATL": ("#ce1141", "#13274f"),
    "BAL": ("#df4601", "#27251f"), "BOS": ("#bd3039", "#0c2340"),
    "CHC": ("#0e3386", "#cc3433"), "CWS": ("#27251f", "#c4ced4"),
    "CIN": ("#c6011f", "#000000"), "CLE": ("#00385d", "#e31937"),
    "COL": ("#33006f", "#c4ced4"), "DET": ("#0c2340", "#fa4616"),
    "HOU": ("#002d62", "#eb6e1f"), "KC":  ("#004687", "#bd9b60"),
    "LAA": ("#ba0021", "#003263"), "LAD": ("#005a9c", "#ef3e42"),
    "MIA": ("#00a3e0", "#ef3340"), "MIL": ("#ffc52f", "#12284b"),
    "MIN": ("#002b5c", "#d31145"), "NYM": ("#002d72", "#ff5910"),
    "NYY": ("#003087", "#c4ced4"), "OAK": ("#003831", "#efb21e"),
    "PHI": ("#e81828", "#002d72"), "PIT": ("#27251f", "#fdb827"),
    "SD":  ("#2f241d", "#ffc425"), "SF":  ("#fd5a1e", "#27251f"),
    "SEA": ("#0c2c56", "#005c5c"), "STL": ("#c41e3a", "#0c2340"),
    "TB":  ("#092c5c", "#8fbce6"), "TEX": ("#003278", "#c0111f"),
    "TOR": ("#134a8e", "#1d2d5c"), "WSH": ("#ab0003", "#14225a"),
}


def _runner_dots(runners: int) -> str:
    """Render a small diamond-shaped base indicator."""
    b1 = "rgba(59,130,246,0.9)" if runners & 0b100 else "rgba(255,255,255,0.12)"
    b2 = "rgba(59,130,246,0.9)" if runners & 0b010 else "rgba(255,255,255,0.12)"
    b3 = "rgba(59,130,246,0.9)" if runners & 0b001 else "rgba(255,255,255,0.12)"
    return f'''
    <div style="display:inline-flex; flex-direction:column; align-items:center; gap:2px; margin:0 6px;">
        <div style="width:8px;height:8px;background:{b2};transform:rotate(45deg);border-radius:1px;"></div>
        <div style="display:flex;gap:8px;">
            <div style="width:8px;height:8px;background:{b3};transform:rotate(45deg);border-radius:1px;"></div>
            <div style="width:8px;height:8px;background:{b1};transform:rotate(45deg);border-radius:1px;"></div>
        </div>
    </div>
    '''


def _out_dots(outs: int) -> str:
    """Render out indicators."""
    dots = ""
    for i in range(3):
        color = "#f59e0b" if i < outs else "rgba(255,255,255,0.12)"
        dots += (
            f'<span style="width:7px;height:7px;border-radius:50%;'
            f'background:{color};display:inline-block;margin:0 1px;"></span>'
        )
    return f'<span style="display:inline-flex;align-items:center;gap:1px;">{dots}</span>'


def _inning_display(inning: int, half: str) -> str:
    """Format inning as arrow + number."""
    arrow = "\u25B2" if half == "top" else "\u25BC"
    return f"{arrow} {inning}"


def _format_duration(started_at) -> str:
    """Format elapsed time since game started."""
    if started_at is None:
        return ""
    now = datetime.now(timezone.utc)
    delta = now - started_at
    total_mins = int(delta.total_seconds() / 60)
    hours = total_mins // 60
    mins = total_mins % 60
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def _inning_progress(inning: int, half: str) -> float:
    """Compute game progress as 0-1 for a 9-inning game.

    Each half-inning is ~1/18 of a regulation game.
    """
    # half_innings completed: (inning-1)*2 + (1 if bottom)
    completed = (inning - 1) * 2
    if half == "bottom":
        completed += 1
    # 18 half-innings in a 9-inning game
    return min(completed / 18.0, 1.0)


def render(svc: PredictionService) -> None:
    """Render the Live Games page with auto-refresh."""

    # Page title
    with ui.row().classes("items-center gap-3 w-full"):
        ui.icon("live_tv").classes("text-3xl text-red-400")
        with ui.column().classes("gap-0"):
            ui.label("Live Games").classes("text-2xl font-bold")
            ui.label("Real-time predictions with 3s refresh").classes(
                "text-sm text-gray-400"
            )

    # Container that refreshes
    content = ui.column().classes("w-full gap-4")

    def refresh_live():
        content.clear()
        with content:
            _render_live_content(svc)

    refresh_live()

    # Auto-refresh timer (every 3 seconds)
    ui.timer(3.0, refresh_live)


def _render_live_content(svc: PredictionService) -> None:
    """Inner rendering logic — called by timer."""
    manager = svc.get_live_manager()

    if manager is None:
        _show_offline()
        return

    snapshots = manager.get_all_snapshots()

    if not snapshots:
        _show_no_games()
        return

    live = [s for s in snapshots if s.status == "live"]
    final = [s for s in snapshots if s.status == "final"]
    pre = [s for s in snapshots if s.status == "pre"]

    # Status bar
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    with ui.row().classes("w-full items-center gap-4"):
        _status_pill("LIVE", len(live), "#ef4444")
        _status_pill("Final", len(final), "#6b7280")
        _status_pill("Pre-Game", len(pre), "#3b82f6")
        ui.space()
        ui.html(
            f'<span style="font-size:0.7rem; color:#6b7280;">Last update: {now_str}</span>'
        )

    # Live games first
    if live:
        with ui.row().classes("section-header w-full mt-2"):
            ui.html(
                '<span style="color:#ef4444; font-size:0.6rem; '
                'animation: pulse 1.5s infinite;">&#9679;</span>'
            )
            ui.label("LIVE").classes("text-lg font-bold text-red-400")

        with ui.row().classes("w-full gap-4 flex-wrap"):
            for snap in live:
                _live_game_card(snap)

    # Pre-game
    if pre:
        with ui.row().classes("section-header w-full mt-2"):
            ui.icon("schedule").classes("text-blue-400")
            ui.label("Pre-Game").classes("text-lg font-semibold")

        with ui.row().classes("w-full gap-4 flex-wrap"):
            for snap in pre:
                _live_game_card(snap)

    # Final
    if final:
        with ui.row().classes("section-header w-full mt-2"):
            ui.icon("flag").classes("text-gray-400")
            ui.label("Final").classes("text-lg font-semibold text-gray-400")

        with ui.row().classes("w-full gap-4 flex-wrap"):
            for snap in final:
                _live_game_card(snap)


def _live_game_card(snap) -> None:
    """Render a single live game card with all details."""
    home = snap.home_team
    away = snap.away_team
    home_color = TEAM_COLORS.get(home, ("#3b82f6", "#1e293b"))[0]
    away_color = TEAM_COLORS.get(away, ("#3b82f6", "#1e293b"))[0]

    is_live = snap.status == "live"
    is_final = snap.status == "final"
    border_color = (
        "#ef4444" if is_live
        else ("#4b5563" if is_final else "#3b82f6")
    )

    with ui.card().classes(
        "glow-card px-5 py-4 min-w-[280px] max-w-[340px] cursor-pointer"
    ).style(
        f"border-top: 2px solid {border_color};"
    ).on("click", lambda gpk=snap.game_pk: ui.navigate.to(f"/game/{gpk}")):

        # Status + inning + duration
        with ui.row().classes("items-center justify-between w-full"):
            if is_live:
                ui.html(
                    f'<span style="display:inline-flex; align-items:center; gap:4px; '
                    f'font-size:0.7rem; color:#ef4444; font-weight:600;">'
                    f'<span style="animation: pulse 1.5s infinite;">&#9679;</span>'
                    f'{_inning_display(snap.inning, snap.half)}</span>'
                )
                ui.html(f'{_out_dots(snap.outs)}')
                ui.html(f'{_runner_dots(snap.runners)}')
            elif is_final:
                ui.html(
                    '<span style="font-size:0.7rem; color:#6b7280; '
                    'font-weight:600;">FINAL</span>'
                )
            else:
                # Pre-game: show scheduled start time
                time_label = snap.game_time if snap.game_time else "PRE-GAME"
                ui.html(
                    f'<span style="font-size:0.7rem; color:#3b82f6; '
                    f'font-weight:600;">{time_label}</span>'
                )

        # Duration badge (for live and final games)
        if is_live or is_final:
            duration = _format_duration(snap.started_at)
            if duration:
                with ui.row().classes("w-full mt-1"):
                    ui.html(
                        f'<span style="font-size:0.6rem; color:#6b7280; '
                        f'display:inline-flex; align-items:center; gap:3px;">'
                        f'<span style="font-size:0.55rem;">&#9200;</span>'
                        f'{duration}</span>'
                    )

        # Score display
        with ui.row().classes("items-center justify-between w-full mt-2"):
            with ui.row().classes("items-center gap-2"):
                ui.html(
                    f'<span style="width:10px; height:10px; border-radius:50%; '
                    f'background:{away_color}; display:inline-block; '
                    f'box-shadow: 0 0 6px {away_color}80;"></span>'
                )
                ui.label(away).classes("font-bold text-lg")
            ui.label(str(snap.away_score)).classes("text-2xl font-bold text-gray-300")

        with ui.row().classes("items-center justify-between w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.html(
                    f'<span style="width:10px; height:10px; border-radius:50%; '
                    f'background:{home_color}; display:inline-block; '
                    f'box-shadow: 0 0 6px {home_color}80;"></span>'
                )
                ui.label(home).classes("font-bold text-lg")
            ui.label(str(snap.home_score)).classes("text-2xl font-bold text-gray-300")

        ui.separator().classes("my-2 opacity-20")

        # Inning progress bar (for live games)
        if is_live:
            progress = _inning_progress(snap.inning, snap.half)
            progress_pct = progress * 100
            ui.html(f'''
                <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                    <div style="flex:1; height:4px; border-radius:2px;
                                background:rgba(255,255,255,0.06); overflow:hidden;">
                        <div style="width:{progress_pct}%; height:100%;
                                    background:linear-gradient(90deg, #3b82f6, #60a5fa);
                                    border-radius:2px; transition: width 0.5s ease;"></div>
                    </div>
                    <span style="font-size:0.55rem; color:#6b7280; min-width:20px;">
                        {snap.inning}/9
                    </span>
                </div>
            ''')

        # Win probability bar
        away_pct = (1 - snap.win_prob) * 100
        home_pct = snap.win_prob * 100
        ui.html(f'''
            <div style="display:flex; width:100%; height:8px; border-radius:4px; overflow:hidden;">
                <div style="width:{away_pct}%; background:{away_color}; transition: width 0.5s ease;"></div>
                <div style="width:{home_pct}%; background:{home_color}; transition: width 0.5s ease;"></div>
            </div>
        ''')
        with ui.row().classes("justify-between w-full mt-1"):
            ui.label(f"{1 - snap.win_prob:.0%}").classes("text-xs text-gray-400")
            ui.label(f"P(Home) {snap.win_prob:.0%}").classes("text-xs text-gray-400")

        # Model details
        with ui.row().classes("justify-between w-full mt-2"):
            with ui.column().classes("gap-0"):
                ui.label("WE Model").classes("text-[0.6rem] text-gray-500 uppercase")
                ui.label(f"{snap.we_prob:.1%}").classes("text-sm font-semibold")
            with ui.column().classes("gap-0 items-center"):
                ui.label("Blend").classes("text-[0.6rem] text-gray-500 uppercase")
                ui.label(f"{snap.we_weight:.0%} WE").classes("text-sm font-semibold")
            with ui.column().classes("gap-0 items-end"):
                ui.label("Pre-Game").classes("text-[0.6rem] text-gray-500 uppercase")
                ui.label(
                    f"{snap.pregame_prob:.1%}" if snap.pregame_prob else "\u2014"
                ).classes("text-sm font-semibold")

        # Market + Edge
        if snap.market_prob is not None:
            ui.separator().classes("my-2 opacity-10")
            with ui.row().classes("justify-between w-full items-center"):
                with ui.column().classes("gap-0"):
                    ui.label("Market").classes("text-[0.6rem] text-gray-500 uppercase")
                    ui.label(f"{snap.market_prob:.1%}").classes("text-sm")
                    ui.html(
                        f'<span style="font-size:0.6rem; color:#6b7280;">'
                        f'{snap.market_source}</span>'
                    )

                if snap.edge is not None:
                    edge_color = "#22c55e" if snap.edge > 0 else "#ef4444"
                    ui.html(f'''
                        <div style="display:inline-flex; align-items:center; gap:4px;
                                    padding:3px 10px; border-radius:12px;
                                    background:{edge_color}15; border:1px solid {edge_color}40;">
                            <span style="font-size:0.8rem; color:{edge_color}; font-weight:700;">
                                Edge {snap.edge:+.1%}
                            </span>
                        </div>
                    ''')


def _status_pill(label: str, count: int, color: str) -> None:
    """Render a status count pill."""
    opacity = "0.4" if count == 0 else "1"
    ui.html(f'''
        <span style="display:inline-flex; align-items:center; gap:5px; padding:3px 10px;
                     border-radius:12px; background:{color}20; border:1px solid {color}50;
                     opacity:{opacity};">
            <span style="font-size:0.7rem; color:{color}; font-weight:600;">{count}</span>
            <span style="font-size:0.65rem; color:{color}90;">{label}</span>
        </span>
    ''')


def _show_offline() -> None:
    """Show offline message."""
    with ui.card().classes("w-full glow-card px-8 py-6").style(
        "border-left: 4px solid #ef4444;"
    ):
        with ui.row().classes("items-center gap-4"):
            ui.icon("wifi_off").classes("text-3xl text-red-400")
            with ui.column().classes("gap-1"):
                ui.label("Live Tracking Offline").classes("text-lg font-semibold")
                ui.label(
                    "The live game manager is not running. "
                    "Start the API server to enable live tracking."
                ).classes("text-gray-400 text-sm")


def _show_no_games() -> None:
    """Show no games message."""
    with ui.card().classes("w-full glow-card px-8 py-6").style(
        "border-left: 4px solid #3b82f6;"
    ):
        with ui.row().classes("items-center gap-4"):
            ui.html('<span style="font-size:2.5rem; opacity:0.5;">&#9918;</span>')
            with ui.column().classes("gap-1"):
                ui.label("No Active Games").classes("text-lg font-semibold")
                ui.label(
                    "No games are currently being tracked. Games will appear here "
                    "when today's schedule is loaded."
                ).classes("text-gray-400 text-sm")
