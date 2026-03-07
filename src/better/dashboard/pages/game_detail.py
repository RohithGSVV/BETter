"""Game Detail page — full breakdown for a single game."""

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

# MLB team full names
TEAM_NAMES: dict[str, str] = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}


def render(svc: PredictionService, game_pk: int) -> None:
    """Render the game detail page for a specific game."""

    # Back button
    ui.button("Back", icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props(
        "flat size=sm color=primary"
    ).classes("rounded-lg mb-2")

    # Find the game in today's predictions
    predictions = svc.get_todays_predictions()
    pred = next((p for p in predictions if p.get("game_pk") == game_pk), None)

    # Check live manager for live data
    manager = svc.get_live_manager()
    live_snap = manager.get_snapshot(game_pk) if manager else None

    if pred is None and live_snap is None:
        with ui.card().classes("w-full glow-card px-8 py-6").style(
            "border-left: 4px solid #ef4444;"
        ):
            with ui.row().classes("items-center gap-3"):
                ui.icon("error").classes("text-red-400 text-2xl")
                ui.label(f"Game {game_pk} not found").classes("text-lg font-semibold")
        return

    home = (pred or {}).get("home_team") or (live_snap.home_team if live_snap else "")
    away = (pred or {}).get("away_team") or (live_snap.away_team if live_snap else "")
    home_color = TEAM_COLORS.get(home, ("#3b82f6", "#1e293b"))[0]
    away_color = TEAM_COLORS.get(away, ("#3b82f6", "#1e293b"))[0]
    home_name = TEAM_NAMES.get(home, home)
    away_name = TEAM_NAMES.get(away, away)

    # Container for live auto-refresh
    content = ui.column().classes("w-full gap-4")

    def render_detail():
        content.clear()
        with content:
            # Re-fetch live snapshot for updates
            snap = manager.get_snapshot(game_pk) if manager else None
            _render_game_detail(svc, game_pk, pred, snap, home, away,
                                home_color, away_color, home_name, away_name)

    render_detail()

    # Auto-refresh every 5s if game is live
    if live_snap and live_snap.status == "live":
        ui.timer(5.0, render_detail)


def _render_game_detail(
    svc: PredictionService,
    game_pk: int,
    pred: dict | None,
    snap,
    home: str,
    away: str,
    home_color: str,
    away_color: str,
    home_name: str,
    away_name: str,
) -> None:
    """Inner render for game detail — called on refresh."""

    is_live = snap is not None and snap.status == "live"
    is_final = snap is not None and snap.status == "final"

    # ── Header card with matchup ─────────────────────────────
    status_color = "#ef4444" if is_live else ("#6b7280" if is_final else "#3b82f6")
    status_label = "LIVE" if is_live else ("FINAL" if is_final else "PRE-GAME")

    with ui.card().classes("w-full glow-card px-6 py-5").style(
        f"border-top: 3px solid {status_color};"
    ):
        # Status badge
        with ui.row().classes("items-center gap-2 mb-3"):
            if is_live:
                ui.html(
                    '<span style="width:8px; height:8px; border-radius:50%; '
                    'background:#ef4444; display:inline-block; '
                    'animation: pulse 1.5s infinite;"></span>'
                )
            ui.html(
                f'<span style="font-size:0.75rem; color:{status_color}; '
                f'font-weight:700; letter-spacing:0.05em;">{status_label}</span>'
            )
            if is_live and snap:
                arrow = "\u25B2" if snap.half == "top" else "\u25BC"
                ui.label(f"{arrow} {snap.inning}").classes("text-sm text-gray-400")

        # Teams and score
        with ui.row().classes("items-center justify-center gap-8 w-full"):
            # Away
            with ui.column().classes("items-center gap-1"):
                ui.html(
                    f'<div style="width:40px; height:40px; border-radius:50%; '
                    f'background:{away_color}; display:flex; align-items:center; '
                    f'justify-content:center; box-shadow: 0 0 15px {away_color}60;">'
                    f'<span style="color:white; font-weight:700; font-size:0.7rem;">'
                    f'{away}</span></div>'
                )
                ui.label(away_name).classes("text-sm text-gray-400")
                if snap:
                    ui.label(str(snap.away_score)).classes("text-3xl font-bold")

            # VS / Score separator
            if snap:
                ui.label("-").classes("text-2xl text-gray-500 font-light")
            else:
                ui.label("vs").classes("text-lg text-gray-500 font-light")

            # Home
            with ui.column().classes("items-center gap-1"):
                ui.html(
                    f'<div style="width:40px; height:40px; border-radius:50%; '
                    f'background:{home_color}; display:flex; align-items:center; '
                    f'justify-content:center; box-shadow: 0 0 15px {home_color}60;">'
                    f'<span style="color:white; font-weight:700; font-size:0.7rem;">'
                    f'{home}</span></div>'
                )
                ui.label(home_name).classes("text-sm text-gray-400")
                if snap:
                    ui.label(str(snap.home_score)).classes("text-3xl font-bold")

    # ── Win probability section ──────────────────────────────
    win_prob = None
    if snap:
        win_prob = snap.win_prob
    elif pred:
        win_prob = (pred.get("meta_prob") or pred.get("consensus_prob")
                    or pred.get("bayesian_prob"))

    if win_prob is not None:
        with ui.card().classes("w-full glow-card px-6 py-4"):
            with ui.row().classes("section-header w-full"):
                ui.icon("analytics").classes("text-blue-400")
                ui.label("Win Probability").classes("text-lg font-semibold")

            # Large probability bar
            away_pct = (1 - win_prob) * 100
            home_pct = win_prob * 100
            ui.html(f'''
                <div style="display:flex; width:100%; height:16px; border-radius:8px;
                            overflow:hidden; margin:8px 0;">
                    <div style="width:{away_pct}%; background:{away_color};
                                transition: width 0.5s ease;"></div>
                    <div style="width:{home_pct}%; background:{home_color};
                                transition: width 0.5s ease;"></div>
                </div>
            ''')
            with ui.row().classes("justify-between w-full"):
                ui.label(f"{away} {1-win_prob:.1%}").classes(
                    "text-sm font-semibold"
                ).style(f"color: {away_color};")
                ui.label(f"{home} {win_prob:.1%}").classes(
                    "text-sm font-semibold"
                ).style(f"color: {home_color};")

    # ── Model breakdown ──────────────────────────────────────
    with ui.card().classes("w-full glow-card px-6 py-4"):
        with ui.row().classes("section-header w-full"):
            ui.icon("hub").classes("text-blue-400")
            ui.label("Model Predictions").classes("text-lg font-semibold")

        models = []
        if pred:
            if pred.get("bayesian_prob") is not None:
                models.append(("Bayesian Kalman", pred["bayesian_prob"], "#3b82f6"))
            if pred.get("monte_carlo_prob") is not None:
                models.append(("Monte Carlo", pred["monte_carlo_prob"], "#a855f7"))
            if pred.get("meta_prob") is not None:
                models.append(("Meta-Learner", pred["meta_prob"], "#22c55e"))
            if pred.get("consensus_prob") is not None:
                models.append(("Consensus", pred["consensus_prob"], "#f59e0b"))

        if snap:
            models.append(("WE Model (Live)", snap.we_prob, "#60a5fa"))
            if snap.pregame_prob is not None:
                models.append(("Pre-Game Prior", snap.pregame_prob, "#94a3b8"))

        if models:
            for name, prob, color in models:
                with ui.row().classes("items-center justify-between w-full py-2").style(
                    "border-bottom: 1px solid rgba(255,255,255,0.04);"
                ):
                    ui.label(name).classes("text-sm text-gray-400")
                    with ui.row().classes("items-center gap-3"):
                        # Mini bar
                        bar_width = prob * 100
                        ui.html(f'''
                            <div style="width:80px; height:6px; border-radius:3px;
                                        background:rgba(255,255,255,0.06); overflow:hidden;">
                                <div style="width:{bar_width}%; height:100%;
                                            background:{color}; border-radius:3px;"></div>
                            </div>
                        ''')
                        ui.label(f"{prob:.1%}").classes("text-sm font-semibold min-w-[50px] text-right")

            # Blend weight if live
            if snap and snap.we_weight:
                ui.separator().classes("my-2 opacity-10")
                with ui.row().classes("items-center gap-2"):
                    ui.icon("tune").classes("text-gray-500 text-sm")
                    ui.label(f"WE blend: {snap.we_weight:.0%} in-game / "
                             f"{1-snap.we_weight:.0%} pre-game").classes(
                        "text-xs text-gray-500"
                    )
        else:
            ui.label("No model predictions available").classes("text-gray-500 text-sm")

    # ── Market & Edge ────────────────────────────────────────
    market_prob = None
    edge = None
    if snap and snap.market_prob is not None:
        market_prob = snap.market_prob
        edge = snap.edge
    elif pred and pred.get("market_implied_prob") is not None:
        market_prob = pred["market_implied_prob"]
        edge = pred.get("edge")

    if market_prob is not None:
        with ui.card().classes("w-full glow-card px-6 py-4"):
            with ui.row().classes("section-header w-full"):
                ui.icon("show_chart").classes("text-blue-400")
                ui.label("Market & Edge").classes("text-lg font-semibold")

            with ui.row().classes("w-full gap-6 flex-wrap"):
                # Market probability
                with ui.column().classes("gap-1"):
                    ui.label("Market Implied").classes("text-[0.65rem] text-gray-500 uppercase")
                    ui.label(f"{market_prob:.1%}").classes("text-xl font-semibold")
                    if snap and snap.market_source:
                        ui.label(snap.market_source).classes("text-[0.6rem] text-gray-600")

                # Model probability
                if win_prob is not None:
                    with ui.column().classes("gap-1"):
                        ui.label("Model").classes("text-[0.65rem] text-gray-500 uppercase")
                        ui.label(f"{win_prob:.1%}").classes("text-xl font-semibold")

                # Edge
                if edge is not None:
                    edge_color = "#22c55e" if edge > 0 else "#ef4444"
                    with ui.column().classes("gap-1"):
                        ui.label("Edge").classes("text-[0.65rem] text-gray-500 uppercase")
                        ui.html(f'''
                            <div style="display:inline-flex; align-items:center; gap:4px;
                                        padding:4px 14px; border-radius:12px;
                                        background:{edge_color}15; border:1px solid {edge_color}40;">
                                <span style="font-size:1.1rem; color:{edge_color}; font-weight:700;">
                                    {edge:+.1%}
                                </span>
                            </div>
                        ''')

    # ── Pitchers ─────────────────────────────────────────────
    if pred:
        home_sp = pred.get("home_sp_name", "")
        away_sp = pred.get("away_sp_name", "")
        if home_sp or away_sp:
            with ui.card().classes("w-full glow-card px-6 py-4"):
                with ui.row().classes("section-header w-full"):
                    ui.icon("person").classes("text-blue-400")
                    ui.label("Starting Pitchers").classes("text-lg font-semibold")

                with ui.row().classes("w-full gap-8"):
                    with ui.column().classes("gap-1"):
                        ui.label(away).classes("text-[0.65rem] text-gray-500 uppercase font-semibold")
                        ui.label(away_sp or "TBD").classes("text-sm")
                    with ui.column().classes("gap-1"):
                        ui.label(home).classes("text-[0.65rem] text-gray-500 uppercase font-semibold")
                        ui.label(home_sp or "TBD").classes("text-sm")

    # ── Bet recommendation ───────────────────────────────────
    if pred:
        recommendations = svc.get_bet_recommendations()
        rec = next((r for r in recommendations if r.get("game_pk") == game_pk), None)
        if rec:
            edge_color = "#22c55e"
            with ui.card().classes("w-full glow-card px-6 py-4").style(
                f"border-left: 4px solid {edge_color};"
            ):
                with ui.row().classes("section-header w-full"):
                    ui.icon("casino").classes("text-green-400")
                    ui.label("Bet Recommendation").classes("text-lg font-semibold")

                with ui.row().classes("w-full gap-6 flex-wrap"):
                    _detail_metric("Side", rec["bet_side"], "#22c55e")
                    _detail_metric("Edge", f"{rec['edge']:+.1%}", "#22c55e")
                    _detail_metric("EV", f"{rec['expected_value']:+.1%}", "#3b82f6")
                    _detail_metric("Kelly", f"{rec['kelly_fraction']:.2%}", "#a855f7")
                    _detail_metric("Bet", f"${rec['bet_amount']:.2f}", "#f59e0b")
                    _detail_metric("Odds", str(rec["odds_american"]), "#94a3b8")
                    _detail_metric("Confidence", rec["confidence"], "#60a5fa")


def _detail_metric(label: str, value: str, color: str) -> None:
    """Render a compact labeled metric."""
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-[0.6rem] text-gray-500 uppercase")
        ui.label(value).classes("text-sm font-semibold").style(f"color: {color};")
