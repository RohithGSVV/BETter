"""Today's Games page — schedule, predictions, edges, bet recommendations."""

from __future__ import annotations

from datetime import date, timedelta

from nicegui import ui

from better.api.services import PredictionService

# ── MLB team colors (primary, secondary) ─────────────────────────────────

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


def _team_badge(abbr: str) -> str:
    """Return an HTML badge with team color dot."""
    primary, _ = TEAM_COLORS.get(abbr, ("#3b82f6", "#1e293b"))
    return (
        f'<span style="display:inline-flex; align-items:center; gap:5px;">'
        f'<span style="width:8px; height:8px; border-radius:50%; '
        f'background:{primary}; display:inline-block; box-shadow: 0 0 6px {primary}80;"></span>'
        f'<span style="font-weight:600; letter-spacing:0.02em;">{abbr}</span>'
        f'</span>'
    )


def render(svc: PredictionService) -> None:
    """Render the Today's Games page."""

    # Page title
    with ui.row().classes("items-center gap-3 w-full"):
        ui.icon("sports_baseball").classes("text-3xl text-blue-400")
        with ui.column().classes("gap-0"):
            ui.label("Today's Games").classes("text-2xl font-bold")
            ui.label(date.today().strftime("%A, %B %d, %Y")).classes(
                "text-sm text-gray-400"
            )

    # Refresh button
    def refresh():
        svc.refresh_predictions()
        ui.navigate.to("/")

    ui.button("Refresh Data", icon="refresh", on_click=refresh).props(
        "flat size=sm color=primary"
    ).classes("rounded-lg")

    predictions = svc.get_todays_predictions()
    recommendations = svc.get_bet_recommendations()

    if not predictions:
        with ui.card().classes(
            "w-full glow-card px-8 py-6"
        ).style("border-left: 4px solid #3b82f6;"):
            with ui.row().classes("items-center gap-4"):
                ui.html(
                    '<span style="font-size:2.5rem; opacity:0.5;">&#9918;</span>'
                )
                with ui.column().classes("gap-1"):
                    ui.label("No games scheduled today").classes(
                        "text-lg font-semibold"
                    )
                    ui.label(
                        "This is normal during the off-season or on off-days."
                    ).classes("text-gray-400 text-sm")
            _show_next_game_day()
        return

    # Summary metrics
    n_games = len(predictions)
    n_bets = len(recommendations)
    edges = [p["edge"] for p in predictions if p.get("edge") is not None]
    avg_edge = sum(edges) / len(edges) if edges else 0

    with ui.row().classes("w-full gap-4 flex-wrap"):
        _metric_card("Games Today", str(n_games), "sports_baseball", "blue")
        _metric_card("Bets Recommended", str(n_bets), "casino", "green")
        _metric_card(
            "Avg Edge",
            f"{avg_edge:+.1%}" if edges else "N/A",
            "trending_up",
            "amber",
        )
        if recommendations:
            total_bet = sum(r["bet_amount"] for r in recommendations)
            _metric_card("Total Wagered", f"${total_bet:.0f}", "payments", "purple")

    # Predictions table
    with ui.row().classes("section-header w-full mt-2"):
        ui.icon("list_alt").classes("text-blue-400")
        ui.label("Predictions").classes("text-xl font-semibold")

    columns = [
        {"name": "matchup", "label": "Matchup", "field": "matchup", "align": "left",
         "sortable": True, "classes": "text-sm", "headerClasses": "text-left"},
        {"name": "home_sp", "label": "Home SP", "field": "home_sp", "align": "left"},
        {"name": "away_sp", "label": "Away SP", "field": "away_sp", "align": "left"},
        {"name": "p_home", "label": "P(Home)", "field": "p_home", "align": "right", "sortable": True},
        {"name": "bayesian", "label": "Bayesian", "field": "bayesian", "align": "right"},
        {"name": "mc", "label": "Monte Carlo", "field": "mc", "align": "right"},
        {"name": "meta", "label": "Meta", "field": "meta", "align": "right"},
        {"name": "market", "label": "Market", "field": "market", "align": "right"},
        {"name": "edge", "label": "Edge", "field": "edge", "align": "right", "sortable": True},
        {"name": "confidence", "label": "Conf.", "field": "confidence", "align": "center"},
    ]

    rows = []
    for p in predictions:
        best_prob = p.get("meta_prob") or p.get("consensus_prob") or p.get("bayesian_prob")
        edge_val = p.get("edge")
        edge_str = f"{edge_val:+.1%}" if edge_val else "\u2014"

        rows.append({
            "matchup": f"{p['away_team']} @ {p['home_team']}",
            "home_sp": p.get("home_sp_name", "TBD"),
            "away_sp": p.get("away_sp_name", "TBD"),
            "p_home": f"{best_prob:.1%}" if best_prob else "\u2014",
            "bayesian": f"{p['bayesian_prob']:.1%}" if p.get("bayesian_prob") else "\u2014",
            "mc": f"{p['monte_carlo_prob']:.1%}" if p.get("monte_carlo_prob") else "\u2014",
            "meta": f"{p['meta_prob']:.1%}" if p.get("meta_prob") else "\u2014",
            "market": f"{p['market_implied_prob']:.1%}" if p.get("market_implied_prob") else "\u2014",
            "edge": edge_str,
            "confidence": p.get("confidence", "\u2014") or "\u2014",
        })

    ui.table(columns=columns, rows=rows, row_key="matchup").classes(
        "w-full"
    ).props("flat bordered dense")

    # Game cards — visual matchup display
    with ui.row().classes("section-header w-full mt-4"):
        ui.icon("view_module").classes("text-blue-400")
        ui.label("Game Cards").classes("text-xl font-semibold")

    with ui.row().classes("w-full gap-4 flex-wrap"):
        for p in predictions[:8]:
            home = p["home_team"]
            away = p["away_team"]
            best = p.get("meta_prob") or p.get("consensus_prob") or p.get("bayesian_prob")
            home_color = TEAM_COLORS.get(home, ("#3b82f6", "#1e293b"))[0]
            away_color = TEAM_COLORS.get(away, ("#3b82f6", "#1e293b"))[0]

            with ui.card().classes("glow-card px-5 py-4 min-w-[220px] max-w-[280px]"):
                with ui.row().classes("items-center justify-between w-full"):
                    ui.html(_team_badge(away)).classes("text-sm")
                    ui.label("@").classes("text-xs text-gray-500")
                    ui.html(_team_badge(home)).classes("text-sm")

                with ui.row().classes("justify-between w-full mt-1"):
                    ui.label(p.get("away_sp_name", "TBD")).classes(
                        "text-[0.65rem] text-gray-500 truncate max-w-[100px]"
                    )
                    ui.label(p.get("home_sp_name", "TBD")).classes(
                        "text-[0.65rem] text-gray-500 truncate max-w-[100px]"
                    )

                ui.separator().classes("my-2 opacity-20")

                if best:
                    away_pct = (1 - best) * 100
                    home_pct = best * 100
                    ui.html(f'''
                        <div style="display:flex; width:100%; height:6px; border-radius:3px; overflow:hidden;">
                            <div style="width:{away_pct}%; background:{away_color};"></div>
                            <div style="width:{home_pct}%; background:{home_color};"></div>
                        </div>
                    ''')
                    with ui.row().classes("justify-between w-full mt-1"):
                        ui.label(f"{1-best:.0%}").classes("text-xs text-gray-400")
                        ui.label(f"P(Home) {best:.0%}").classes("text-xs text-gray-400")

                edge_val = p.get("edge")
                if edge_val is not None:
                    color = "#22c55e" if edge_val > 0 else "#ef4444"
                    ui.html(f'''
                        <div style="display:inline-flex; align-items:center; gap:4px; margin-top:6px;
                                    padding:2px 8px; border-radius:10px;
                                    background:{color}15; border:1px solid {color}40;">
                            <span style="font-size:0.7rem; color:{color}; font-weight:600;">
                                Edge {edge_val:+.1%}
                            </span>
                        </div>
                    ''')

    # Bet recommendations
    if recommendations:
        with ui.row().classes("section-header w-full mt-4"):
            ui.icon("casino").classes("text-green-400")
            ui.label("Bet Recommendations").classes("text-xl font-semibold")
            ui.badge(str(len(recommendations))).props("color=green")

        rec_columns = [
            {"name": "matchup", "label": "Matchup", "field": "matchup", "align": "left"},
            {"name": "side", "label": "Side", "field": "side", "align": "center"},
            {"name": "model_p", "label": "Model P", "field": "model_p", "align": "right"},
            {"name": "market_p", "label": "Market P", "field": "market_p", "align": "right"},
            {"name": "edge", "label": "Edge", "field": "edge", "align": "right"},
            {"name": "ev", "label": "EV", "field": "ev", "align": "right"},
            {"name": "kelly", "label": "Kelly %", "field": "kelly", "align": "right"},
            {"name": "bet", "label": "Bet $", "field": "bet", "align": "right"},
            {"name": "odds", "label": "Odds", "field": "odds", "align": "right"},
            {"name": "confidence", "label": "Conf.", "field": "confidence", "align": "center"},
        ]

        rec_rows = []
        for r in recommendations:
            rec_rows.append({
                "matchup": f"{r['away_team']} @ {r['home_team']}",
                "side": r["bet_side"],
                "model_p": f"{r['model_prob']:.1%}",
                "market_p": f"{r['market_prob']:.1%}",
                "edge": f"{r['edge']:+.1%}",
                "ev": f"{r['expected_value']:+.1%}",
                "kelly": f"{r['kelly_fraction']:.2%}",
                "bet": f"${r['bet_amount']:.2f}",
                "odds": str(r["odds_american"]),
                "confidence": r["confidence"],
            })

        ui.table(columns=rec_columns, rows=rec_rows, row_key="matchup").classes(
            "w-full"
        ).props("flat bordered dense")

    elif predictions:
        with ui.card().classes("w-full glow-card px-6 py-4").style(
            "border-left: 4px solid #f59e0b;"
        ):
            with ui.row().classes("items-center gap-3"):
                ui.icon("info").classes("text-amber-400 text-xl")
                with ui.column().classes("gap-0"):
                    ui.label("No bets recommended").classes("font-semibold text-gray-300")
                    ui.label(
                        "No games meet the minimum edge threshold with current odds."
                    ).classes("text-gray-500 text-sm")


def _metric_card(title: str, value: str, icon: str, color: str = "blue") -> None:
    """Render a styled metric card with colored accent."""
    color_map = {
        "blue": ("#3b82f6", "metric-blue"),
        "green": ("#22c55e", "metric-green"),
        "amber": ("#f59e0b", "metric-amber"),
        "red": ("#ef4444", "metric-red"),
        "purple": ("#a855f7", "metric-purple"),
    }
    hex_color, css_class = color_map.get(color, color_map["blue"])

    with ui.card().classes(f"glow-card {css_class} px-5 py-4 min-w-[160px]"):
        with ui.row().classes("items-center gap-3"):
            ui.icon(icon).classes("text-2xl").style(f"color: {hex_color};")
            with ui.column().classes("gap-0"):
                ui.label(value).classes("text-2xl font-bold")
                ui.label(title).classes(
                    "text-[0.65rem] text-gray-400 uppercase tracking-wider"
                )


def _show_next_game_day() -> None:
    """Try to find the next day with scheduled games."""
    try:
        from better.data.ingest.mlb_api import MLBStatsClient

        client = MLBStatsClient()
        try:
            today = date.today()
            for offset in range(1, 8):
                check_date = today + timedelta(days=offset)
                games = client.get_schedule(check_date)
                if games:
                    with ui.row().classes("items-center gap-2 mt-3"):
                        ui.icon("event").classes("text-blue-300 text-sm")
                        ui.label(
                            f"Next games: {check_date.strftime('%A, %B %d')} "
                            f"({len(games)} games scheduled)"
                        ).classes("text-sm text-blue-300")
                    return
            ui.label("No games found in the next 7 days.").classes(
                "text-sm text-gray-500 mt-2"
            )
        finally:
            client.close()
    except Exception:
        ui.label("Could not check upcoming schedule.").classes(
            "text-sm text-gray-500 mt-2"
        )
