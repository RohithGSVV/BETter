"""Today's Games — matchup strip dashboard with inline probability bars."""

from __future__ import annotations

from datetime import date

from nicegui import ui

from better.api.services import PredictionService
from better.config import settings

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
    "ATH": ("#003831", "#efb21e"),
}


def render(svc: PredictionService) -> None:
    """Render the Today's Games dashboard with matchup strips."""

    # ── Page header ──────────────────────────────────────────────────
    with ui.row().classes("items-center justify-between w-full"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("sports_baseball").classes("text-3xl text-blue-400")
            with ui.column().classes("gap-0"):
                ui.label("Today's Games").classes("text-2xl font-bold")
                ui.label(date.today().strftime("%A, %B %d, %Y")).classes(
                    "text-sm text-gray-400"
                )

        def refresh():
            svc.refresh_predictions()
            ui.navigate.to("/")

        ui.button("Refresh", icon="refresh", on_click=refresh).props(
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
        return

    # ── Summary metrics ──────────────────────────────────────────────
    n_games = len(predictions)
    n_bets = len(recommendations)
    has_odds = bool(settings.odds_api_key)
    has_market = any(p.get("market_implied_prob") for p in predictions)
    edges = [p["edge"] for p in predictions if p.get("edge") is not None]
    avg_edge = sum(edges) / len(edges) if edges else 0

    with ui.row().classes("w-full gap-4 flex-wrap"):
        _metric_card(
            "Games Today", str(n_games), "sports_baseball", "blue",
            on_click=lambda: ui.navigate.to("/live"), subtitle="View Live",
        )

        # Bets Recommended — contextual messaging
        if not has_odds:
            _metric_card(
                "Bets Recommended", "---", "casino", "green",
                subtitle="No Odds API Key",
            )
        elif not has_market:
            _metric_card(
                "Bets Recommended", "---", "casino", "green",
                subtitle="No market odds",
            )
        else:
            _metric_card("Bets Recommended", str(n_bets), "casino", "green")

        # Avg Edge
        if has_market and edges:
            _metric_card("Avg Edge", f"{avg_edge:+.1%}", "trending_up", "amber")
        elif not has_odds:
            _metric_card(
                "Avg Edge", "---", "trending_up", "amber",
                subtitle="No Odds API Key",
            )
        elif not has_market:
            _metric_card(
                "Avg Edge", "---", "trending_up", "amber",
                subtitle="No market odds",
            )
        else:
            _metric_card(
                "Avg Edge", "---", "trending_up", "amber",
                subtitle="No edges",
            )

        if recommendations:
            total_bet = sum(r["bet_amount"] for r in recommendations)
            _metric_card("Total Wagered", f"${total_bet:.0f}", "payments", "purple")

    # ── Matchup strips ───────────────────────────────────────────────
    with ui.row().classes("section-header w-full mt-2"):
        ui.icon("list_alt").classes("text-blue-400")
        ui.label("Predictions").classes("text-xl font-semibold")
        ui.badge(str(n_games)).props("color=primary outline")

    with ui.card().classes("glow-card w-full px-0 py-0 overflow-hidden"):
        for i, p in enumerate(predictions):
            if i > 0:
                ui.html(
                    '<div style="height:1px; background:rgba(255,255,255,0.04);'
                    ' margin:0 16px;"></div>'
                )
            _game_strip(p)

    # ── Bet recommendations ──────────────────────────────────────────
    if recommendations:
        with ui.row().classes("section-header w-full mt-4"):
            ui.icon("casino").classes("text-green-400")
            ui.label("Bet Recommendations").classes("text-xl font-semibold")
            ui.badge(str(len(recommendations))).props("color=green")

        with ui.card().classes("glow-card w-full px-0 py-0 overflow-hidden"):
            for i, r in enumerate(recommendations):
                if i > 0:
                    ui.html(
                        '<div style="height:1px;'
                        ' background:rgba(255,255,255,0.04);'
                        ' margin:0 16px;"></div>'
                    )
                _bet_strip(r)


# ── Components ────────────────────────────────────────────────────────


def _game_strip(p: dict) -> None:
    """Render a single game as a horizontal matchup strip."""
    away = p["away_team"]
    home = p["home_team"]
    away_color = TEAM_COLORS.get(away, ("#3b82f6", "#1e293b"))[0]
    home_color = TEAM_COLORS.get(home, ("#3b82f6", "#1e293b"))[0]
    game_pk = p.get("game_pk")

    best_prob = (
        p.get("meta_prob")
        or p.get("consensus_prob")
        or p.get("bayesian_prob")
        or 0.5
    )
    away_pct = (1 - best_prob) * 100
    home_pct = best_prob * 100
    home_favored = best_prob > 0.5
    favored_color = home_color if home_favored else away_color

    away_sp = p.get("away_sp_name") or "TBD"
    home_sp = p.get("home_sp_name") or "TBD"

    # Model values
    models = []
    for label, key in [
        ("Bay", "bayesian_prob"),
        ("MC", "monte_carlo_prob"),
        ("Meta", "meta_prob"),
        ("Mkt", "market_implied_prob"),
    ]:
        val = p.get(key)
        models.append(f"{label} {val:.0%}" if val else f"{label} \u2014")
    models_text = " &middot; ".join(models)

    # Edge
    edge_val = p.get("edge")
    if edge_val is not None:
        ec = "#22c55e" if edge_val > 0 else "#ef4444"
        edge_html = (
            f'<span style="font-size:0.65rem; font-weight:700; color:{ec};'
            f" padding:1px 8px; border-radius:10px;"
            f" background:{ec}12; border:1px solid {ec}35;"
            f'">Edge {edge_val:+.1%}</span>'
        )
    else:
        edge_html = (
            '<span style="font-size:0.6rem; color:#4b5563;">No edge</span>'
        )

    # Favored team styling
    away_weight = "700" if not home_favored else "400"
    home_weight = "700" if home_favored else "400"
    away_name_color = "#e2e8f0" if not home_favored else "#94a3b8"
    home_name_color = "#e2e8f0" if home_favored else "#94a3b8"

    strip_html = f'''
    <div style="display:flex; flex-direction:column; gap:5px;
                padding:14px 20px; cursor:pointer;
                border-left:3px solid {favored_color};
                transition: background 0.15s ease;"
         onmouseenter="this.style.background='rgba(59,130,246,0.04)'"
         onmouseleave="this.style.background='transparent'">

        <!-- Row 1: Teams + probability bar -->
        <div style="display:flex; align-items:center; gap:10px; width:100%;">
            <div style="display:flex; align-items:center; gap:6px;
                        min-width:90px;">
                <span style="width:10px; height:10px; border-radius:50%;
                             background:{away_color}; display:inline-block;
                             box-shadow:0 0 5px {away_color}70;
                             flex-shrink:0;"></span>
                <span style="font-size:0.95rem; font-weight:{away_weight};
                             color:{away_name_color};">{away}</span>
            </div>

            <span style="font-size:0.8rem; color:#94a3b8; min-width:32px;
                         text-align:right; font-weight:500;">
                {1 - best_prob:.0%}
            </span>

            <div style="flex:1; display:flex; height:8px; border-radius:4px;
                        overflow:hidden; min-width:80px;">
                <div style="width:{away_pct}%; background:{away_color};
                            transition:width 0.3s;"></div>
                <div style="width:{home_pct}%; background:{home_color};
                            transition:width 0.3s;"></div>
            </div>

            <span style="font-size:0.8rem; color:#94a3b8; min-width:32px;
                         text-align:left; font-weight:500;">
                {best_prob:.0%}
            </span>

            <div style="display:flex; align-items:center; gap:6px;
                        min-width:90px; justify-content:flex-end;">
                <span style="font-size:0.95rem; font-weight:{home_weight};
                             color:{home_name_color};">{home}</span>
                <span style="width:10px; height:10px; border-radius:50%;
                             background:{home_color}; display:inline-block;
                             box-shadow:0 0 5px {home_color}70;
                             flex-shrink:0;"></span>
            </div>
        </div>

        <!-- Row 2: Pitchers -->
        <div style="display:flex; align-items:center;
                    justify-content:space-between; padding:0 2px;">
            <span style="font-size:0.7rem; color:#6b7280;
                         min-width:90px;">{away_sp}</span>
            <span style="font-size:0.55rem; color:#4b5563;
                         letter-spacing:0.1em;">VS</span>
            <span style="font-size:0.7rem; color:#6b7280;
                         min-width:90px;
                         text-align:right;">{home_sp}</span>
        </div>

        <!-- Row 3: Model values + Edge -->
        <div style="display:flex; align-items:center;
                    justify-content:space-between; padding:0 2px;">
            <span style="font-size:0.6rem; color:#6b7280;
                         letter-spacing:0.02em;">{models_text}</span>
            {edge_html}
        </div>
    </div>
    '''

    strip = ui.html(strip_html)
    if game_pk:
        strip.on(
            "click",
            lambda gpk=game_pk: ui.navigate.to(f"/game/{gpk}"),
        )


def _bet_strip(r: dict) -> None:
    """Render a single bet recommendation as a horizontal strip."""
    edge_val = r["edge"]
    ec = "#22c55e" if edge_val > 0 else "#ef4444"
    side = r["bet_side"].upper()

    strip_html = f'''
    <div style="display:flex; align-items:center; gap:16px;
                padding:12px 20px; border-left:3px solid {ec};">
        <div style="display:flex; flex-direction:column; gap:0;
                    min-width:120px;">
            <span style="font-size:0.9rem; font-weight:600;
                         color:#e2e8f0;">
                {r["away_team"]} @ {r["home_team"]}
            </span>
            <span style="font-size:0.65rem; color:{ec};
                         font-weight:700;">{side}</span>
        </div>
        <div style="display:flex; gap:16px; flex:1; flex-wrap:wrap;">
            <div style="display:flex; flex-direction:column; gap:0;">
                <span style="font-size:0.5rem; color:#6b7280;
                             text-transform:uppercase;
                             letter-spacing:0.05em;">Model</span>
                <span style="font-size:0.8rem; font-weight:600;
                             color:#e2e8f0;">{r["model_prob"]:.1%}</span>
            </div>
            <div style="display:flex; flex-direction:column; gap:0;">
                <span style="font-size:0.5rem; color:#6b7280;
                             text-transform:uppercase;
                             letter-spacing:0.05em;">Market</span>
                <span style="font-size:0.8rem; font-weight:600;
                             color:#e2e8f0;">{r["market_prob"]:.1%}</span>
            </div>
            <div style="display:flex; flex-direction:column; gap:0;">
                <span style="font-size:0.5rem; color:#6b7280;
                             text-transform:uppercase;
                             letter-spacing:0.05em;">Edge</span>
                <span style="font-size:0.8rem; font-weight:700;
                             color:{ec};">{r["edge"]:+.1%}</span>
            </div>
            <div style="display:flex; flex-direction:column; gap:0;">
                <span style="font-size:0.5rem; color:#6b7280;
                             text-transform:uppercase;
                             letter-spacing:0.05em;">Kelly</span>
                <span style="font-size:0.8rem; font-weight:600;
                             color:#e2e8f0;">{r["kelly_fraction"]:.2%}</span>
            </div>
        </div>
        <div style="display:flex; flex-direction:column;
                    align-items:flex-end; gap:0;">
            <span style="font-size:1.1rem; font-weight:700;
                         color:#22c55e;">${r["bet_amount"]:.2f}</span>
            <span style="font-size:0.65rem;
                         color:#6b7280;">Odds: {r["odds_american"]}</span>
        </div>
    </div>
    '''
    ui.html(strip_html)


def _metric_card(
    title: str,
    value: str,
    icon: str,
    color: str = "blue",
    on_click=None,
    subtitle: str = "",
) -> None:
    """Render a styled metric card with colored accent and optional click."""
    color_map = {
        "blue": ("#3b82f6", "metric-blue"),
        "green": ("#22c55e", "metric-green"),
        "amber": ("#f59e0b", "metric-amber"),
        "red": ("#ef4444", "metric-red"),
        "purple": ("#a855f7", "metric-purple"),
    }
    hex_color, css_class = color_map.get(color, color_map["blue"])

    card_classes = f"glow-card {css_class} px-5 py-4 min-w-[160px]"
    if on_click:
        card_classes += " cursor-pointer"

    card = ui.card().classes(card_classes)
    if on_click:
        card.on("click", on_click)

    with card:
        with ui.row().classes("items-center gap-3"):
            ui.icon(icon).classes("text-2xl").style(f"color: {hex_color};")
            with ui.column().classes("gap-0"):
                ui.label(value).classes("text-2xl font-bold")
                ui.label(title).classes(
                    "text-[0.65rem] text-gray-400 uppercase tracking-wider"
                )
                if subtitle:
                    ui.label(subtitle).classes(
                        "text-[0.6rem] text-gray-500 italic"
                    )
