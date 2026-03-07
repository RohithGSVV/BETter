"""Today's Games — predictions dashboard with visual game cards."""

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
    """Render the Today's Games dashboard with visual prediction cards."""

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
    edges = [p["edge"] for p in predictions if p.get("edge") is not None]
    avg_edge = sum(edges) / len(edges) if edges else 0
    has_odds = bool(settings.odds_api_key)

    with ui.row().classes("w-full gap-4 flex-wrap"):
        _metric_card(
            "Games Today", str(n_games), "sports_baseball", "blue",
            on_click=lambda: ui.navigate.to("/live"), subtitle="View Live",
        )
        if has_odds:
            _metric_card("Bets Recommended", str(n_bets), "casino", "green")
        else:
            _metric_card(
                "Bets Recommended", "---", "casino", "green",
                subtitle="No Odds API Key",
            )
        if has_odds and edges:
            _metric_card("Avg Edge", f"{avg_edge:+.1%}", "trending_up", "amber")
        else:
            _metric_card(
                "Avg Edge", "---", "trending_up", "amber",
                subtitle="Needs Odds API" if not has_odds else "No edges",
            )
        if recommendations:
            total_bet = sum(r["bet_amount"] for r in recommendations)
            _metric_card("Total Wagered", f"${total_bet:.0f}", "payments", "purple")

    # ── Game prediction cards ────────────────────────────────────────
    with ui.row().classes("section-header w-full mt-2"):
        ui.icon("list_alt").classes("text-blue-400")
        ui.label("Predictions").classes("text-xl font-semibold")
        ui.badge(str(n_games)).props("color=primary outline")

    with ui.element("div").classes("w-full").style(
        "display: grid;"
        "grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));"
        "gap: 16px;"
    ):
        for p in predictions:
            _game_prediction_card(p)

    # ── Bet recommendations ──────────────────────────────────────────
    if recommendations:
        with ui.row().classes("section-header w-full mt-4"):
            ui.icon("casino").classes("text-green-400")
            ui.label("Bet Recommendations").classes("text-xl font-semibold")
            ui.badge(str(len(recommendations))).props("color=green")

        with ui.element("div").classes("w-full").style(
            "display: grid;"
            "grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));"
            "gap: 16px;"
        ):
            for r in recommendations:
                _bet_rec_card(r)


# ── Components ────────────────────────────────────────────────────────


def _game_prediction_card(p: dict) -> None:
    """Render a single game prediction as a visual card."""
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

    # Determine favored team
    home_favored = best_prob > 0.5

    card = ui.card().classes("glow-card px-5 py-4 cursor-pointer")
    if game_pk:
        card.on("click", lambda gpk=game_pk: ui.navigate.to(f"/game/{gpk}"))

    with card:
        # ── Teams row ────────────────────────────────────────
        with ui.row().classes("items-center justify-between w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.html(
                    f'<span style="width:10px; height:10px; border-radius:50%;'
                    f' background:{away_color}; display:inline-block;'
                    f' box-shadow: 0 0 6px {away_color}80;"></span>'
                )
                bold = "font-bold" if not home_favored else ""
                ui.label(away).classes(f"text-base {bold}")
            ui.label("@").classes("text-gray-600 text-xs")
            with ui.row().classes("items-center gap-2"):
                bold = "font-bold" if home_favored else ""
                ui.label(home).classes(f"text-base {bold}")
                ui.html(
                    f'<span style="width:10px; height:10px; border-radius:50%;'
                    f' background:{home_color}; display:inline-block;'
                    f' box-shadow: 0 0 6px {home_color}80;"></span>'
                )

        # ── Starting pitchers ────────────────────────────────
        away_sp = p.get("away_sp_name", "TBD")
        home_sp = p.get("home_sp_name", "TBD")
        with ui.row().classes("justify-between w-full"):
            ui.label(away_sp).classes("text-[0.7rem] text-gray-500")
            ui.label("vs").classes("text-[0.6rem] text-gray-600")
            ui.label(home_sp).classes("text-[0.7rem] text-gray-500")

        ui.separator().classes("my-2 opacity-10")

        # ── Win probability bar ──────────────────────────────
        ui.html(f'''
            <div style="display:flex; width:100%; height:6px; border-radius:3px;
                        overflow:hidden;">
                <div style="width:{away_pct}%; background:{away_color};
                            transition: width 0.3s;"></div>
                <div style="width:{home_pct}%; background:{home_color};
                            transition: width 0.3s;"></div>
            </div>
        ''')
        with ui.row().classes("justify-between w-full mt-1"):
            ui.label(f"{1 - best_prob:.0%}").classes("text-[0.65rem] text-gray-500")
            ui.html(
                f'<span style="font-size:0.65rem; font-weight:600;'
                f' color:{"#a5b4fc" if home_favored else "#94a3b8"};">'
                f'P(Home) {best_prob:.1%}</span>'
            )

        # ── Model stats row ──────────────────────────────────
        with ui.row().classes("w-full justify-between mt-3"):
            for label, key in [
                ("Bay", "bayesian_prob"),
                ("MC", "monte_carlo_prob"),
                ("Meta", "meta_prob"),
                ("Mkt", "market_implied_prob"),
            ]:
                val = p.get(key)
                val_str = f"{val:.0%}" if val else "\u2014"
                ui.html(f'''
                    <div style="display:flex; flex-direction:column;
                                align-items:center; min-width:40px;">
                        <span style="font-size:0.55rem; color:#6b7280;
                                     text-transform:uppercase;
                                     letter-spacing:0.05em;">{label}</span>
                        <span style="font-size:0.85rem; font-weight:600;
                                     color:#e2e8f0;">{val_str}</span>
                    </div>
                ''')

        # ── Edge + Confidence ────────────────────────────────
        edge_val = p.get("edge")
        conf = p.get("confidence", "\u2014") or "\u2014"

        with ui.row().classes("w-full items-center justify-between mt-2"):
            if edge_val is not None:
                edge_color = "#22c55e" if edge_val > 0 else "#ef4444"
                ui.html(f'''
                    <div style="display:inline-flex; align-items:center; gap:4px;
                                padding:2px 10px; border-radius:12px;
                                background:{edge_color}15;
                                border:1px solid {edge_color}40;">
                        <span style="font-size:0.7rem; color:{edge_color};
                                     font-weight:700;">
                            Edge {edge_val:+.1%}
                        </span>
                    </div>
                ''')
            else:
                ui.html(
                    '<span style="font-size:0.6rem; color:#4b5563;'
                    ' padding:2px 8px;">No edge data</span>'
                )

            if conf != "\u2014":
                ui.html(f'''
                    <span style="font-size:0.6rem; color:#6b7280; padding:2px 8px;
                                 border-radius:8px;
                                 background:rgba(255,255,255,0.04);">
                        Conf: {conf}
                    </span>
                ''')


def _bet_rec_card(r: dict) -> None:
    """Render a single bet recommendation as a card."""
    side = r["bet_side"]
    edge_val = r["edge"]
    edge_color = "#22c55e" if edge_val > 0 else "#ef4444"

    with ui.card().classes("glow-card px-5 py-4").style(
        f"border-left: 3px solid {edge_color};"
    ):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(f"{r['away_team']} @ {r['home_team']}").classes(
                "font-semibold text-base"
            )
            ui.html(f'''
                <span style="font-size:0.7rem; font-weight:700; color:{edge_color};
                             padding:2px 10px; border-radius:12px;
                             background:{edge_color}15;
                             border:1px solid {edge_color}40;">
                    {side.upper()}
                </span>
            ''')

        with ui.row().classes("w-full gap-4 mt-2 flex-wrap"):
            for label, value in [
                ("Model", f"{r['model_prob']:.1%}"),
                ("Market", f"{r['market_prob']:.1%}"),
                ("Edge", f"{r['edge']:+.1%}"),
                ("EV", f"{r['expected_value']:+.1%}"),
                ("Kelly", f"{r['kelly_fraction']:.2%}"),
            ]:
                ui.html(f'''
                    <div style="display:flex; flex-direction:column; gap:0;">
                        <span style="font-size:0.55rem; color:#6b7280;
                                     text-transform:uppercase;
                                     letter-spacing:0.05em;">{label}</span>
                        <span style="font-size:0.85rem; font-weight:600;
                                     color:#e2e8f0;">{value}</span>
                    </div>
                ''')

        with ui.row().classes("w-full items-center justify-between mt-2"):
            ui.html(f'''
                <span style="font-size:1rem; font-weight:700; color:#22c55e;">
                    ${r["bet_amount"]:.2f}
                </span>
            ''')
            ui.html(f'''
                <span style="font-size:0.75rem; color:#94a3b8;">
                    Odds: {r["odds_american"]}
                </span>
            ''')


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
