"""Today's Games page — predictions overview, clean and compact."""

from __future__ import annotations

from datetime import date

from nicegui import ui

from better.api.services import PredictionService
from better.config import settings


def render(svc: PredictionService) -> None:
    """Render the Today's Games page — focused on predictions table."""

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
        return

    # Summary metrics (clickable)
    n_games = len(predictions)
    n_bets = len(recommendations)
    edges = [p["edge"] for p in predictions if p.get("edge") is not None]
    avg_edge = sum(edges) / len(edges) if edges else 0
    has_odds = bool(settings.odds_api_key)

    with ui.row().classes("w-full gap-4 flex-wrap"):
        # Games Today → navigates to Live Games
        _metric_card(
            "Games Today", str(n_games), "sports_baseball", "blue",
            on_click=lambda: ui.navigate.to("/live"),
            subtitle="View Live",
        )

        # Bets Recommended
        if has_odds:
            _metric_card("Bets Recommended", str(n_bets), "casino", "green")
        else:
            _metric_card(
                "Bets Recommended", "---", "casino", "green",
                subtitle="No Odds API Key",
            )

        # Avg Edge
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

    # Odds API key missing hint
    if not has_odds:
        with ui.card().classes("w-full glow-card px-5 py-3").style(
            "border-left: 3px solid #3b82f6;"
        ):
            with ui.row().classes("items-center gap-3"):
                ui.icon("info").classes("text-blue-400 text-lg")
                with ui.column().classes("gap-0"):
                    ui.label("Odds & Edge Unavailable").classes(
                        "text-sm font-semibold text-gray-300"
                    )
                    ui.label(
                        "Set ODDS_API_KEY in your .env to enable market odds, "
                        "edge detection, and bet recommendations."
                    ).classes("text-[0.7rem] text-gray-500")

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
