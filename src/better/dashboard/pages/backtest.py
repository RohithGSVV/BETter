"""Backtest Results page — bankroll curve, stats, by-year breakdown."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from nicegui import ui

from better.api.services import PredictionService


def render(svc: PredictionService) -> None:
    """Render the Backtest Results page."""

    # Page title
    with ui.row().classes("items-center gap-3 w-full"):
        ui.icon("show_chart").classes("text-3xl text-blue-400")
        with ui.column().classes("gap-0"):
            ui.label("Backtest Results").classes("text-2xl font-bold")
            ui.label("Historical betting simulation using out-of-fold predictions").classes(
                "text-sm text-gray-400"
            )

    # Controls row
    with ui.card().classes("glow-card w-full px-6 py-4"):
        with ui.row().classes("w-full items-end gap-6 flex-wrap"):
            edge_input = ui.number(
                "Min Edge", value=0.03, min=0.01, max=0.15, step=0.01, format="%.2f"
            ).classes("w-32")
            kelly_input = ui.number(
                "Kelly Fraction", value=0.25, min=0.05, max=1.0, step=0.05, format="%.2f"
            ).classes("w-32")
            model_select = ui.select(
                ["meta_learner", "gbm_ensemble", "bayesian_kalman", "monte_carlo"],
                value="meta_learner",
                label="Model",
            ).classes("w-48")

            ui.button("Run Backtest", icon="play_arrow", on_click=lambda: run_backtest()).props(
                "color=primary"
            ).classes("rounded-lg")

    # Container for results
    results_container = ui.column().classes("w-full gap-6")

    def run_backtest():
        results_container.clear()
        edge = edge_input.value or 0.03
        kelly = kelly_input.value or 0.25
        model = model_select.value or "meta_learner"

        with results_container:
            try:
                summary = svc.get_backtest_summary(edge, kelly, model)
                curve_data = svc.get_backtest_bankroll_curve(edge, kelly, model)
            except Exception as exc:
                with ui.card().classes("glow-card w-full px-6 py-4").style(
                    "border-left: 4px solid #ef4444;"
                ):
                    with ui.row().classes("items-center gap-3"):
                        ui.icon("error").classes("text-red-400 text-xl")
                        ui.label(f"Backtest failed: {exc}").classes("text-red-300")
                return

            roi = summary.get("roi_pct", 0)
            is_profitable = roi > 0

            # Metrics row
            with ui.row().classes("w-full gap-4 flex-wrap"):
                _metric(
                    "Final Bankroll",
                    f"${summary['final_bankroll']:,.0f}",
                    "account_balance",
                    "green" if is_profitable else "red",
                )
                _metric("Total Bets", str(summary["total_bets"]), "receipt_long", "blue")
                _metric(
                    "Win Rate",
                    f"{summary['win_rate']:.1%}",
                    "emoji_events",
                    "green" if summary["win_rate"] > 0.5 else "amber",
                )
                _metric(
                    "ROI",
                    f"{roi:+.1f}%",
                    "trending_up" if is_profitable else "trending_down",
                    "green" if is_profitable else "red",
                )
                _metric(
                    "Sharpe",
                    f"{summary['sharpe_ratio']:.2f}",
                    "speed",
                    "green" if summary["sharpe_ratio"] > 1 else "amber",
                )
                _metric(
                    "Max Drawdown",
                    f"{summary['max_drawdown_pct']:.1f}%",
                    "trending_down",
                    "red",
                )

            # Bankroll curve chart
            if curve_data:
                with ui.row().classes("section-header w-full"):
                    ui.icon("timeline").classes("text-blue-400")
                    ui.label("Bankroll Over Time").classes("text-xl font-semibold")

                curve_df = pd.DataFrame(curve_data)
                curve_df["game_date"] = pd.to_datetime(curve_df["game_date"])

                line_color = "#22c55e" if is_profitable else "#ef4444"
                fill_color = (
                    "rgba(34,197,94,0.08)" if is_profitable else "rgba(239,68,68,0.08)"
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=curve_df["game_date"],
                    y=curve_df["bankroll"],
                    mode="lines",
                    name="Bankroll",
                    line=dict(color=line_color, width=2),
                    fill="tozeroy",
                    fillcolor=fill_color,
                    hovertemplate=(
                        "Date: %{x|%b %d, %Y}<br>Bankroll: $%{y:,.0f}<extra></extra>"
                    ),
                ))
                fig.add_hline(
                    y=summary["initial_bankroll"],
                    line_dash="dot",
                    line_color="rgba(255,255,255,0.15)",
                    annotation_text=f"Initial ${summary['initial_bankroll']:,.0f}",
                    annotation_font_color="rgba(255,255,255,0.4)",
                    annotation_font_size=10,
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="rgba(255,255,255,0.8)",
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True),
                    yaxis=dict(
                        gridcolor="rgba(255,255,255,0.04)",
                        title="Bankroll ($)",
                        showgrid=True,
                    ),
                    hovermode="x unified",
                    hoverlabel=dict(
                        bgcolor="rgba(15,23,42,0.95)",
                        font_color="white",
                        bordercolor="rgba(59,130,246,0.3)",
                    ),
                )
                ui.plotly(fig).classes("w-full h-80")

            # Detailed stats
            with ui.row().classes("section-header w-full"):
                ui.icon("table_chart").classes("text-blue-400")
                ui.label("Detailed Statistics").classes("text-xl font-semibold")

            with ui.card().classes("glow-card w-full max-w-2xl px-6 py-4"):
                stats = [
                    ("Initial Bankroll", f"${summary['initial_bankroll']:,.2f}",
                     "account_balance_wallet"),
                    ("Final Bankroll", f"${summary['final_bankroll']:,.2f}",
                     "savings" if is_profitable else "money_off"),
                    ("Total Bets", str(summary["total_bets"]), "receipt_long"),
                    ("Wins / Losses", f"{summary['wins']} / {summary['losses']}", "scoreboard"),
                    ("Win Rate", f"{summary['win_rate']:.1%}", "percent"),
                    ("ROI", f"{summary['roi_pct']:+.1f}%", "trending_up"),
                    ("Yield", f"{summary['yield_pct']:+.1f}%", "paid"),
                    ("Max Drawdown", f"{summary['max_drawdown_pct']:.1f}%", "trending_down"),
                    ("Avg Edge", f"{summary['avg_edge']:.2%}", "analytics"),
                    ("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}", "speed"),
                    ("Longest Losing Streak", str(summary["longest_losing_streak"]), "warning"),
                ]

                for label, value, icon in stats:
                    with ui.row().classes(
                        "w-full items-center justify-between py-2"
                    ).style("border-bottom: 1px solid rgba(255,255,255,0.04);"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon(icon).classes("text-gray-500 text-sm")
                            ui.label(label).classes("text-gray-400 text-sm")
                        ui.label(value).classes("font-semibold text-sm")

    # Auto-run on load
    run_backtest()


def _metric(title: str, value: str, icon: str, color: str = "blue") -> None:
    """Render a compact metric card with colored accent."""
    color_map = {
        "blue": ("#3b82f6", "metric-blue"),
        "green": ("#22c55e", "metric-green"),
        "amber": ("#f59e0b", "metric-amber"),
        "red": ("#ef4444", "metric-red"),
        "purple": ("#a855f7", "metric-purple"),
    }
    hex_color, css_class = color_map.get(color, color_map["blue"])

    with ui.card().classes(f"glow-card {css_class} px-5 py-3 min-w-[140px]"):
        with ui.row().classes("items-center gap-2"):
            ui.icon(icon).classes("text-lg").style(f"color: {hex_color};")
            with ui.column().classes("gap-0"):
                ui.label(value).classes("text-xl font-bold")
                ui.label(title).classes(
                    "text-[0.6rem] text-gray-400 uppercase tracking-wider"
                )
