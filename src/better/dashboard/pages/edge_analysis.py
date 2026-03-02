"""Edge Analysis page — calibration, edge by month/team, model comparison."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from nicegui import ui

from better.config import settings


def render() -> None:
    """Render the Edge Analysis page."""

    # Page title
    with ui.row().classes("items-center gap-3 w-full"):
        ui.icon("insights").classes("text-3xl text-blue-400")
        with ui.column().classes("gap-0"):
            ui.label("Edge Analysis").classes("text-2xl font-bold")
            ui.label("Model calibration, accuracy lift, and edge distribution").classes(
                "text-sm text-gray-400"
            )

    oof_path = settings.project_root / "results" / "oof_details.csv"
    if not oof_path.exists():
        with ui.card().classes("w-full glow-card px-8 py-6").style(
            "border-left: 4px solid #f59e0b;"
        ):
            with ui.row().classes("items-center gap-4"):
                ui.icon("warning").classes("text-amber-400 text-3xl")
                with ui.column().classes("gap-1"):
                    ui.label("OOF details not found").classes("text-lg font-semibold text-amber-300")
                    ui.label(
                        "Run uv run better model train first to generate out-of-fold predictions."
                    ).classes("text-gray-400 text-sm")
        return

    oof = pd.read_csv(oof_path)

    # Detect available models
    model_cols = [c for c in oof.columns if c.endswith("_prob") and c != "elo_home_win_prob"]
    model_names = [c.replace("_prob", "") for c in model_cols]

    if not model_names:
        ui.label("No model predictions found in OOF data.").classes("text-amber-400")
        return

    # Model selector in styled card
    with ui.card().classes("glow-card px-6 py-4 w-fit"):
        with ui.row().classes("items-center gap-4"):
            ui.icon("tune").classes("text-blue-400")
            model_select = ui.select(
                model_names, value=model_names[0], label="Select Model"
            ).classes("w-56")

    # Container for analysis content
    content = ui.column().classes("w-full gap-6")

    def refresh_analysis():
        content.clear()
        model = model_select.value

        from better.betting.edge import EdgeAnalyzer

        analyzer = EdgeAnalyzer(oof, model=model)

        with content:
            # ── Model comparison ─────────────────────────────
            with ui.row().classes("section-header w-full"):
                ui.icon("compare").classes("text-blue-400")
                ui.label("Model Comparison vs Elo Baseline").classes("text-xl font-semibold")

            try:
                comparison = analyzer.model_comparison()
                _render_dataframe_table(comparison)
            except Exception as exc:
                _error_card(str(exc))

            # ── Calibration plot ─────────────────────────────
            with ui.row().classes("section-header w-full"):
                ui.icon("ssid_chart").classes("text-blue-400")
                ui.label(f"Calibration \u2014 {model}").classes("text-xl font-semibold")

            try:
                cal = analyzer.calibration_report()
                if not cal.empty and "predicted_prob" in cal.columns:
                    fig = go.Figure()

                    # Perfect calibration line
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode="lines",
                        line=dict(dash="dash", color="rgba(255,255,255,0.15)", width=1),
                        name="Perfect",
                        showlegend=True,
                    ))

                    # Model calibration
                    sizes = (
                        (cal["count"] / cal["count"].max() * 25 + 6)
                        if cal["count"].max() > 0
                        else 12
                    )
                    fig.add_trace(go.Scatter(
                        x=cal["predicted_prob"],
                        y=cal["actual_win_rate"],
                        mode="lines+markers",
                        marker=dict(
                            size=sizes,
                            color="#60a5fa",
                            line=dict(color="#3b82f6", width=1),
                        ),
                        line=dict(color="#3b82f6", width=2),
                        name=model.replace("_", " ").title(),
                        text=cal.apply(lambda r: f"n={int(r['count'])}", axis=1),
                        hovertemplate=(
                            "Predicted: %{x:.1%}<br>Actual: %{y:.1%}<br>%{text}<extra></extra>"
                        ),
                    ))

                    fig.update_layout(
                        margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="rgba(255,255,255,0.8)",
                        xaxis=dict(
                            title="Predicted Probability",
                            range=[0, 1],
                            gridcolor="rgba(255,255,255,0.04)",
                        ),
                        yaxis=dict(
                            title="Actual Win Rate",
                            range=[0, 1],
                            gridcolor="rgba(255,255,255,0.04)",
                        ),
                        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
                        hoverlabel=dict(
                            bgcolor="rgba(15,23,42,0.95)",
                            font_color="white",
                            bordercolor="rgba(59,130,246,0.3)",
                        ),
                    )
                    ui.plotly(fig).classes("w-full h-80")

                    # ECE metric badge
                    if "calibration_error" in cal.columns:
                        weights = cal["count"] / cal["count"].sum()
                        ece = (cal["calibration_error"].abs() * weights).sum()
                        ece_color = "#22c55e" if ece < 0.03 else "#f59e0b" if ece < 0.06 else "#ef4444"
                        ui.html(f'''
                            <div style="display:inline-flex; align-items:center; gap:6px;
                                        padding:4px 12px; border-radius:8px;
                                        background:rgba(30,41,59,0.8); border:1px solid {ece_color}40;">
                                <span style="font-size:0.75rem; color:rgba(148,163,184,1);">ECE:</span>
                                <span style="font-size:0.85rem; font-weight:700; color:{ece_color};">
                                    {ece:.4f}
                                </span>
                            </div>
                        ''')
                else:
                    _render_dataframe_table(cal)
            except Exception as exc:
                _error_card(str(exc))

            # ── Win Rate by Edge Threshold ────────────────────
            with ui.row().classes("section-header w-full"):
                ui.icon("filter_alt").classes("text-blue-400")
                ui.label("Win Rate by Edge Threshold").classes("text-xl font-semibold")

            try:
                roi = analyzer.roi_by_edge_threshold()
                if not roi.empty:
                    _render_dataframe_table(roi)
            except Exception as exc:
                _error_card(str(exc))

            # ── Edge by Probability Range ─────────────────────
            with ui.row().classes("section-header w-full"):
                ui.icon("bar_chart").classes("text-blue-400")
                ui.label("Edge by Probability Range").classes("text-xl font-semibold")

            try:
                by_prob = analyzer.edge_by_probability_range()
                if not by_prob.empty:
                    _render_dataframe_table(by_prob)
            except Exception as exc:
                _error_card(str(exc))

            # ── Edge by Month ─────────────────────────────────
            with ui.row().classes("section-header w-full"):
                ui.icon("calendar_month").classes("text-blue-400")
                ui.label("Edge by Month").classes("text-xl font-semibold")

            try:
                by_month = analyzer.edge_by_month()
                if not by_month.empty and "accuracy_lift" in by_month.columns:
                    x_vals = by_month.get("month_name", by_month.index)
                    y_vals = by_month["accuracy_lift"]

                    # Color bars by positive/negative
                    colors = [
                        "#22c55e" if v > 0 else "#ef4444" for v in y_vals
                    ]

                    fig_month = go.Figure(go.Bar(
                        x=x_vals,
                        y=y_vals,
                        marker_color=colors,
                        marker_line_color=colors,
                        marker_line_width=0,
                        hovertemplate="Month: %{x}<br>Lift: %{y:.3f}<extra></extra>",
                    ))
                    fig_month.add_hline(
                        y=0,
                        line_color="rgba(255,255,255,0.15)",
                        line_width=1,
                    )
                    fig_month.update_layout(
                        margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="rgba(255,255,255,0.8)",
                        xaxis=dict(
                            title="Month",
                            gridcolor="rgba(255,255,255,0.04)",
                        ),
                        yaxis=dict(
                            title="Accuracy Lift vs Elo",
                            gridcolor="rgba(255,255,255,0.04)",
                        ),
                        hoverlabel=dict(
                            bgcolor="rgba(15,23,42,0.95)",
                            font_color="white",
                            bordercolor="rgba(59,130,246,0.3)",
                        ),
                    )
                    ui.plotly(fig_month).classes("w-full h-64")
                elif not by_month.empty:
                    _render_dataframe_table(by_month)
                else:
                    ui.label("Month-level data not available.").classes("text-gray-500")
            except Exception as exc:
                _error_card(str(exc))

    model_select.on_value_change(lambda _: refresh_analysis())
    refresh_analysis()


def _render_dataframe_table(df: pd.DataFrame) -> None:
    """Render a pandas DataFrame as a styled NiceGUI table."""
    if df.empty:
        ui.label("No data").classes("text-gray-500")
        return

    columns = []
    for col in df.columns:
        columns.append({
            "name": col,
            "label": col.replace("_", " ").title(),
            "field": col,
            "align": "right" if df[col].dtype in ("float64", "int64") else "left",
            "sortable": True,
        })

    rows = []
    for _, row in df.iterrows():
        r = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                r[col] = f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
            else:
                r[col] = str(val)
        rows.append(r)

    ui.table(columns=columns, rows=rows, row_key=df.columns[0]).classes(
        "w-full"
    ).props("flat bordered dense")


def _error_card(msg: str) -> None:
    """Show an error in a styled card."""
    with ui.card().classes("glow-card px-4 py-3").style("border-left: 3px solid #ef4444;"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("error_outline").classes("text-red-400 text-sm")
            ui.label(msg).classes("text-red-300 text-sm")
