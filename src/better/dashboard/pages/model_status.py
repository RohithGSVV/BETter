"""Model Status page — loaded models, accuracy metrics, training info."""

from __future__ import annotations

import pandas as pd
from nicegui import ui

from better.api.services import PredictionService
from better.config import settings


def render(svc: PredictionService) -> None:
    """Render the Model Status page."""

    # Page title
    with ui.row().classes("items-center gap-3 w-full"):
        ui.icon("hub").classes("text-3xl text-blue-400")
        with ui.column().classes("gap-0"):
            ui.label("Model Status").classes("text-2xl font-bold")
            ui.label("Loaded models, accuracy metrics, and storage details").classes(
                "text-sm text-gray-400"
            )

    status = svc.get_model_status()

    # Model display config: (display_name, icon, description, emoji)
    model_display = {
        "gbm_ensemble": (
            "GBM Ensemble", "forest", "XGBoost + LightGBM + CatBoost",
            '<span style="font-size:1.5rem;">&#127795;</span>',  # tree
        ),
        "bayesian_kalman": (
            "Bayesian Kalman", "psychology", "Kalman filter team strength tracking",
            '<span style="font-size:1.5rem;">&#129504;</span>',  # brain
        ),
        "monte_carlo": (
            "Monte Carlo", "casino", "10K-game run-scoring simulation",
            '<span style="font-size:1.5rem;">&#127922;</span>',  # dice
        ),
        "meta_learner": (
            "Meta-Learner", "hub", "Logistic stacking + Platt calibration",
            '<span style="font-size:1.5rem;">&#129302;</span>',  # robot
        ),
    }

    # Model load status cards
    with ui.row().classes("section-header w-full"):
        ui.icon("check_circle").classes("text-blue-400")
        ui.label("Loaded Models").classes("text-xl font-semibold")
        # Count loaded
        n_loaded = sum(1 for v in status["models_loaded"].values() if v)
        n_total = len(model_display)
        badge_color = "green" if n_loaded == n_total else "amber" if n_loaded > 0 else "red"
        ui.badge(f"{n_loaded}/{n_total}").props(f"color={badge_color}")

    with ui.row().classes("w-full gap-4 flex-wrap"):
        for name, (display, icon, desc, emoji) in model_display.items():
            loaded = status["models_loaded"].get(name, False)

            if loaded:
                border_color = "#22c55e"
                bg_style = (
                    "background: linear-gradient(145deg, rgba(34,197,94,0.08) 0%, "
                    "rgba(15,23,42,0.9) 100%);"
                )
                status_html = (
                    '<span style="display:inline-flex; align-items:center; gap:4px; '
                    'padding:2px 8px; border-radius:10px; background:rgba(34,197,94,0.15); '
                    'border:1px solid rgba(34,197,94,0.3);">'
                    '<span style="width:6px; height:6px; border-radius:50%; '
                    'background:#22c55e; animation: pulse 2s infinite;"></span>'
                    '<span style="font-size:0.7rem; color:#22c55e; font-weight:600;">Active</span>'
                    '</span>'
                )
            else:
                border_color = "#ef4444"
                bg_style = (
                    "background: linear-gradient(145deg, rgba(239,68,68,0.06) 0%, "
                    "rgba(15,23,42,0.9) 100%);"
                )
                status_html = (
                    '<span style="display:inline-flex; align-items:center; gap:4px; '
                    'padding:2px 8px; border-radius:10px; background:rgba(239,68,68,0.15); '
                    'border:1px solid rgba(239,68,68,0.3);">'
                    '<span style="font-size:0.7rem; color:#ef4444; font-weight:600;">'
                    'Not Found</span></span>'
                )

            with ui.card().classes("glow-card px-5 py-4 min-w-[240px]").style(
                f"{bg_style} border-left: 3px solid {border_color};"
            ):
                with ui.row().classes("items-center gap-3"):
                    ui.html(emoji)
                    with ui.column().classes("gap-0"):
                        ui.label(display).classes("font-semibold text-lg")
                        ui.label(desc).classes("text-[0.65rem] text-gray-500")
                with ui.row().classes("mt-2"):
                    ui.html(status_html)

    if status.get("last_training_date"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("schedule").classes("text-gray-500 text-sm")
            ui.label(f"Last training: {status['last_training_date']}").classes(
                "text-sm text-gray-400"
            )

    # ── Accuracy metrics ─────────────────────────────────
    with ui.row().classes("section-header w-full"):
        ui.icon("leaderboard").classes("text-blue-400")
        ui.label("Accuracy Metrics").classes("text-xl font-semibold")

    summary_path = settings.project_root / "results" / "summary.csv"
    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
            _render_df(summary_df)
        except Exception as exc:
            _error_card(f"Could not load summary: {exc}")
    else:
        with ui.card().classes("glow-card w-full px-6 py-4").style(
            "border-left: 4px solid #f59e0b;"
        ):
            with ui.row().classes("items-center gap-3"):
                ui.icon("info").classes("text-amber-400")
                with ui.column().classes("gap-0"):
                    ui.label("No summary.csv found").classes("text-gray-300 font-semibold")
                    ui.label("Run uv run better model train to generate metrics.").classes(
                        "text-sm text-gray-500"
                    )

    # ── Model file details ────────────────────────────────
    details = status.get("model_details", {})
    if details:
        with ui.row().classes("section-header w-full"):
            ui.icon("tune").classes("text-blue-400")
            ui.label("Model Details").classes("text-xl font-semibold")

        detail_rows = []
        for name, info in details.items():
            row = {"model": name}
            if "avg_accuracy" in info:
                row["accuracy"] = f"{info['avg_accuracy']:.1%}"
            if "avg_log_loss" in info:
                row["log_loss"] = f"{info['avg_log_loss']:.4f}"
            if "avg_brier" in info:
                row["brier"] = f"{info['avg_brier']:.4f}"
            if "last_modified" in info:
                row["last_modified"] = info["last_modified"]
            detail_rows.append(row)

        if detail_rows:
            cols = [
                {
                    "name": k,
                    "label": k.replace("_", " ").title(),
                    "field": k,
                    "align": "left" if k == "model" else "right",
                    "sortable": True,
                }
                for k in detail_rows[0].keys()
            ]
            ui.table(columns=cols, rows=detail_rows, row_key="model").classes(
                "w-full max-w-3xl"
            ).props("flat bordered dense")

    # ── Storage ───────────────────────────────────────────
    with ui.row().classes("section-header w-full"):
        ui.icon("storage").classes("text-blue-400")
        ui.label("Storage").classes("text-xl font-semibold")

    models_dir = settings.models_dir
    if models_dir.exists():
        storage_rows = []
        total_size = 0
        for model_dir in sorted(models_dir.iterdir()):
            if model_dir.is_dir():
                dir_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                file_count = sum(1 for f in model_dir.rglob("*") if f.is_file())
                total_size += dir_size
                storage_rows.append({
                    "model": model_dir.name,
                    "size": _format_size(dir_size),
                    "files": str(file_count),
                })

        if storage_rows:
            storage_rows.append({
                "model": "TOTAL",
                "size": _format_size(total_size),
                "files": "",
            })

            storage_cols = [
                {"name": "model", "label": "Model", "field": "model", "align": "left"},
                {"name": "size", "label": "Size", "field": "size", "align": "right"},
                {"name": "files", "label": "Files", "field": "files", "align": "right"},
            ]
            ui.table(columns=storage_cols, rows=storage_rows, row_key="model").classes(
                "w-full max-w-xl"
            ).props("flat bordered dense")

    # ── Fold results ──────────────────────────────────────
    fold_path = settings.project_root / "results" / "fold_results.csv"
    if fold_path.exists():
        with ui.expansion("Per-Fold Results", icon="unfold_more").classes("w-full"):
            try:
                fold_df = pd.read_csv(fold_path)
                _render_df(fold_df)
            except Exception as exc:
                _error_card(f"Could not load fold results: {exc}")


def _render_df(df: pd.DataFrame) -> None:
    """Render a pandas DataFrame as a NiceGUI table."""
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


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _error_card(msg: str) -> None:
    """Show an error in a styled card."""
    with ui.card().classes("glow-card px-4 py-3").style("border-left: 3px solid #ef4444;"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("error_outline").classes("text-red-400 text-sm")
            ui.label(msg).classes("text-red-300 text-sm")
