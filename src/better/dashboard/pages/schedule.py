"""Schedule page — upcoming games with styled tables and team colors."""

from __future__ import annotations

from datetime import datetime

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
    "ATH": ("#003831", "#efb21e"),
}

TYPE_COLORS: dict[str, str] = {
    "Spring": "blue-grey",
    "Regular": "primary",
    "Wild Card": "orange",
    "Division": "purple",
    "League": "deep-purple",
    "World Series": "amber",
    "Exhibition": "grey",
}


def render(svc: PredictionService) -> None:
    """Render the Schedule page with upcoming games in styled tables."""

    # ── Page header ──────────────────────────────────────────────────
    with ui.row().classes("items-center gap-3 w-full"):
        ui.icon("event").classes("text-3xl text-blue-400")
        with ui.column().classes("gap-0"):
            ui.label("Upcoming Schedule").classes("text-2xl font-bold")
            ui.label("Next 5 days of MLB games").classes(
                "text-sm text-gray-400"
            )

    try:
        upcoming = svc.get_upcoming_schedule(days=5)
    except Exception:
        upcoming = {}

    if not upcoming:
        with ui.card().classes("w-full glow-card px-8 py-6").style(
            "border-left: 4px solid #3b82f6;"
        ):
            with ui.row().classes("items-center gap-4"):
                ui.html(
                    '<span style="font-size:2.5rem; opacity:0.5;">&#9918;</span>'
                )
                with ui.column().classes("gap-1"):
                    ui.label("No upcoming games found").classes(
                        "text-lg font-semibold"
                    )
                    ui.label(
                        "This can happen during the off-season."
                    ).classes("text-gray-400 text-sm")
        return

    type_labels = {
        "R": "Regular", "S": "Spring", "F": "Wild Card",
        "D": "Division", "L": "League", "W": "World Series",
        "E": "Exhibition",
    }

    for date_str, games in sorted(upcoming.items()):
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            day_label = dt.strftime("%A, %B %d")
        except ValueError:
            day_label = date_str

        # ── Day card ─────────────────────────────────────────
        with ui.card().classes("glow-card w-full px-5 py-4 mt-2"):
            # Day header
            with ui.row().classes("items-center gap-3 w-full mb-3"):
                ui.icon("calendar_today").classes("text-blue-400 text-lg")
                ui.label(day_label).classes(
                    "text-lg font-semibold text-gray-200"
                )
                ui.badge(f"{len(games)} games").props("color=primary outline")

            # Table columns
            columns = [
                {"name": "matchup", "label": "Matchup", "field": "matchup",
                 "align": "left", "sortable": True},
                {"name": "time", "label": "Time", "field": "time",
                 "align": "center"},
                {"name": "away_sp", "label": "Away SP", "field": "away_sp",
                 "align": "left"},
                {"name": "home_sp", "label": "Home SP", "field": "home_sp",
                 "align": "left"},
                {"name": "venue", "label": "Venue", "field": "venue",
                 "align": "left"},
                {"name": "type", "label": "Type", "field": "type",
                 "align": "center"},
            ]

            rows = []
            for g in games:
                away = g.get("away_team", "")
                home = g.get("home_team", "")
                game_type = g.get("game_type", "R")
                rows.append({
                    "matchup": f"{away} @ {home}",
                    "time": g.get("game_time", "TBD"),
                    "away_sp": g.get("away_sp_name", "") or "TBD",
                    "home_sp": g.get("home_sp_name", "") or "TBD",
                    "venue": g.get("venue", ""),
                    "type": type_labels.get(game_type, game_type),
                    # Extra data for slot rendering
                    "away_color": TEAM_COLORS.get(
                        away, ("#3b82f6",)
                    )[0],
                    "home_color": TEAM_COLORS.get(
                        home, ("#3b82f6",)
                    )[0],
                })

            table = ui.table(
                columns=columns, rows=rows, row_key="matchup",
            ).classes("w-full schedule-table").props("flat dense")

            # ── Custom slot: matchup with team color dots ────
            table.add_slot(
                "body-cell-matchup",
                '''
                <q-td :props="props">
                    <div style="display:flex; align-items:center; gap:6px;">
                        <span :style="'width:8px; height:8px; border-radius:50%;'
                              + 'display:inline-block; background:'
                              + props.row.away_color
                              + '; box-shadow: 0 0 4px '
                              + props.row.away_color + '60;'"></span>
                        <span style="font-weight:500;">
                            {{ props.value }}
                        </span>
                        <span :style="'width:8px; height:8px; border-radius:50%;'
                              + 'display:inline-block; background:'
                              + props.row.home_color
                              + '; box-shadow: 0 0 4px '
                              + props.row.home_color + '60;'"></span>
                    </div>
                </q-td>
                ''',
            )

            # ── Custom slot: type as colored badge ───────────
            table.add_slot(
                "body-cell-type",
                '''
                <q-td :props="props">
                    <q-badge
                        :color="props.value === 'Spring' ? 'blue-grey'
                              : props.value === 'Regular' ? 'primary'
                              : props.value === 'Wild Card' ? 'orange'
                              : 'grey'"
                        outline
                        :label="props.value"
                        style="font-size: 0.65rem;"
                    />
                </q-td>
                ''',
            )

            # ── Custom slot: time with clock icon ────────────
            table.add_slot(
                "body-cell-time",
                '''
                <q-td :props="props">
                    <div style="display:flex; align-items:center; gap:4px;
                                justify-content:center;">
                        <span style="font-size:0.6rem; opacity:0.5;">&#128339;</span>
                        <span style="font-size:0.8rem; font-weight:500;">
                            {{ props.value }}
                        </span>
                    </div>
                </q-td>
                ''',
            )
