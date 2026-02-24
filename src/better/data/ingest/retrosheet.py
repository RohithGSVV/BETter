"""Retrosheet game log ingestion.

Downloads and parses Retrosheet game logs into the `games` table.
Source: https://www.retrosheet.org/gamelogs/
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

from better.config import settings
from better.constants import RETROSHEET_TEAM_MAP
from better.data.db import get_connection
from better.utils.logging import get_logger

log = get_logger(__name__)

# Retrosheet game log column names (subset of ~170 fields we use)
# Full spec: https://www.retrosheet.org/gamelogs/glfields.txt
GAMELOG_COLUMNS = [
    "date", "game_num", "day_of_week", "visiting_team", "visiting_league",
    "visiting_game_num", "home_team", "home_league", "home_game_num",
    "visiting_score", "home_score", "length_outs", "day_night", "completion",
    "forfeit", "protest", "park_id", "attendance", "time_of_game",
    "visiting_line_score", "home_line_score",
    "visiting_ab", "visiting_h", "visiting_2b", "visiting_3b", "visiting_hr",
    "visiting_rbi", "visiting_sh", "visiting_sf", "visiting_hbp",
    "visiting_bb", "visiting_ibb", "visiting_so", "visiting_sb", "visiting_cs",
    "visiting_gdp", "visiting_ci", "visiting_lob", "visiting_pitchers_used",
    "visiting_individual_er", "visiting_team_er", "visiting_wp", "visiting_bk",
    "visiting_po", "visiting_a", "visiting_e", "visiting_passed_balls",
    "visiting_dp", "visiting_tp",
    "home_ab", "home_h", "home_2b", "home_3b", "home_hr",
    "home_rbi", "home_sh", "home_sf", "home_hbp",
    "home_bb", "home_ibb", "home_so", "home_sb", "home_cs",
    "home_gdp", "home_ci", "home_lob", "home_pitchers_used",
    "home_individual_er", "home_team_er", "home_wp", "home_bk",
    "home_po", "home_a", "home_e", "home_passed_balls",
    "home_dp", "home_tp",
    "hp_umpire_id", "hp_umpire_name",
    "1b_umpire_id", "1b_umpire_name",
    "2b_umpire_id", "2b_umpire_name",
    "3b_umpire_id", "3b_umpire_name",
    "lf_umpire_id", "lf_umpire_name",
    "rf_umpire_id", "rf_umpire_name",
    "visiting_manager_id", "visiting_manager_name",
    "home_manager_id", "home_manager_name",
    "winning_pitcher_id", "winning_pitcher_name",
    "losing_pitcher_id", "losing_pitcher_name",
    "save_pitcher_id", "save_pitcher_name",
    "gw_rbi_batter_id", "gw_rbi_batter_name",
    "visiting_starter_id", "visiting_starter_name",
    "home_starter_id", "home_starter_name",
    # Visiting lineup (9 batters x id+name+pos = 27 columns)
    *[f"visiting_batter_{i}_{f}" for i in range(1, 10) for f in ("id", "name", "pos")],
    # Home lineup (9 batters x id+name+pos = 27 columns)
    *[f"home_batter_{i}_{f}" for i in range(1, 10) for f in ("id", "name", "pos")],
    "additional_info", "acquisition_info",
]

BASE_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"


def _normalize_team(code: str) -> str:
    """Map Retrosheet team code to standard abbreviation."""
    return RETROSHEET_TEAM_MAP.get(code, code)


def download_gamelog(year: int) -> pd.DataFrame:
    """Download and parse a single year's Retrosheet game log.

    Returns a DataFrame with standardized columns ready for insertion.
    """
    url = BASE_URL.format(year=year)
    log.info("downloading_retrosheet", year=year, url=url)

    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # The zip contains a single file like GL2024.TXT
        names = zf.namelist()
        txt_file = [n for n in names if n.upper().endswith(".TXT")][0]
        with zf.open(txt_file) as f:
            # Retrosheet files are CSV with no header
            num_cols = len(GAMELOG_COLUMNS)
            raw = pd.read_csv(
                f,
                header=None,
                encoding="latin-1",
            )
            # Trim or pad columns to match our expected schema
            if raw.shape[1] >= num_cols:
                raw = raw.iloc[:, :num_cols]
            raw.columns = GAMELOG_COLUMNS[: raw.shape[1]]

    log.info("parsed_retrosheet", year=year, games=len(raw))
    return raw


def transform_gamelog(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Transform raw Retrosheet data into our games table schema."""
    df = pd.DataFrame()

    # Parse date: format is YYYYMMDD — DuckDB DATE column accepts pandas datetime64[ns]
    df["game_date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d")
    df["season"] = year

    # Team codes
    df["home_team"] = raw["home_team"].map(_normalize_team)
    df["away_team"] = raw["visiting_team"].map(_normalize_team)

    # Scores
    df["home_score"] = pd.to_numeric(raw["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(raw["visiting_score"], errors="coerce").astype("Int64")
    df["home_win"] = df["home_score"] > df["away_score"]

    # Scores as runs scored/allowed
    df["home_runs_scored"] = df["home_score"]
    df["home_runs_allowed"] = df["away_score"]
    df["away_runs_scored"] = df["away_score"]
    df["away_runs_allowed"] = df["home_score"]

    # Batting stats
    df["home_hits"] = pd.to_numeric(raw.get("home_h", 0), errors="coerce").astype("Int64")
    df["away_hits"] = pd.to_numeric(raw.get("visiting_h", 0), errors="coerce").astype("Int64")
    df["home_errors"] = pd.to_numeric(raw.get("home_e", 0), errors="coerce").astype("Int64")
    df["away_errors"] = pd.to_numeric(raw.get("visiting_e", 0), errors="coerce").astype("Int64")
    df["home_walks"] = pd.to_numeric(raw.get("home_bb", 0), errors="coerce").astype("Int64")
    df["away_walks"] = pd.to_numeric(raw.get("visiting_bb", 0), errors="coerce").astype("Int64")
    df["home_strikeouts"] = pd.to_numeric(raw.get("home_so", 0), errors="coerce").astype("Int64")
    df["away_strikeouts"] = pd.to_numeric(
        raw.get("visiting_so", 0), errors="coerce"
    ).astype("Int64")
    df["home_home_runs"] = pd.to_numeric(raw.get("home_hr", 0), errors="coerce").astype("Int64")
    df["away_home_runs"] = pd.to_numeric(
        raw.get("visiting_hr", 0), errors="coerce"
    ).astype("Int64")

    # Starting pitchers
    if "home_starter_id" in raw.columns:
        df["home_sp_name"] = raw.get("home_starter_name", "")
        df["away_sp_name"] = raw.get("visiting_starter_name", "")
    else:
        df["home_sp_name"] = ""
        df["away_sp_name"] = ""

    # Park and game info
    df["park_id"] = raw.get("park_id", "")
    df["attendance"] = pd.to_numeric(raw.get("attendance", 0), errors="coerce").astype("Int64")
    df["game_duration_minutes"] = pd.to_numeric(
        raw.get("time_of_game", 0), errors="coerce"
    ).astype("Int64")
    df["day_night"] = raw.get("day_night", "N")

    # Innings played (length_outs / 3 per team, but reported as total outs)
    length_outs = pd.to_numeric(raw.get("length_outs", 54), errors="coerce")
    df["innings_played"] = (length_outs / 6).round().astype("Int64")  # rough approximation

    df["is_postseason"] = False
    df["data_source"] = "retrosheet"

    # Generate a game_pk: YYYYMMDD * 1000 + sequential row index within the day
    # This guarantees uniqueness regardless of doubleheaders or same-day collisions
    date_int = df["game_date"].dt.strftime("%Y%m%d").astype(int)
    day_seq = df.groupby("game_date").cumcount()  # 0-based sequential within each date
    df["game_pk"] = (date_int * 1000 + day_seq).astype("int64")

    # Starting pitcher IDs — store Retrosheet string IDs for later resolution
    df["home_sp_id"] = None
    df["away_sp_id"] = None
    if "home_starter_id" in raw.columns:
        df["home_sp_retrosheet_id"] = raw["home_starter_id"].astype(str).str.strip()
    else:
        df["home_sp_retrosheet_id"] = None
    if "visiting_starter_id" in raw.columns:
        df["away_sp_retrosheet_id"] = raw["visiting_starter_id"].astype(str).str.strip()
    else:
        df["away_sp_retrosheet_id"] = None

    return df


def ingest_retrosheet(
    start_year: int | None = None,
    end_year: int | None = None,
) -> int:
    """Download and load Retrosheet game logs into DuckDB.

    Returns the total number of games loaded.
    """
    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()
    total = 0

    for year in tqdm(range(start, end + 1), desc="Retrosheet"):
        try:
            raw = download_gamelog(year)
            df = transform_gamelog(raw, year)

            # Insert into DuckDB with explicit column names to avoid positional mismatch
            conn.execute("DELETE FROM games WHERE season = ? AND data_source = 'retrosheet'", [year])
            cols = ", ".join(df.columns.tolist())
            conn.execute(f"INSERT INTO games ({cols}) SELECT {cols} FROM df")
            total += len(df)
            log.info("loaded_retrosheet", year=year, games=len(df))
        except Exception as e:
            log.error("retrosheet_failed", year=year, error=str(e))
            continue

    # Resolve Retrosheet starter IDs → integer player IDs via player_ids table
    if total > 0:
        _resolve_starter_ids(conn)

    log.info("retrosheet_complete", total_games=total)
    return total


def _resolve_starter_ids(conn) -> None:
    """Backfill home_sp_id and away_sp_id from Retrosheet starter string IDs."""
    updated_home = conn.execute("""
        UPDATE games SET home_sp_id = p.player_id
        FROM player_ids p
        WHERE games.home_sp_retrosheet_id = p.retrosheet_id
          AND games.home_sp_retrosheet_id IS NOT NULL
          AND games.home_sp_retrosheet_id != ''
          AND games.home_sp_id IS NULL
    """).fetchone()

    updated_away = conn.execute("""
        UPDATE games SET away_sp_id = p.player_id
        FROM player_ids p
        WHERE games.away_sp_retrosheet_id = p.retrosheet_id
          AND games.away_sp_retrosheet_id IS NOT NULL
          AND games.away_sp_retrosheet_id != ''
          AND games.away_sp_id IS NULL
    """).fetchone()

    # Log coverage
    result = conn.execute("""
        SELECT COUNT(*) as total,
               COUNT(home_sp_id) as home_resolved,
               COUNT(away_sp_id) as away_resolved
        FROM games
    """).fetchone()
    log.info(
        "starter_ids_resolved",
        total_games=result[0],
        home_sp_resolved=result[1],
        away_sp_resolved=result[2],
        home_pct=round(result[1] / result[0] * 100, 1) if result[0] > 0 else 0,
        away_pct=round(result[2] / result[0] * 100, 1) if result[0] > 0 else 0,
    )
