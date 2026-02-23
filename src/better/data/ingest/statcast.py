"""Statcast pitch-level data ingestion via pybaseball.

Loads pitch-by-pitch tracking data from Baseball Savant (2015+).
Used for: Transformer pretraining, Monte Carlo distributions, advanced metrics.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from tqdm import tqdm

from better.config import settings
from better.data.db import get_connection
from better.utils.dates import chunk_date_range, season_date_range
from better.utils.logging import get_logger

log = get_logger(__name__)

# Key Statcast columns we keep (from ~90 available)
STATCAST_KEEP_COLS = [
    "game_pk", "at_bat_number", "pitch_number", "game_date",
    "pitcher", "batter", "player_name", "batter_name",
    "pitch_type", "release_speed", "release_spin_rate",
    "plate_x", "plate_z",
    "launch_speed", "launch_angle", "hit_distance_sc",
    "events", "description",
    "zone", "stand", "p_throws",
    "home_team", "away_team",
    "inning", "inning_topbot",
    "outs_when_up", "balls", "strikes",
    "on_1b", "on_2b", "on_3b",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "delta_run_exp",
]


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Statcast columns to match our schema."""
    rename_map = {
        "pitcher": "pitcher_id",
        "batter": "batter_id",
        "player_name": "pitcher_name",
        "batter_name": "batter_name",
        "release_spin_rate": "release_spin_rate",
        "hit_distance_sc": "hit_distance",
    }
    df = df.rename(columns=rename_map)
    return df


def ingest_statcast_season(year: int, chunk_days: int = 5) -> int:
    """Load one season of Statcast data into DuckDB.

    Chunks queries into small date windows to respect the 25K row API limit.
    Returns the number of pitches loaded.
    """
    from pybaseball import statcast, cache

    cache.enable()

    conn = get_connection()
    start, end = season_date_range(year)
    chunks = chunk_date_range(start, end, chunk_days)
    total_pitches = 0

    # Clear existing data for this season
    conn.execute(
        "DELETE FROM statcast_pitches WHERE game_date BETWEEN ? AND ?",
        [start.isoformat(), end.isoformat()],
    )

    for chunk_start, chunk_end in tqdm(
        chunks, desc=f"Statcast {year}", leave=False
    ):
        try:
            raw = statcast(
                start_dt=chunk_start.strftime("%Y-%m-%d"),
                end_dt=chunk_end.strftime("%Y-%m-%d"),
            )
            if raw is None or raw.empty:
                continue

            # Keep only columns we need
            available_cols = [c for c in STATCAST_KEEP_COLS if c in raw.columns]
            df = raw[available_cols].copy()
            df = _rename_columns(df)

            # Ensure proper types
            df["game_date"] = pd.to_datetime(df["game_date"])
            for int_col in ["game_pk", "at_bat_number", "pitch_number"]:
                if int_col in df.columns:
                    df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")

            conn.execute("INSERT INTO statcast_pitches SELECT * FROM df")
            total_pitches += len(df)

        except Exception as e:
            log.warning(
                "statcast_chunk_failed",
                start=str(chunk_start),
                end=str(chunk_end),
                error=str(e),
            )
            continue

    log.info("statcast_season_loaded", year=year, pitches=total_pitches)
    return total_pitches


def ingest_statcast(
    start_year: int | None = None,
    end_year: int | None = None,
    chunk_days: int = 5,
) -> int:
    """Load multiple seasons of Statcast data.

    Returns total pitches loaded across all seasons.
    """
    start = start_year or settings.statcast_start_year
    end = end_year or settings.train_end_year
    total = 0

    for year in tqdm(range(start, end + 1), desc="Statcast"):
        total += ingest_statcast_season(year, chunk_days)

    log.info("statcast_complete", total_pitches=total)
    return total
