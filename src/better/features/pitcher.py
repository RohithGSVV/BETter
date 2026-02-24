"""Pitcher matchup features for starting pitchers.

Populates the ``pitcher_features`` table.  Season-level stats come from
``player_pitching``; days-rest is computed from the ``games`` table once
starting-pitcher IDs have been resolved.

Columns that require per-start pitching lines (``ip_last_30d``,
``game_score_avg_5``) are set to NULL — they will be populated once
Statcast / box-score data is ingested.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from tqdm import tqdm

from better.config import settings
from better.data.db import fetch_df, get_connection
from better.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_pitcher_season_stats(start_year: int, end_year: int) -> pd.DataFrame:
    """Season-level stats for starters, keyed by MLB AM player_id.

    ``player_pitching.player_id`` is a **FanGraphs ID**, while
    ``games.home_sp_id`` is an **MLB AM ID**.  We join through
    ``player_ids`` to map FanGraphs → MLB AM so the lookup key
    matches the games table.
    """
    return fetch_df(
        """
        SELECT pi.player_id AS pitcher_id,   -- MLB AM ID
               pp.season, pp.team,
               pp.siera, pp.fip, pp.xfip, pp.stuff_plus,
               pp.k_pct, pp.bb_pct, pp.k_minus_bb_pct,
               pp.throws
        FROM player_pitching pp
        INNER JOIN player_ids pi
            ON pp.player_id = pi.fangraphs_id
        WHERE pp.season BETWEEN ? AND ?
          AND pp.is_starter = TRUE
          AND pi.player_id IS NOT NULL
        """,
        [start_year, end_year],
    )


def _load_sp_appearances(start_year: int, end_year: int) -> pd.DataFrame:
    """Every starting-pitcher appearance (from games table)."""
    return fetch_df(
        """
        SELECT game_date, season, home_sp_id AS pitcher_id, 'home' AS side
        FROM games
        WHERE home_sp_id IS NOT NULL AND season BETWEEN ? AND ?
          AND is_postseason = FALSE
        UNION ALL
        SELECT game_date, season, away_sp_id AS pitcher_id, 'away' AS side
        FROM games
        WHERE away_sp_id IS NOT NULL AND season BETWEEN ? AND ?
          AND is_postseason = FALSE
        ORDER BY game_date
        """,
        [start_year, end_year, start_year, end_year],
    )


def _compute_days_rest(appearances: pd.DataFrame) -> pd.DataFrame:
    """Add ``days_rest`` = days since this pitcher's previous start.

    Returns the same DataFrame with an extra column.
    """
    appearances = appearances.sort_values(["pitcher_id", "game_date"]).copy()
    appearances["prev_start"] = appearances.groupby("pitcher_id")["game_date"].shift(1)
    appearances["days_rest"] = (
        pd.to_datetime(appearances["game_date"]) - pd.to_datetime(appearances["prev_start"])
    ).dt.days
    # First start of the season → NULL
    season_changed = appearances["season"] != appearances.groupby("pitcher_id")["season"].shift(1)
    appearances.loc[season_changed, "days_rest"] = None
    return appearances


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PITCHER_FEAT_COLS = [
    "pitcher_id", "as_of_date",
    "siera", "fip", "xfip", "stuff_plus",
    "k_pct", "bb_pct", "k_minus_bb_pct",
    "ip_last_30d", "days_rest", "game_score_avg_5",
    "throws",
]


def build_pitcher_features(start_year: int, end_year: int) -> int:
    """Build ``pitcher_features`` table from season stats + games table.

    Returns the number of rows inserted.
    """
    conn = get_connection()

    # 1. Season-level pitcher stats (keyed by pitcher_id × season)
    season_stats = _load_pitcher_season_stats(start_year, end_year)
    if season_stats.empty:
        log.warning("no_pitcher_season_stats")
        return 0

    # Build fast lookup: (pitcher_id, season) → stats row
    stat_idx = season_stats.set_index(["pitcher_id", "season"])

    # 2. All SP appearances with days rest
    appearances = _load_sp_appearances(start_year, end_year)
    if appearances.empty:
        log.warning("no_sp_appearances")
        return 0
    appearances = _compute_days_rest(appearances)

    # 3. For each appearance, look up season stats (or prior season)
    rows: list[dict] = []
    for tup in tqdm(appearances.itertuples(index=False), total=len(appearances), desc="Pitcher features"):
        pid = int(tup.pitcher_id)
        season = int(tup.season)

        # Try current season first, then prior
        stats = None
        for s in [season, season - 1]:
            try:
                stats = stat_idx.loc[(pid, s)]
                break
            except KeyError:
                continue

        if stats is None:
            continue

        rows.append({
            "pitcher_id": pid,
            "as_of_date": tup.game_date,
            "siera": _safe_float(stats, "siera"),
            "fip": _safe_float(stats, "fip"),
            "xfip": _safe_float(stats, "xfip"),
            "stuff_plus": _safe_float(stats, "stuff_plus"),
            "k_pct": _safe_float(stats, "k_pct"),
            "bb_pct": _safe_float(stats, "bb_pct"),
            "k_minus_bb_pct": _safe_float(stats, "k_minus_bb_pct"),
            "ip_last_30d": None,       # Statcast enrichment
            "days_rest": tup.days_rest if pd.notna(tup.days_rest) else None,
            "game_score_avg_5": None,  # Statcast enrichment
            "throws": _safe_str(stats, "throws"),
        })

    if not rows:
        log.warning("no_pitcher_features_generated")
        return 0

    df = pd.DataFrame(rows)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["days_rest"] = df["days_rest"].astype("Int64")

    # Deduplicate: if a pitcher starts twice on the same date (very rare),
    # keep the first
    df = df.drop_duplicates(subset=["pitcher_id", "as_of_date"], keep="first")

    # Write to DuckDB
    conn.execute(
        "DELETE FROM pitcher_features WHERE as_of_date BETWEEN ? AND ?",
        [f"{start_year}-01-01", f"{end_year}-12-31"],
    )
    cols = ", ".join(_PITCHER_FEAT_COLS)
    conn.execute(f"INSERT INTO pitcher_features ({cols}) SELECT {cols} FROM df")

    log.info("pitcher_features_built", rows=len(df), pitchers=df["pitcher_id"].nunique())
    return len(df)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _safe_float(stats, col: str) -> float | None:
    """Extract a float from a stats row (Series or named-tuple)."""
    try:
        v = stats[col] if isinstance(stats, pd.Series) else getattr(stats, col, None)
        return float(v) if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def _safe_str(stats, col: str) -> str | None:
    try:
        v = stats[col] if isinstance(stats, pd.Series) else getattr(stats, col, None)
        return str(v) if pd.notna(v) and v else None
    except (TypeError, ValueError):
        return None
