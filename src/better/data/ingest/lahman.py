"""Team and player statistics ingestion via pybaseball (FanGraphs).

Uses pybaseball's FanGraphs scrapers for team and player stats.
This gives us advanced metrics (wRC+, FIP, SIERA, Stuff+, WAR) directly
without needing the Lahman CSV files, which have moved to an inaccessible location.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from better.config import settings
from better.data.db import get_connection
from better.utils.logging import get_logger

log = get_logger(__name__)


def ingest_teams(start_year: int | None = None, end_year: int | None = None) -> int:
    """Load FanGraphs team batting and pitching stats into DuckDB."""
    from pybaseball import team_batting, team_pitching

    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()

    all_batting: list[pd.DataFrame] = []
    all_pitching: list[pd.DataFrame] = []

    for year in tqdm(range(start, end + 1), desc="Team Stats"):
        try:
            tb = team_batting(year)
            if tb is not None and not tb.empty:
                tb = tb.copy()
                tb["season"] = year
                all_batting.append(tb)
        except Exception as e:
            log.warning("team_batting_failed", year=year, error=str(e))

        try:
            tp = team_pitching(year)
            if tp is not None and not tp.empty:
                tp = tp.copy()
                tp["season"] = year
                all_pitching.append(tp)
        except Exception as e:
            log.warning("team_pitching_failed", year=year, error=str(e))

    # --- Team batting ---
    if all_batting:
        raw_tb = pd.concat(all_batting, ignore_index=True)
        batting = pd.DataFrame()
        batting["team"] = raw_tb["Team"]
        batting["season"] = raw_tb["season"].astype(int)
        batting["games"] = pd.to_numeric(raw_tb.get("G"), errors="coerce").astype("Int64")
        batting["plate_appearances"] = pd.to_numeric(raw_tb.get("PA"), errors="coerce").astype("Int64")
        batting["at_bats"] = pd.to_numeric(raw_tb.get("AB"), errors="coerce").astype("Int64")
        batting["hits"] = pd.to_numeric(raw_tb.get("H"), errors="coerce").astype("Int64")
        batting["doubles"] = pd.to_numeric(raw_tb.get("2B"), errors="coerce").astype("Int64")
        batting["triples"] = pd.to_numeric(raw_tb.get("3B"), errors="coerce").astype("Int64")
        batting["home_runs"] = pd.to_numeric(raw_tb.get("HR"), errors="coerce").astype("Int64")
        batting["runs"] = pd.to_numeric(raw_tb.get("R"), errors="coerce").astype("Int64")
        batting["rbi"] = pd.to_numeric(raw_tb.get("RBI"), errors="coerce").astype("Int64")
        batting["walks"] = pd.to_numeric(raw_tb.get("BB"), errors="coerce").astype("Int64")
        batting["strikeouts"] = pd.to_numeric(raw_tb.get("SO"), errors="coerce").astype("Int64")
        batting["stolen_bases"] = pd.to_numeric(raw_tb.get("SB"), errors="coerce").astype("Int64")
        batting["batting_avg"] = pd.to_numeric(raw_tb.get("AVG"), errors="coerce").round(3)
        batting["obp"] = pd.to_numeric(raw_tb.get("OBP"), errors="coerce").round(3)
        batting["slg"] = pd.to_numeric(raw_tb.get("SLG"), errors="coerce").round(3)
        batting["ops"] = pd.to_numeric(raw_tb.get("OPS"), errors="coerce").round(3)
        batting["woba"] = pd.to_numeric(raw_tb.get("wOBA"), errors="coerce").round(3)
        batting["wrc_plus"] = pd.to_numeric(raw_tb.get("wRC+"), errors="coerce").round(1)
        batting["iso"] = pd.to_numeric(raw_tb.get("ISO"), errors="coerce").round(3)
        batting["babip"] = pd.to_numeric(raw_tb.get("BABIP"), errors="coerce").round(3)
        batting = batting.dropna(subset=["team", "season"])
        conn.execute("DELETE FROM team_batting WHERE season BETWEEN ? AND ?", [start, end])
        conn.execute("INSERT INTO team_batting SELECT * FROM batting")
        log.info("team_batting_loaded", rows=len(batting))

    # --- Team pitching ---
    if all_pitching:
        raw_tp = pd.concat(all_pitching, ignore_index=True)
        pitching = pd.DataFrame()
        pitching["team"] = raw_tp["Team"]
        pitching["season"] = raw_tp["season"].astype(int)
        pitching["games"] = pd.to_numeric(raw_tp.get("G"), errors="coerce").astype("Int64")
        pitching["innings_pitched"] = pd.to_numeric(raw_tp.get("IP"), errors="coerce").round(1)
        pitching["era"] = pd.to_numeric(raw_tp.get("ERA"), errors="coerce").round(2)
        pitching["fip"] = pd.to_numeric(raw_tp.get("FIP"), errors="coerce").round(2)
        pitching["xfip"] = pd.to_numeric(raw_tp.get("xFIP"), errors="coerce").round(2)
        pitching["siera"] = pd.to_numeric(raw_tp.get("SIERA"), errors="coerce").round(2)
        pitching["whip"] = pd.to_numeric(raw_tp.get("WHIP"), errors="coerce").round(3)
        pitching["k_per_9"] = pd.to_numeric(raw_tp.get("K/9"), errors="coerce").round(2)
        pitching["bb_per_9"] = pd.to_numeric(raw_tp.get("BB/9"), errors="coerce").round(2)
        pitching["hr_per_9"] = pd.to_numeric(raw_tp.get("HR/9"), errors="coerce").round(2)
        k_pct = pd.to_numeric(raw_tp.get("K%"), errors="coerce")
        bb_pct = pd.to_numeric(raw_tp.get("BB%"), errors="coerce")
        pitching["k_pct"] = k_pct.round(3)
        pitching["bb_pct"] = bb_pct.round(3)
        pitching["k_minus_bb_pct"] = (k_pct - bb_pct).round(3)
        pitching["avg_against"] = pd.to_numeric(raw_tp.get("AVG"), errors="coerce").round(3)
        pitching["runs_allowed"] = pd.to_numeric(raw_tp.get("R"), errors="coerce").astype("Int64")
        pitching["earned_runs"] = pd.to_numeric(raw_tp.get("ER"), errors="coerce").astype("Int64")
        pitching = pitching.dropna(subset=["team", "season"])
        conn.execute("DELETE FROM team_pitching WHERE season BETWEEN ? AND ?", [start, end])
        conn.execute("INSERT INTO team_pitching SELECT * FROM pitching")
        log.info("team_pitching_loaded", rows=len(pitching))

    return len(all_batting) + len(all_pitching)


def ingest_player_pitching(start_year: int | None = None, end_year: int | None = None) -> int:
    """Load FanGraphs individual pitcher stats into DuckDB."""
    from pybaseball import pitching_stats

    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()
    total = 0

    for year in tqdm(range(start, end + 1), desc="Pitcher Stats"):
        try:
            raw = pitching_stats(year, qual=10)
            if raw is None or raw.empty:
                continue

            df = pd.DataFrame()
            df["player_id"] = pd.to_numeric(raw.get("IDfg"), errors="coerce").astype("Int64")
            df["season"] = year
            df["team"] = raw.get("Team", "")
            df["games"] = pd.to_numeric(raw.get("G"), errors="coerce").astype("Int64")
            df["games_started"] = pd.to_numeric(raw.get("GS"), errors="coerce").astype("Int64")
            df["innings_pitched"] = pd.to_numeric(raw.get("IP"), errors="coerce").round(1)
            df["wins"] = pd.to_numeric(raw.get("W"), errors="coerce").astype("Int64")
            df["losses"] = pd.to_numeric(raw.get("L"), errors="coerce").astype("Int64")
            df["era"] = pd.to_numeric(raw.get("ERA"), errors="coerce").round(2)
            df["fip"] = pd.to_numeric(raw.get("FIP"), errors="coerce").round(2)
            df["xfip"] = pd.to_numeric(raw.get("xFIP"), errors="coerce").round(2)
            df["siera"] = pd.to_numeric(raw.get("SIERA"), errors="coerce").round(2)
            df["whip"] = pd.to_numeric(raw.get("WHIP"), errors="coerce").round(3)
            df["k_per_9"] = pd.to_numeric(raw.get("K/9"), errors="coerce").round(2)
            df["bb_per_9"] = pd.to_numeric(raw.get("BB/9"), errors="coerce").round(2)
            k_pct = pd.to_numeric(raw.get("K%"), errors="coerce")
            bb_pct = pd.to_numeric(raw.get("BB%"), errors="coerce")
            df["k_pct"] = k_pct.round(3)
            df["bb_pct"] = bb_pct.round(3)
            df["k_minus_bb_pct"] = (k_pct - bb_pct).round(3)
            df["war"] = pd.to_numeric(raw.get("WAR"), errors="coerce").round(1)
            df["stuff_plus"] = pd.to_numeric(raw.get("Stuff+"), errors="coerce").round(1)
            df["throws"] = None
            gs = pd.to_numeric(raw.get("GS"), errors="coerce").fillna(0)
            g = pd.to_numeric(raw.get("G"), errors="coerce").replace(0, 1).fillna(1)
            df["is_starter"] = gs > (g * 0.5)
            df = df.dropna(subset=["player_id"])

            conn.execute("DELETE FROM player_pitching WHERE season = ?", [year])
            conn.execute("INSERT INTO player_pitching SELECT * FROM df")
            total += len(df)
        except Exception as e:
            log.warning("pitcher_stats_failed", year=year, error=str(e))

    log.info("player_pitching_loaded", total=total)
    return total


def ingest_player_batting(start_year: int | None = None, end_year: int | None = None) -> int:
    """Load FanGraphs individual batter stats into DuckDB."""
    from pybaseball import batting_stats

    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()
    total = 0

    for year in tqdm(range(start, end + 1), desc="Batter Stats"):
        try:
            raw = batting_stats(year, qual=50)
            if raw is None or raw.empty:
                continue

            df = pd.DataFrame()
            df["player_id"] = pd.to_numeric(raw.get("IDfg"), errors="coerce").astype("Int64")
            df["season"] = year
            df["team"] = raw.get("Team", "")
            df["games"] = pd.to_numeric(raw.get("G"), errors="coerce").astype("Int64")
            df["plate_appearances"] = pd.to_numeric(raw.get("PA"), errors="coerce").astype("Int64")
            df["at_bats"] = pd.to_numeric(raw.get("AB"), errors="coerce").astype("Int64")
            df["hits"] = pd.to_numeric(raw.get("H"), errors="coerce").astype("Int64")
            df["doubles"] = pd.to_numeric(raw.get("2B"), errors="coerce").astype("Int64")
            df["triples"] = pd.to_numeric(raw.get("3B"), errors="coerce").astype("Int64")
            df["home_runs"] = pd.to_numeric(raw.get("HR"), errors="coerce").astype("Int64")
            df["runs"] = pd.to_numeric(raw.get("R"), errors="coerce").astype("Int64")
            df["rbi"] = pd.to_numeric(raw.get("RBI"), errors="coerce").astype("Int64")
            df["walks"] = pd.to_numeric(raw.get("BB"), errors="coerce").astype("Int64")
            df["strikeouts"] = pd.to_numeric(raw.get("SO"), errors="coerce").astype("Int64")
            df["batting_avg"] = pd.to_numeric(raw.get("AVG"), errors="coerce").round(3)
            df["obp"] = pd.to_numeric(raw.get("OBP"), errors="coerce").round(3)
            df["slg"] = pd.to_numeric(raw.get("SLG"), errors="coerce").round(3)
            df["ops"] = pd.to_numeric(raw.get("OPS"), errors="coerce").round(3)
            df["woba"] = pd.to_numeric(raw.get("wOBA"), errors="coerce").round(3)
            df["wrc_plus"] = pd.to_numeric(raw.get("wRC+"), errors="coerce").round(1)
            df["iso"] = pd.to_numeric(raw.get("ISO"), errors="coerce").round(3)
            df["babip"] = pd.to_numeric(raw.get("BABIP"), errors="coerce").round(3)
            df["war"] = pd.to_numeric(raw.get("WAR"), errors="coerce").round(1)
            df["bats"] = None
            df = df.dropna(subset=["player_id"])

            conn.execute("DELETE FROM player_batting WHERE season = ?", [year])
            conn.execute("INSERT INTO player_batting SELECT * FROM df")
            total += len(df)
        except Exception as e:
            log.warning("batter_stats_failed", year=year, error=str(e))

    log.info("player_batting_loaded", total=total)
    return total


def ingest_lahman(start_year: int | None = None, end_year: int | None = None) -> int:
    """Run all team and player stats ingestion. Returns total rows loaded."""
    total = 0
    total += ingest_teams(start_year, end_year)
    total += ingest_player_pitching(start_year, end_year)
    total += ingest_player_batting(start_year, end_year)
    log.info("stats_ingestion_complete", total_rows=total)
    return total
