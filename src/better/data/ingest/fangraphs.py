"""FanGraphs advanced metrics ingestion via pybaseball.

Loads season-level advanced stats: wRC+, FIP, SIERA, Stuff+, WAR, park factors.
Updates the team_batting, team_pitching, player_batting, and player_pitching tables
with advanced metrics not available in the Lahman database.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from better.config import settings
from better.data.db import get_connection
from better.utils.logging import get_logger

log = get_logger(__name__)


def ingest_team_batting_advanced(
    start_year: int | None = None, end_year: int | None = None
) -> int:
    """Update team_batting with FanGraphs advanced metrics (wOBA, wRC+)."""
    from pybaseball import team_batting

    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()
    updated = 0

    for year in tqdm(range(start, end + 1), desc="FG Team Batting"):
        try:
            raw = team_batting(year)
            if raw is None or raw.empty:
                continue

            for _, row in raw.iterrows():
                team = row.get("Team", row.get("Tm", ""))
                woba = row.get("wOBA", None)
                wrc_plus = row.get("wRC+", None)

                if team and (woba is not None or wrc_plus is not None):
                    conn.execute(
                        """
                        UPDATE team_batting
                        SET woba = COALESCE(?, woba),
                            wrc_plus = COALESCE(?, wrc_plus)
                        WHERE team = ? AND season = ?
                        """,
                        [woba, wrc_plus, team, year],
                    )
                    updated += 1
        except Exception as e:
            log.warning("fg_team_batting_failed", year=year, error=str(e))
            continue

    log.info("fg_team_batting_updated", rows=updated)
    return updated


def ingest_team_pitching_advanced(
    start_year: int | None = None, end_year: int | None = None
) -> int:
    """Update team_pitching with FanGraphs advanced metrics (FIP, xFIP, SIERA)."""
    from pybaseball import team_pitching

    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()
    updated = 0

    for year in tqdm(range(start, end + 1), desc="FG Team Pitching"):
        try:
            raw = team_pitching(year)
            if raw is None or raw.empty:
                continue

            for _, row in raw.iterrows():
                team = row.get("Team", row.get("Tm", ""))
                fip = row.get("FIP", None)
                xfip = row.get("xFIP", None)
                siera = row.get("SIERA", None)
                k_pct = row.get("K%", None)
                bb_pct = row.get("BB%", None)

                if team:
                    conn.execute(
                        """
                        UPDATE team_pitching
                        SET fip = COALESCE(?, fip),
                            xfip = COALESCE(?, xfip),
                            siera = COALESCE(?, siera),
                            k_pct = COALESCE(?, k_pct),
                            bb_pct = COALESCE(?, bb_pct),
                            k_minus_bb_pct = COALESCE(? - ?, k_minus_bb_pct)
                        WHERE team = ? AND season = ?
                        """,
                        [fip, xfip, siera, k_pct, bb_pct, k_pct, bb_pct, team, year],
                    )
                    updated += 1
        except Exception as e:
            log.warning("fg_team_pitching_failed", year=year, error=str(e))
            continue

    log.info("fg_team_pitching_updated", rows=updated)
    return updated


def ingest_pitcher_stats(
    start_year: int | None = None, end_year: int | None = None
) -> int:
    """Update player_pitching with FanGraphs advanced pitcher metrics."""
    from pybaseball import pitching_stats

    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()
    updated = 0

    for year in tqdm(range(start, end + 1), desc="FG Pitcher Stats"):
        try:
            raw = pitching_stats(year, qual=10)  # Minimum 10 IP
            if raw is None or raw.empty:
                continue

            for _, row in raw.iterrows():
                fg_id = row.get("IDfg", None)
                if fg_id is None:
                    continue

                fip = row.get("FIP", None)
                xfip = row.get("xFIP", None)
                siera = row.get("SIERA", None)
                war = row.get("WAR", None)
                stuff_plus = row.get("Stuff+", None)
                k_pct = row.get("K%", None)
                bb_pct = row.get("BB%", None)

                # Look up internal player_id from fangraphs_id
                result = conn.execute(
                    "SELECT player_id FROM player_ids WHERE fangraphs_id = ?",
                    [int(fg_id)],
                ).fetchone()

                if result:
                    player_id = result[0]
                    k_minus_bb = None
                    if k_pct is not None and bb_pct is not None:
                        k_minus_bb = k_pct - bb_pct

                    conn.execute(
                        """
                        UPDATE player_pitching
                        SET fip = COALESCE(?, fip),
                            xfip = COALESCE(?, xfip),
                            siera = COALESCE(?, siera),
                            war = COALESCE(?, war),
                            stuff_plus = COALESCE(?, stuff_plus),
                            k_pct = COALESCE(?, k_pct),
                            bb_pct = COALESCE(?, bb_pct),
                            k_minus_bb_pct = COALESCE(?, k_minus_bb_pct)
                        WHERE player_id = ? AND season = ?
                        """,
                        [fip, xfip, siera, war, stuff_plus, k_pct, bb_pct, k_minus_bb,
                         player_id, year],
                    )
                    updated += 1
        except Exception as e:
            log.warning("fg_pitcher_stats_failed", year=year, error=str(e))
            continue

    log.info("fg_pitcher_stats_updated", rows=updated)
    return updated


def ingest_batter_stats(
    start_year: int | None = None, end_year: int | None = None
) -> int:
    """Update player_batting with FanGraphs advanced batter metrics."""
    from pybaseball import batting_stats

    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()
    updated = 0

    for year in tqdm(range(start, end + 1), desc="FG Batter Stats"):
        try:
            raw = batting_stats(year, qual=50)  # Minimum 50 PA
            if raw is None or raw.empty:
                continue

            for _, row in raw.iterrows():
                fg_id = row.get("IDfg", None)
                if fg_id is None:
                    continue

                woba = row.get("wOBA", None)
                wrc_plus = row.get("wRC+", None)
                war = row.get("WAR", None)
                babip = row.get("BABIP", None)

                result = conn.execute(
                    "SELECT player_id FROM player_ids WHERE fangraphs_id = ?",
                    [int(fg_id)],
                ).fetchone()

                if result:
                    player_id = result[0]
                    conn.execute(
                        """
                        UPDATE player_batting
                        SET woba = COALESCE(?, woba),
                            wrc_plus = COALESCE(?, wrc_plus),
                            war = COALESCE(?, war),
                            babip = COALESCE(?, babip)
                        WHERE player_id = ? AND season = ?
                        """,
                        [woba, wrc_plus, war, babip, player_id, year],
                    )
                    updated += 1
        except Exception as e:
            log.warning("fg_batter_stats_failed", year=year, error=str(e))
            continue

    log.info("fg_batter_stats_updated", rows=updated)
    return updated


def ingest_fangraphs(start_year: int | None = None, end_year: int | None = None) -> int:
    """Run all FanGraphs ingestion steps. Returns total rows updated."""
    total = 0
    total += ingest_team_batting_advanced(start_year, end_year)
    total += ingest_team_pitching_advanced(start_year, end_year)
    total += ingest_pitcher_stats(start_year, end_year)
    total += ingest_batter_stats(start_year, end_year)
    log.info("fangraphs_complete", total_updated=total)
    return total
