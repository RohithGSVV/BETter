"""Lahman Database ingestion.

Downloads and loads season-level team and player statistics.
Source: https://github.com/chadwickbureau/baseballdatabank
"""

from __future__ import annotations

import io

import httpx
import pandas as pd
from tqdm import tqdm

from better.config import settings
from better.constants import RETROSHEET_TEAM_MAP
from better.data.db import get_connection
from better.utils.logging import get_logger

log = get_logger(__name__)

# Lahman data is hosted on the Chadwick Bureau GitHub
LAHMAN_BASE_URL = (
    "https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/"
)

LAHMAN_FILES = {
    "teams": "Teams.csv",
    "batting": "Batting.csv",
    "pitching": "Pitching.csv",
    "people": "People.csv",
}


def _download_csv(filename: str) -> pd.DataFrame:
    """Download a CSV file from the Lahman GitHub repository."""
    url = LAHMAN_BASE_URL + filename
    log.info("downloading_lahman", file=filename, url=url)
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def _normalize_team(team_id: str) -> str:
    """Normalize Lahman team IDs to standard abbreviations."""
    return RETROSHEET_TEAM_MAP.get(team_id, team_id)


def ingest_teams(start_year: int | None = None, end_year: int | None = None) -> int:
    """Load Lahman Teams table into team_batting and team_pitching."""
    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()

    raw = _download_csv(LAHMAN_FILES["teams"])
    raw = raw[(raw["yearID"] >= start) & (raw["yearID"] <= end)]

    # Team batting
    batting = pd.DataFrame()
    batting["team"] = raw["teamIDBR"].map(_normalize_team)
    batting["season"] = raw["yearID"]
    batting["games"] = raw["G"]
    batting["plate_appearances"] = raw.get("PA", raw["AB"] + raw["BB"] + raw.get("HBP", 0))
    batting["at_bats"] = raw["AB"]
    batting["hits"] = raw["H"]
    batting["doubles"] = raw["2B"]
    batting["triples"] = raw["3B"]
    batting["home_runs"] = raw["HR"]
    batting["runs"] = raw["R"]
    batting["rbi"] = raw.get("RBI", 0)
    batting["walks"] = raw["BB"]
    batting["strikeouts"] = raw["SO"]
    batting["stolen_bases"] = raw.get("SB", 0)

    # Compute rate stats
    batting["batting_avg"] = (batting["hits"] / batting["at_bats"]).round(3)
    batting["obp"] = (
        (batting["hits"] + batting["walks"])
        / batting["plate_appearances"]
    ).round(3)
    batting["slg"] = (
        (batting["hits"] - batting["doubles"] - batting["triples"] - batting["home_runs"]
         + 2 * batting["doubles"] + 3 * batting["triples"] + 4 * batting["home_runs"])
        / batting["at_bats"]
    ).round(3)
    batting["ops"] = (batting["obp"] + batting["slg"]).round(3)
    batting["iso"] = (batting["slg"] - batting["batting_avg"]).round(3)
    batting["babip"] = None  # Requires more data to compute accurately
    batting["woba"] = None  # Computed from FanGraphs data
    batting["wrc_plus"] = None  # Computed from FanGraphs data

    conn.execute("DELETE FROM team_batting WHERE season BETWEEN ? AND ?", [start, end])
    conn.execute("INSERT INTO team_batting SELECT * FROM batting")

    # Team pitching
    pitching = pd.DataFrame()
    pitching["team"] = raw["teamIDBR"].map(_normalize_team)
    pitching["season"] = raw["yearID"]
    pitching["games"] = raw["G"]
    pitching["innings_pitched"] = raw.get("IPouts", raw["G"] * 27) / 3  # IPouts to IP
    pitching["era"] = raw.get("ERA", None)
    pitching["fip"] = None  # Computed from FanGraphs
    pitching["xfip"] = None
    pitching["siera"] = None
    pitching["whip"] = None
    pitching["k_per_9"] = (raw["SOA"] / pitching["innings_pitched"] * 9).round(2) if "SOA" in raw.columns else None
    pitching["bb_per_9"] = (raw["BBA"] / pitching["innings_pitched"] * 9).round(2) if "BBA" in raw.columns else None
    pitching["hr_per_9"] = (raw["HRA"] / pitching["innings_pitched"] * 9).round(2) if "HRA" in raw.columns else None
    pitching["k_pct"] = None
    pitching["bb_pct"] = None
    pitching["k_minus_bb_pct"] = None
    pitching["avg_against"] = None
    pitching["runs_allowed"] = raw["RA"]
    pitching["earned_runs"] = raw["ER"]

    conn.execute("DELETE FROM team_pitching WHERE season BETWEEN ? AND ?", [start, end])
    conn.execute("INSERT INTO team_pitching SELECT * FROM pitching")

    total = len(batting) + len(pitching)
    log.info("lahman_teams_loaded", batting_rows=len(batting), pitching_rows=len(pitching))
    return total


def ingest_player_batting(start_year: int | None = None, end_year: int | None = None) -> int:
    """Load Lahman individual batting stats."""
    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()

    raw = _download_csv(LAHMAN_FILES["batting"])
    raw = raw[(raw["yearID"] >= start) & (raw["yearID"] <= end)]

    # Load People for handedness
    people = _download_csv(LAHMAN_FILES["people"])
    people_map = dict(zip(people["playerID"], people["bats"]))

    df = pd.DataFrame()
    df["player_id"] = raw["playerID"].apply(hash).astype(int) % (2**31)
    df["season"] = raw["yearID"]
    df["team"] = raw["teamID"].map(_normalize_team)
    df["games"] = raw["G"]
    df["plate_appearances"] = raw["AB"] + raw["BB"] + raw.get("HBP", 0) + raw.get("SF", 0)
    df["at_bats"] = raw["AB"]
    df["hits"] = raw["H"]
    df["doubles"] = raw["2B"]
    df["triples"] = raw["3B"]
    df["home_runs"] = raw["HR"]
    df["runs"] = raw["R"]
    df["rbi"] = raw["RBI"]
    df["walks"] = raw["BB"]
    df["strikeouts"] = raw["SO"]

    df["batting_avg"] = (df["hits"] / df["at_bats"].replace(0, 1)).round(3)
    df["obp"] = (
        (df["hits"] + df["walks"]) / df["plate_appearances"].replace(0, 1)
    ).round(3)
    df["slg"] = (
        (df["hits"] - df["doubles"] - df["triples"] - df["home_runs"]
         + 2 * df["doubles"] + 3 * df["triples"] + 4 * df["home_runs"])
        / df["at_bats"].replace(0, 1)
    ).round(3)
    df["ops"] = (df["obp"] + df["slg"]).round(3)
    df["woba"] = None
    df["wrc_plus"] = None
    df["iso"] = (df["slg"] - df["batting_avg"]).round(3)
    df["babip"] = None
    df["war"] = None
    df["bats"] = raw["playerID"].map(people_map)

    conn.execute("DELETE FROM player_batting WHERE season BETWEEN ? AND ?", [start, end])
    conn.execute("INSERT INTO player_batting SELECT * FROM df")

    log.info("lahman_batting_loaded", rows=len(df))
    return len(df)


def ingest_player_pitching(start_year: int | None = None, end_year: int | None = None) -> int:
    """Load Lahman individual pitching stats."""
    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    conn = get_connection()

    raw = _download_csv(LAHMAN_FILES["pitching"])
    raw = raw[(raw["yearID"] >= start) & (raw["yearID"] <= end)]

    people = _download_csv(LAHMAN_FILES["people"])
    people_throws = dict(zip(people["playerID"], people["throws"]))

    df = pd.DataFrame()
    df["player_id"] = raw["playerID"].apply(hash).astype(int) % (2**31)
    df["season"] = raw["yearID"]
    df["team"] = raw["teamID"].map(_normalize_team)
    df["games"] = raw["G"]
    df["games_started"] = raw["GS"]
    df["innings_pitched"] = raw["IPouts"] / 3
    df["wins"] = raw["W"]
    df["losses"] = raw["L"]
    df["era"] = raw["ERA"]
    df["fip"] = None
    df["xfip"] = None
    df["siera"] = None
    df["whip"] = ((raw["H"] + raw["BB"]) / df["innings_pitched"].replace(0, 1)).round(3)
    df["k_per_9"] = (raw["SO"] / df["innings_pitched"].replace(0, 1) * 9).round(2)
    df["bb_per_9"] = (raw["BB"] / df["innings_pitched"].replace(0, 1) * 9).round(2)
    df["k_pct"] = None
    df["bb_pct"] = None
    df["k_minus_bb_pct"] = None
    df["war"] = None
    df["stuff_plus"] = None
    df["throws"] = raw["playerID"].map(people_throws)
    df["is_starter"] = raw["GS"] > raw["G"] * 0.5

    conn.execute("DELETE FROM player_pitching WHERE season BETWEEN ? AND ?", [start, end])
    conn.execute("INSERT INTO player_pitching SELECT * FROM df")

    log.info("lahman_pitching_loaded", rows=len(df))
    return len(df)


def ingest_lahman(start_year: int | None = None, end_year: int | None = None) -> int:
    """Run all Lahman ingestion steps. Returns total rows loaded."""
    total = 0
    total += ingest_teams(start_year, end_year)
    total += ingest_player_batting(start_year, end_year)
    total += ingest_player_pitching(start_year, end_year)
    log.info("lahman_complete", total_rows=total)
    return total
