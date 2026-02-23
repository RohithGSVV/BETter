"""Player ID crosswalk from the Chadwick Bureau register.

Maps between MLBAM, FanGraphs, Baseball Reference, and Retrosheet player IDs.
Source: https://github.com/chadwickbureau/register
"""

from __future__ import annotations

import io

import httpx
import pandas as pd

from better.data.db import get_connection
from better.utils.logging import get_logger

log = get_logger(__name__)

CHADWICK_URL = (
    "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people.csv"
)


def download_chadwick_register() -> pd.DataFrame:
    """Download the Chadwick Bureau player register."""
    log.info("downloading_chadwick_register")
    response = httpx.get(CHADWICK_URL, follow_redirects=True, timeout=120)
    response.raise_for_status()

    raw = pd.read_csv(io.StringIO(response.text), low_memory=False)
    log.info("chadwick_downloaded", rows=len(raw))
    return raw


def transform_register(raw: pd.DataFrame) -> pd.DataFrame:
    """Transform Chadwick register into our player_ids schema."""
    df = pd.DataFrame()

    # Use key_mlbam as primary ID where available
    df["mlb_id"] = pd.to_numeric(raw.get("key_mlbam", None), errors="coerce").astype("Int64")
    df["retrosheet_id"] = raw.get("key_retro", "")
    df["fangraphs_id"] = pd.to_numeric(
        raw.get("key_fangraphs", None), errors="coerce"
    ).astype("Int64")
    df["bbref_id"] = raw.get("key_bbref", "")
    df["name_first"] = raw.get("name_first", "")
    df["name_last"] = raw.get("name_last", "")
    df["birth_date"] = pd.to_datetime(
        raw[["birth_year", "birth_month", "birth_day"]].rename(
            columns={"birth_year": "year", "birth_month": "month", "birth_day": "day"}
        ),
        errors="coerce",
    )
    df["throws"] = raw.get("throws", None)
    df["bats"] = raw.get("bats", None)

    # Generate player_id as row index for players with at least an MLB ID
    df = df[df["mlb_id"].notna()].copy()
    df["player_id"] = df["mlb_id"]

    return df


def ingest_player_ids() -> int:
    """Download and load Chadwick register into DuckDB.

    Returns the number of players loaded.
    """
    conn = get_connection()
    raw = download_chadwick_register()
    df = transform_register(raw)

    conn.execute("DELETE FROM player_ids")
    conn.execute("INSERT INTO player_ids SELECT * FROM df")

    log.info("player_ids_loaded", players=len(df))
    return len(df)


def lookup_mlb_id(fangraphs_id: int) -> int | None:
    """Look up MLBAM ID from a FanGraphs ID."""
    conn = get_connection()
    result = conn.execute(
        "SELECT mlb_id FROM player_ids WHERE fangraphs_id = ?", [fangraphs_id]
    ).fetchone()
    return result[0] if result else None


def lookup_fangraphs_id(mlb_id: int) -> int | None:
    """Look up FanGraphs ID from an MLBAM ID."""
    conn = get_connection()
    result = conn.execute(
        "SELECT fangraphs_id FROM player_ids WHERE mlb_id = ?", [mlb_id]
    ).fetchone()
    return result[0] if result else None
