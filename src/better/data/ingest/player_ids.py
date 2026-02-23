"""Player ID crosswalk from the Chadwick Bureau register.

Maps between MLBAM, FanGraphs, Baseball Reference, and Retrosheet player IDs.
Uses pybaseball's built-in chadwick_register() which fetches the zip archive
from https://github.com/chadwickbureau/register/archive/refs/heads/master.zip
"""

from __future__ import annotations

import pandas as pd

from better.data.db import get_connection
from better.utils.logging import get_logger

log = get_logger(__name__)


def download_chadwick_register() -> pd.DataFrame:
    """Download the Chadwick Bureau player register via pybaseball."""
    from pybaseball.playerid_lookup import chadwick_register

    log.info("downloading_chadwick_register")
    raw = chadwick_register()
    log.info("chadwick_downloaded", rows=len(raw))
    return raw


def transform_register(raw: pd.DataFrame) -> pd.DataFrame:
    """Transform Chadwick register into our player_ids schema.

    pybaseball's chadwick_register() returns columns:
      name_last, name_first, key_mlbam, key_retro, key_bbref,
      key_fangraphs, mlb_played_first, mlb_played_last
    """
    df = pd.DataFrame()

    df["mlb_id"] = pd.to_numeric(raw.get("key_mlbam", None), errors="coerce").astype("Int64")
    df["retrosheet_id"] = raw.get("key_retro", pd.Series([""] * len(raw))).fillna("").astype(str)
    df["fangraphs_id"] = pd.to_numeric(
        raw.get("key_fangraphs", None), errors="coerce"
    ).astype("Int64")
    df["bbref_id"] = raw.get("key_bbref", pd.Series([""] * len(raw))).fillna("").astype(str)
    df["name_first"] = raw.get("name_first", pd.Series([""] * len(raw))).fillna("").astype(str)
    df["name_last"] = raw.get("name_last", pd.Series([""] * len(raw))).fillna("").astype(str)

    # pybaseball's register does not include birth date, throws, or bats
    df["birth_date"] = None
    df["throws"] = None
    df["bats"] = None

    # Keep only players with a valid MLB ID (key_mlbam != -1 and not null)
    df = df[(df["mlb_id"].notna()) & (df["mlb_id"] > 0)].copy()
    df["player_id"] = df["mlb_id"]

    # Column order must match CREATE TABLE statement
    return df[
        ["player_id", "mlb_id", "retrosheet_id", "fangraphs_id",
         "bbref_id", "name_first", "name_last", "birth_date", "throws", "bats"]
    ]


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
