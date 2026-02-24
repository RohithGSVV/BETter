"""Team daily rolling features computed from the ``games`` table.

Populates the ``team_features_daily`` table.  All computations use
``shift(1)`` on chronologically-sorted game logs so that features for
a game on date D reflect only results **before** D (strict no-leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from better.config import settings
from better.constants import ELO_K_FACTOR, ELO_MEAN, TEAM_ABBREVS
from better.data.db import fetch_df, get_connection
from better.utils.logging import get_logger
from better.utils.stats import pythagorean_win_pct

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_games(start_year: int, end_year: int) -> pd.DataFrame:
    return fetch_df(
        """
        SELECT game_pk, game_date, season, home_team, away_team,
               home_score, away_score, home_win, park_id, day_night,
               home_sp_id, away_sp_id
        FROM games
        WHERE season BETWEEN ? AND ?
          AND is_postseason = FALSE
        ORDER BY game_date, game_pk
        """,
        [start_year, end_year],
    )


def _team_game_log(games: pd.DataFrame, team: str) -> pd.DataFrame:
    """Build one chronological row-per-game log for *team*."""
    home = games[games["home_team"] == team].copy()
    home["is_home"] = True
    home["runs_scored"] = home["home_score"].astype(float)
    home["runs_allowed"] = home["away_score"].astype(float)
    home["win"] = home["home_win"].astype(float)

    away = games[games["away_team"] == team].copy()
    away["is_home"] = False
    away["runs_scored"] = away["away_score"].astype(float)
    away["runs_allowed"] = away["home_score"].astype(float)
    away["win"] = (~away["home_win"]).astype(float)

    team_log = pd.concat([home, away], ignore_index=True)
    team_log = team_log.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    team_log["run_diff"] = team_log["runs_scored"] - team_log["runs_allowed"]
    return team_log


def _rolling_features_for_team(
    team_log: pd.DataFrame,
    team: str,
    elo_lookup: dict[tuple[str, object], float] | None,
) -> pd.DataFrame:
    """Vectorised rolling feature computation for a single team."""
    results: list[pd.DataFrame] = []

    for season in sorted(team_log["season"].unique()):
        sg = (
            team_log[team_log["season"] == season]
            .sort_values(["game_date", "game_pk"])
            .reset_index(drop=True)
            .copy()
        )
        n = len(sg)
        if n == 0:
            continue

        # ---- shifted series (exclude current game) ----
        win_s = sg["win"].shift(1)
        rs_s = sg["runs_scored"].shift(1)
        ra_s = sg["runs_allowed"].shift(1)
        rd_s = sg["run_diff"].shift(1)

        sg["team"] = team
        sg["as_of_date"] = sg["game_date"]

        # Season record (games BEFORE this one)
        sg["games_played_season"] = np.arange(n)  # 0 for first game
        sg["wins_season"] = win_s.fillna(0).cumsum().astype(int)
        sg["losses_season"] = sg["games_played_season"] - sg["wins_season"]

        # Run differentials (7 / 14 / 30 game windows)
        for w in settings.rolling_windows:
            sg[f"run_diff_{w}"] = rd_s.rolling(w, min_periods=1).sum()

        # EWMA win rates
        for w in settings.rolling_windows:
            sg[f"ewma_win_rate_{w}"] = win_s.ewm(span=w, adjust=False).mean()

        # Pythagorean win pct (30-game window, min 10 games)
        rs_30 = rs_s.rolling(30, min_periods=10).sum()
        ra_30 = ra_s.rolling(30, min_periods=10).sum()
        exp = settings.pythagorean_exponent
        sg["pythag_win_pct_30"] = np.where(
            rs_30.notna() & ra_30.notna() & ((rs_30 + ra_30) > 0),
            rs_30 ** exp / (rs_30 ** exp + ra_30 ** exp),
            np.nan,
        )

        # Elo-derived Bayesian strength
        if elo_lookup is not None:
            sg["bayesian_strength"] = sg.apply(
                lambda r: (elo_lookup.get((team, r["as_of_date"]), ELO_MEAN) - ELO_MEAN) / 400.0,
                axis=1,
            )
        else:
            sg["bayesian_strength"] = 0.0

        sg["bayesian_strength_var"] = ELO_K_FACTOR / (sg["games_played_season"] + ELO_K_FACTOR)

        # Bullpen IP proxy: count of team's games in last 3 calendar days * 3.0 IP
        sg["_date_dt"] = pd.to_datetime(sg["as_of_date"])
        bullpen_proxy = []
        date_list = sg["_date_dt"].tolist()
        for i in range(n):
            d = date_list[i]
            three_days_ago = d - pd.Timedelta(days=3)
            cnt = sum(1 for j in range(i) if date_list[j] > three_days_ago)
            bullpen_proxy.append(cnt * 3.0)
        sg["bullpen_ip_last_3d"] = bullpen_proxy
        sg.drop(columns=["_date_dt"], inplace=True)

        results.append(sg)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return combined


def _season_level_features(
    features: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Join season-level OBP, wRC+, FIP and bullpen FIP."""
    team_bat = fetch_df(
        "SELECT team, season, obp, wrc_plus FROM team_batting WHERE season BETWEEN ? AND ?",
        [start_year, end_year],
    )
    team_pit = fetch_df(
        "SELECT team, season, fip FROM team_pitching WHERE season BETWEEN ? AND ?",
        [start_year, end_year],
    )
    bullpen_fip = fetch_df(
        """
        SELECT team, season, AVG(fip) as bullpen_fip
        FROM player_pitching
        WHERE is_starter = FALSE AND fip IS NOT NULL
          AND season BETWEEN ? AND ?
        GROUP BY team, season
        """,
        [start_year, end_year],
    )

    # For early-season games (<30 GP) use prior season's stats
    features["_stat_season"] = features.apply(
        lambda r: r["season"] if r["games_played_season"] >= 30 else r["season"] - 1,
        axis=1,
    )

    features = features.merge(
        team_bat.rename(columns={
            "season": "_stat_season",
            "obp": "obp_rolling_30",
            "wrc_plus": "wrc_plus_rolling_30",
        }),
        on=["team", "_stat_season"],
        how="left",
    )
    features = features.merge(
        team_pit.rename(columns={
            "season": "_stat_season",
            "fip": "team_fip_rolling_30",
        }),
        on=["team", "_stat_season"],
        how="left",
    )
    features = features.merge(
        bullpen_fip.rename(columns={
            "season": "_stat_season",
            "bullpen_fip": "bullpen_fip_composite",
        }),
        on=["team", "_stat_season"],
        how="left",
    )

    features.drop(columns=["_stat_season"], inplace=True)
    return features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_TEAM_FEAT_COLS = [
    "team", "as_of_date",
    "bayesian_strength", "bayesian_strength_var",
    "pythag_win_pct_30",
    "run_diff_30", "run_diff_14", "run_diff_7",
    "ewma_win_rate_7", "ewma_win_rate_14", "ewma_win_rate_30",
    "obp_rolling_30", "wrc_plus_rolling_30", "team_fip_rolling_30",
    "bullpen_fip_composite", "bullpen_ip_last_3d",
    "games_played_season", "wins_season", "losses_season",
]


def build_team_features(
    start_year: int,
    end_year: int,
    elo_df: pd.DataFrame | None = None,
) -> int:
    """Compute and insert ``team_features_daily`` rows.

    Parameters
    ----------
    elo_df : DataFrame with ``(team, as_of_date, elo_rating)``
        If provided, used to derive ``bayesian_strength``.

    Returns
    -------
    int  – number of rows inserted.
    """
    conn = get_connection()
    games = _load_games(start_year, end_year)

    # Build Elo lookup dict for O(1) access
    elo_lookup: dict[tuple[str, object], float] | None = None
    if elo_df is not None and not elo_df.empty:
        elo_lookup = {
            (row.team, row.as_of_date): row.elo_rating
            for row in elo_df.itertuples(index=False)
        }

    # Determine which teams actually appear in the data
    teams = sorted(set(games["home_team"].unique()) | set(games["away_team"].unique()))

    all_features: list[pd.DataFrame] = []
    for team in tqdm(teams, desc="Team features"):
        team_log = _team_game_log(games, team)
        if team_log.empty:
            continue
        feat = _rolling_features_for_team(team_log, team, elo_lookup)
        if not feat.empty:
            all_features.append(feat)

    if not all_features:
        log.warning("no_team_features_generated")
        return 0

    features = pd.concat(all_features, ignore_index=True)

    # Add season-level stats
    features = _season_level_features(features, start_year, end_year)

    # Select exactly the columns the schema expects
    for col in _TEAM_FEAT_COLS:
        if col not in features.columns:
            features[col] = None
    features = features[_TEAM_FEAT_COLS].copy()

    # Ensure correct types for DuckDB
    features["as_of_date"] = pd.to_datetime(features["as_of_date"])
    features["games_played_season"] = features["games_played_season"].astype("Int64")
    features["wins_season"] = features["wins_season"].astype("Int64")
    features["losses_season"] = features["losses_season"].astype("Int64")

    # Deduplicate (team, as_of_date) keeping first occurrence
    features = features.drop_duplicates(subset=["team", "as_of_date"], keep="first")

    # Write to DuckDB
    conn.execute(
        "DELETE FROM team_features_daily WHERE as_of_date BETWEEN ? AND ?",
        [f"{start_year}-01-01", f"{end_year}-12-31"],
    )
    cols = ", ".join(_TEAM_FEAT_COLS)
    conn.execute(f"INSERT INTO team_features_daily ({cols}) SELECT {cols} FROM features")

    log.info("team_features_built", rows=len(features), teams=len(teams))
    return len(features)
