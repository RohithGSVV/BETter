"""Game-level matchup feature builder.

Joins ``team_features_daily``, ``pitcher_features``, Elo ratings, and
park factors into a single training DataFrame where each row is one game.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from better.constants import ELO_HOME_ADVANTAGE, PARK_FACTORS
from better.data.db import fetch_df
from better.features.elo import elo_expected
from better.utils.logging import get_logger

log = get_logger(__name__)


def build_matchup_features(
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Build the complete game-level training DataFrame via DuckDB SQL.

    Each row corresponds to one regular-season game with:
      - Home/away team rolling features
      - Home/away starting pitcher features
      - Derived matchup differentials
      - Park + context features
      - Target: ``home_win``

    Returns
    -------
    pd.DataFrame  – ready for model training.
    """
    df = fetch_df(
        """
        SELECT
            g.game_pk,
            g.game_date,
            g.season,
            g.home_team,
            g.away_team,
            g.home_win,
            g.home_score,
            g.away_score,
            g.day_night,

            -- Home team features
            htf.bayesian_strength     AS home_bayesian_strength,
            htf.bayesian_strength_var AS home_bayesian_strength_var,
            htf.pythag_win_pct_30     AS home_pythag_win_pct_30,
            htf.run_diff_30           AS home_run_diff_30,
            htf.run_diff_14           AS home_run_diff_14,
            htf.run_diff_7            AS home_run_diff_7,
            htf.ewma_win_rate_7       AS home_ewma_win_rate_7,
            htf.ewma_win_rate_14      AS home_ewma_win_rate_14,
            htf.ewma_win_rate_30      AS home_ewma_win_rate_30,
            htf.obp_rolling_30        AS home_obp_rolling_30,
            htf.wrc_plus_rolling_30   AS home_wrc_plus_rolling_30,
            htf.team_fip_rolling_30   AS home_team_fip_rolling_30,
            htf.bullpen_fip_composite AS home_bullpen_fip,
            htf.bullpen_ip_last_3d    AS home_bullpen_ip_last_3d,
            htf.games_played_season   AS home_games_played,
            htf.wins_season           AS home_wins,
            htf.losses_season         AS home_losses,

            -- Away team features
            atf.bayesian_strength     AS away_bayesian_strength,
            atf.bayesian_strength_var AS away_bayesian_strength_var,
            atf.pythag_win_pct_30     AS away_pythag_win_pct_30,
            atf.run_diff_30           AS away_run_diff_30,
            atf.run_diff_14           AS away_run_diff_14,
            atf.run_diff_7            AS away_run_diff_7,
            atf.ewma_win_rate_7       AS away_ewma_win_rate_7,
            atf.ewma_win_rate_14      AS away_ewma_win_rate_14,
            atf.ewma_win_rate_30      AS away_ewma_win_rate_30,
            atf.obp_rolling_30        AS away_obp_rolling_30,
            atf.wrc_plus_rolling_30   AS away_wrc_plus_rolling_30,
            atf.team_fip_rolling_30   AS away_team_fip_rolling_30,
            atf.bullpen_fip_composite AS away_bullpen_fip,
            atf.bullpen_ip_last_3d    AS away_bullpen_ip_last_3d,
            atf.games_played_season   AS away_games_played,
            atf.wins_season           AS away_wins,
            atf.losses_season         AS away_losses,

            -- Home starting pitcher
            hpf.fip       AS home_sp_fip,
            hpf.xfip      AS home_sp_xfip,
            hpf.siera     AS home_sp_siera,
            hpf.stuff_plus AS home_sp_stuff_plus,
            hpf.k_pct     AS home_sp_k_pct,
            hpf.bb_pct    AS home_sp_bb_pct,
            hpf.k_minus_bb_pct AS home_sp_k_minus_bb_pct,
            hpf.days_rest  AS home_sp_days_rest,
            hpf.throws     AS home_sp_throws,

            -- Away starting pitcher
            apf.fip       AS away_sp_fip,
            apf.xfip      AS away_sp_xfip,
            apf.siera     AS away_sp_siera,
            apf.stuff_plus AS away_sp_stuff_plus,
            apf.k_pct     AS away_sp_k_pct,
            apf.bb_pct    AS away_sp_bb_pct,
            apf.k_minus_bb_pct AS away_sp_k_minus_bb_pct,
            apf.days_rest  AS away_sp_days_rest,
            apf.throws     AS away_sp_throws

        FROM games g
        LEFT JOIN team_features_daily htf
            ON g.home_team = htf.team AND g.game_date = htf.as_of_date
        LEFT JOIN team_features_daily atf
            ON g.away_team = atf.team AND g.game_date = atf.as_of_date
        LEFT JOIN pitcher_features hpf
            ON g.home_sp_id = hpf.pitcher_id AND g.game_date = hpf.as_of_date
        LEFT JOIN pitcher_features apf
            ON g.away_sp_id = apf.pitcher_id AND g.game_date = apf.as_of_date
        WHERE g.season BETWEEN ? AND ?
          AND g.is_postseason = FALSE
        ORDER BY g.game_date, g.game_pk
        """,
        [start_year, end_year],
    )

    if df.empty:
        log.warning("matchup_features_empty")
        return df

    # --- Derived matchup features ---

    # Elo diff & expected win prob (use bayesian_strength × 400 + 1500 as proxy)
    df["elo_diff"] = (
        df["home_bayesian_strength"].fillna(0) - df["away_bayesian_strength"].fillna(0)
    ) * 400.0

    df["elo_home_win_prob"] = df["elo_diff"].apply(
        lambda d: elo_expected(1500 + d / 2 + ELO_HOME_ADVANTAGE, 1500 - d / 2)
    )

    # Strength differential
    df["strength_diff"] = (
        df["home_bayesian_strength"].fillna(0) - df["away_bayesian_strength"].fillna(0)
    )

    # Starting pitcher FIP differential (positive = home SP advantage)
    df["sp_fip_diff"] = df["away_sp_fip"].fillna(4.5) - df["home_sp_fip"].fillna(4.5)

    # Run diff momentum (7-day)
    df["run_diff_momentum"] = (
        df["home_run_diff_7"].fillna(0) - df["away_run_diff_7"].fillna(0)
    )

    # Park factors
    df["park_runs_factor"] = df["home_team"].map(
        lambda t: PARK_FACTORS.get(t, {}).get("runs", 1.0)
    )
    df["park_hr_factor"] = df["home_team"].map(
        lambda t: PARK_FACTORS.get(t, {}).get("hr", 1.0)
    )

    # Context
    df["is_day_game"] = (df["day_night"] == "D").astype(int)

    # Clean up: drop raw columns not needed for modelling
    df.drop(columns=["day_night", "home_score", "away_score"], inplace=True, errors="ignore")

    log.info(
        "matchup_features_built",
        rows=len(df),
        columns=len(df.columns),
        home_win_rate=round(df["home_win"].mean(), 4),
    )
    return df
