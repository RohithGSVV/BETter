"""FiveThirtyEight-style Elo rating system for MLB teams.

Processes games chronologically, recording pre-game Elo for each team,
then updating after the result.  Between seasons the ratings revert
toward the league-mean by a configurable fraction.

No-leakage guarantee: Elo for a team on date D reflects ONLY games
completed strictly before D.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from better.constants import (
    ELO_HOME_ADVANTAGE,
    ELO_K_FACTOR,
    ELO_MEAN,
    ELO_REVERSION_FACTOR,
    TEAM_ABBREVS,
)
from better.data.db import fetch_df
from better.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core Elo math
# ---------------------------------------------------------------------------

def elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for team A against team B (0-1)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(
    home_elo: float,
    away_elo: float,
    home_win: bool,
    k: float = ELO_K_FACTOR,
    home_advantage: float = ELO_HOME_ADVANTAGE,
) -> tuple[float, float]:
    """Compute new Elo ratings after a game.

    Returns ``(new_home_elo, new_away_elo)``.
    """
    # Adjust for home-field advantage when computing expectations
    expected_home = elo_expected(home_elo + home_advantage, away_elo)

    actual_home = 1.0 if home_win else 0.0
    delta = k * (actual_home - expected_home)

    return home_elo + delta, away_elo - delta


def revert_to_mean(
    rating: float,
    mean: float = ELO_MEAN,
    factor: float = ELO_REVERSION_FACTOR,
) -> float:
    """Regress a rating toward the mean between seasons."""
    return rating - factor * (rating - mean)


# ---------------------------------------------------------------------------
# Walk-forward Elo builder
# ---------------------------------------------------------------------------

def build_elo_ratings(
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Compute Elo ratings for every team on every game-date.

    Returns a DataFrame with columns ``(team, as_of_date, elo_rating)``.
    The ``elo_rating`` for a row is the team's rating **before** playing
    on ``as_of_date`` (i.e. it only incorporates prior results).
    """
    games = fetch_df(
        """
        SELECT game_pk, game_date, season, home_team, away_team, home_win
        FROM games
        WHERE season BETWEEN ? AND ?
          AND is_postseason = FALSE
        ORDER BY game_date, game_pk
        """,
        [start_year, end_year],
    )

    # Initialise all known teams at the mean
    ratings: dict[str, float] = {t: ELO_MEAN for t in TEAM_ABBREVS}

    rows: list[dict] = []
    current_season: int | None = None

    for tup in games.itertuples(index=False):
        season = tup.season
        game_date = tup.game_date
        home = tup.home_team
        away = tup.away_team
        home_win = bool(tup.home_win)

        # Season boundary → revert all ratings
        if season != current_season:
            if current_season is not None:
                for team in list(ratings):
                    ratings[team] = revert_to_mean(ratings[team])
            current_season = season

        # Ensure both teams have a rating (handles expansion / renamed teams)
        ratings.setdefault(home, ELO_MEAN)
        ratings.setdefault(away, ELO_MEAN)

        # Record **pre-game** Elo for both teams
        rows.append({"team": home, "as_of_date": game_date, "elo_rating": ratings[home]})
        rows.append({"team": away, "as_of_date": game_date, "elo_rating": ratings[away]})

        # Update after the game
        new_home, new_away = elo_update(
            ratings[home], ratings[away], home_win,
        )
        ratings[home] = new_home
        ratings[away] = new_away

    elo_df = pd.DataFrame(rows)

    # If a team plays twice on the same date (doubleheader) the *second*
    # row already reflects the first game's update → keep the **first**
    # pre-game value for features (safest no-leakage choice).
    elo_df = elo_df.drop_duplicates(subset=["team", "as_of_date"], keep="first")

    log.info(
        "elo_ratings_built",
        rows=len(elo_df),
        teams=elo_df["team"].nunique(),
        seasons=f"{start_year}-{end_year}",
    )
    return elo_df
