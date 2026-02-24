"""Feature engineering pipeline orchestrator.

Sequences all feature-building steps in dependency order:
  1. Elo ratings  (no dependencies)
  2. Team rolling features  (uses Elo)
  3. Pitcher features  (uses games.home_sp_id)
"""

from __future__ import annotations

import pandas as pd

from better.config import settings
from better.features.elo import build_elo_ratings
from better.features.matchup import build_matchup_features
from better.features.pitcher import build_pitcher_features
from better.features.team_rolling import build_team_features
from better.utils.logging import get_logger

log = get_logger(__name__)


def run_feature_pipeline(
    start_year: int | None = None,
    end_year: int | None = None,
    skip_pitcher: bool = False,
) -> dict:
    """Execute the full feature engineering pipeline.

    Parameters
    ----------
    start_year, end_year : int, optional
        Season range (defaults to ``settings.train_start_year`` / ``end``).
    skip_pitcher : bool
        If ``True``, skip pitcher features (useful when SP IDs are
        not yet populated).

    Returns
    -------
    dict  – row counts for each pipeline step.
    """
    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year

    log.info("feature_pipeline_starting", start_year=start, end_year=end)
    results: dict[str, int] = {}

    # Step 1 — Elo ratings
    log.info("step_1_elo_ratings")
    elo_df = build_elo_ratings(start, end)
    results["elo_rows"] = len(elo_df)

    # Step 2 — Team rolling features (uses Elo)
    log.info("step_2_team_features")
    team_rows = build_team_features(start, end, elo_df=elo_df)
    results["team_feature_rows"] = team_rows

    # Step 3 — Pitcher features
    if not skip_pitcher:
        log.info("step_3_pitcher_features")
        pitcher_rows = build_pitcher_features(start, end)
        results["pitcher_feature_rows"] = pitcher_rows
    else:
        log.info("skipping_pitcher_features")
        results["pitcher_feature_rows"] = 0

    log.info("feature_pipeline_complete", **results)
    return results


def build_training_set(
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Build the complete game-level training DataFrame.

    Assumes ``team_features_daily`` and ``pitcher_features`` are already
    populated (run ``run_feature_pipeline`` first).
    """
    start = start_year or settings.train_start_year
    end = end_year or settings.train_end_year
    return build_matchup_features(start, end)
