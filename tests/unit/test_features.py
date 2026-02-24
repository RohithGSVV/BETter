"""Unit tests for feature engineering modules."""

import pandas as pd
import numpy as np
import pytest

from better.features.team_rolling import _team_game_log, _rolling_features_for_team


@pytest.fixture
def sample_games() -> pd.DataFrame:
    """Minimal game data for 5 games of team NYY."""
    dates = pd.date_range("2024-04-01", periods=5, freq="D")
    return pd.DataFrame({
        "game_pk": range(2024040100, 2024040100 + 5),
        "game_date": dates,
        "season": 2024,
        "home_team": ["NYY", "BOS", "NYY", "BOS", "NYY"],
        "away_team": ["BOS", "NYY", "BOS", "NYY", "BOS"],
        "home_score": [5, 3, 7, 2, 4],
        "away_score": [3, 6, 1, 8, 4],
        "home_win": [True, False, True, False, False],
        "park_id": ["NYC", "BOS", "NYC", "BOS", "NYC"],
        "day_night": ["N", "N", "D", "N", "D"],
        "home_sp_id": [1, 2, 1, 2, 1],
        "away_sp_id": [2, 1, 2, 1, 2],
    })


class TestTeamGameLog:
    def test_correct_rows(self, sample_games):
        log = _team_game_log(sample_games, "NYY")
        # NYY is home in games 0,2,4 and away in 1,3 → 5 total
        assert len(log) == 5

    def test_runs_scored_home(self, sample_games):
        log = _team_game_log(sample_games, "NYY")
        log = log.sort_values("game_date").reset_index(drop=True)
        # Game 0: NYY at home, scored 5
        assert log.iloc[0]["runs_scored"] == 5
        assert log.iloc[0]["runs_allowed"] == 3

    def test_runs_scored_away(self, sample_games):
        log = _team_game_log(sample_games, "NYY")
        log = log.sort_values("game_date").reset_index(drop=True)
        # Game 1: NYY away at BOS, BOS scored 3, NYY scored 6
        assert log.iloc[1]["runs_scored"] == 6
        assert log.iloc[1]["runs_allowed"] == 3

    def test_win_column(self, sample_games):
        log = _team_game_log(sample_games, "NYY")
        log = log.sort_values("game_date").reset_index(drop=True)
        # Game 0: NYY home 5-3 → W, Game 1: NYY away 6-3 → W
        # Game 2: NYY home 7-1 → W, Game 3: NYY away 8-2 → W
        # Game 4: NYY home 4-4 → tie treated as loss (home_win=False)
        wins = log["win"].tolist()
        assert wins[0] == 1.0  # home win
        assert wins[1] == 1.0  # away win (BOS home 3, NYY away 6)
        assert wins[2] == 1.0  # home win
        assert wins[3] == 1.0  # away win (BOS home 2, NYY away 8)
        assert wins[4] == 0.0  # tie → home_win=False


class TestRollingFeatures:
    def test_no_leakage_first_game(self, sample_games):
        """First game of season should have 0 games_played."""
        log = _team_game_log(sample_games, "NYY")
        feat = _rolling_features_for_team(log, "NYY", elo_lookup=None)
        feat = feat.sort_values("as_of_date").reset_index(drop=True)
        assert feat.iloc[0]["games_played_season"] == 0

    def test_games_played_increments(self, sample_games):
        log = _team_game_log(sample_games, "NYY")
        feat = _rolling_features_for_team(log, "NYY", elo_lookup=None)
        feat = feat.sort_values("as_of_date").reset_index(drop=True)
        gp = feat["games_played_season"].tolist()
        assert gp == [0, 1, 2, 3, 4]

    def test_ewma_nan_for_first_game(self, sample_games):
        """EWMA should be NaN before any games have been played."""
        log = _team_game_log(sample_games, "NYY")
        feat = _rolling_features_for_team(log, "NYY", elo_lookup=None)
        feat = feat.sort_values("as_of_date").reset_index(drop=True)
        # First game: shift(1) gives NaN → EWMA of NaN is NaN
        assert pd.isna(feat.iloc[0]["ewma_win_rate_7"])

    def test_run_diff_accumulates(self, sample_games):
        log = _team_game_log(sample_games, "NYY")
        feat = _rolling_features_for_team(log, "NYY", elo_lookup=None)
        feat = feat.sort_values("as_of_date").reset_index(drop=True)
        # First game: no prior data → NaN from shift
        # Second game: only game 0's run_diff is available (5-3=+2)
        assert feat.iloc[1]["run_diff_7"] == pytest.approx(2.0)

    def test_bayesian_strength_without_elo(self, sample_games):
        """Without Elo lookup, bayesian_strength should be 0."""
        log = _team_game_log(sample_games, "NYY")
        feat = _rolling_features_for_team(log, "NYY", elo_lookup=None)
        assert (feat["bayesian_strength"] == 0.0).all()

    def test_bayesian_strength_with_elo(self, sample_games):
        log = _team_game_log(sample_games, "NYY")
        # Mock Elo lookup: NYY at 1600 on all dates
        dates = sorted(log["game_date"].unique())
        elo_lookup = {("NYY", d): 1600.0 for d in dates}
        feat = _rolling_features_for_team(log, "NYY", elo_lookup)
        expected = (1600 - 1500) / 400.0  # 0.25
        assert feat.iloc[0]["bayesian_strength"] == pytest.approx(expected)
