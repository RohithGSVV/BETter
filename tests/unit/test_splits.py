"""Tests for walk-forward cross-validation split generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from better.training.splits import generate_walk_forward_splits


@pytest.fixture
def multi_season_df() -> pd.DataFrame:
    """DataFrame spanning 2010–2020 with 100 rows per season."""
    rng = np.random.default_rng(42)
    rows = []
    for year in range(2010, 2021):
        for i in range(100):
            rows.append({"season": year, "home_win": rng.integers(0, 2)})
    return pd.DataFrame(rows)


class TestWalkForwardSplits:
    def test_fold_count(self, multi_season_df):
        folds = generate_walk_forward_splits(
            multi_season_df, test_start_year=2015, end_year=2020
        )
        assert len(folds) == 6  # 2015, 2016, 2017, 2018, 2019, 2020

    def test_no_overlap(self, multi_season_df):
        folds = generate_walk_forward_splits(
            multi_season_df, test_start_year=2015, end_year=2020
        )
        for fold in folds:
            overlap = set(fold.train_idx) & set(fold.test_idx)
            assert len(overlap) == 0, f"Fold {fold.fold_id} has overlapping indices"

    def test_temporal_ordering(self, multi_season_df):
        folds = generate_walk_forward_splits(
            multi_season_df, test_start_year=2015, end_year=2020
        )
        for fold in folds:
            for ty in fold.train_years:
                assert ty < fold.test_year

    def test_expanding_window(self, multi_season_df):
        folds = generate_walk_forward_splits(
            multi_season_df, test_start_year=2015, end_year=2020
        )
        for i in range(1, len(folds)):
            assert len(folds[i].train_idx) > len(folds[i - 1].train_idx)

    def test_test_set_is_single_year(self, multi_season_df):
        folds = generate_walk_forward_splits(
            multi_season_df, test_start_year=2015, end_year=2020
        )
        for fold in folds:
            test_seasons = multi_season_df.loc[fold.test_idx, "season"].unique()
            assert len(test_seasons) == 1
            assert test_seasons[0] == fold.test_year

    def test_custom_start_year(self, multi_season_df):
        folds = generate_walk_forward_splits(
            multi_season_df, test_start_year=2018, end_year=2020
        )
        assert len(folds) == 3
        assert folds[0].test_year == 2018
