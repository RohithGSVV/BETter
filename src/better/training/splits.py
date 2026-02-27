"""Walk-forward cross-validation split generator.

Produces temporal train/test splits where training uses all data up to year Y
and testing uses year Y+1.  This strictly prevents lookahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from better.config import settings
from better.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardFold:
    """One fold of walk-forward cross-validation."""

    fold_id: int
    train_years: list[int]
    test_year: int
    train_idx: pd.Index
    test_idx: pd.Index


def generate_walk_forward_splits(
    df: pd.DataFrame,
    test_start_year: int | None = None,
    end_year: int | None = None,
) -> list[WalkForwardFold]:
    """Generate walk-forward CV folds from a time-sorted DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``season`` column (integer year).
    test_start_year : int
        First year used as a test set.  Defaults to
        ``settings.walk_forward_test_start`` (2015).
    end_year : int
        Last year to include as a test set.  Defaults to
        ``settings.train_end_year`` (2025).

    Returns
    -------
    list[WalkForwardFold]
    """
    start = test_start_year or settings.walk_forward_test_start
    end = end_year or settings.train_end_year

    folds: list[WalkForwardFold] = []
    available_years = sorted(df["season"].unique())

    for fold_id, test_year in enumerate(range(start, end + 1)):
        if test_year not in available_years:
            log.warning("skipping_fold_no_test_data", test_year=test_year)
            continue

        train_years = [y for y in available_years if y < test_year]
        if not train_years:
            log.warning("skipping_fold_no_train_data", test_year=test_year)
            continue

        train_mask = df["season"].isin(train_years)
        test_mask = df["season"] == test_year

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_years=train_years,
                test_year=test_year,
                train_idx=df.index[train_mask],
                test_idx=df.index[test_mask],
            )
        )

        log.info(
            "walk_forward_fold",
            fold_id=fold_id,
            train_years=f"{train_years[0]}-{train_years[-1]}",
            test_year=test_year,
            train_size=int(train_mask.sum()),
            test_size=int(test_mask.sum()),
        )

    return folds
