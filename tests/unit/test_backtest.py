"""Tests for the Backtester."""

from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pytest

from better.betting.backtest import Backtester, BacktestResult


@pytest.fixture
def synthetic_oof_csv(tmp_path: Path) -> Path:
    """Create a synthetic OOF details CSV for testing."""
    np.random.seed(42)
    n_games = 500

    # Model is slightly better than Elo (synthetic edge)
    elo_probs = np.random.uniform(0.40, 0.65, n_games)
    model_probs = elo_probs + np.random.normal(0.02, 0.03, n_games)
    model_probs = np.clip(model_probs, 0.01, 0.99)

    # Home team wins based on "true" probability (between model and elo)
    true_probs = (model_probs + elo_probs) / 2
    outcomes = np.random.binomial(1, true_probs)

    dates = pd.date_range("2020-04-01", periods=n_games, freq="D")

    df = pd.DataFrame({
        "game_pk": range(1000, 1000 + n_games),
        "game_date": dates,
        "season": [d.year for d in dates],
        "home_team": np.random.choice(["NYY", "BOS", "LAD", "HOU", "ATL"], n_games),
        "away_team": np.random.choice(["NYM", "TBR", "SFG", "SEA", "PHI"], n_games),
        "home_win": outcomes,
        "elo_home_win_prob": elo_probs,
        "meta_learner_prob": model_probs,
        "gbm_ensemble_prob": model_probs + np.random.normal(0, 0.01, n_games),
        "bayesian_kalman_prob": elo_probs + np.random.normal(0.01, 0.02, n_games),
        "monte_carlo_prob": elo_probs + np.random.normal(0, 0.02, n_games),
    })

    path = tmp_path / "oof_details.csv"
    df.to_csv(path, index=False)
    return path


class TestBacktester:
    """Tests for historical backtesting."""

    def test_backtest_runs(self, synthetic_oof_csv: Path):
        """Backtest completes without errors on synthetic data."""
        bt = Backtester(
            initial_bankroll=1000.0,
            min_edge=0.03,
            kelly_fraction=0.25,
            model="meta_learner",
        )
        result = bt.run(oof_path=synthetic_oof_csv)

        assert isinstance(result, BacktestResult)
        assert result.stats.total_bets > 0
        assert result.stats.initial_bankroll == 1000.0

    def test_higher_edge_threshold_fewer_bets(self, synthetic_oof_csv: Path):
        """Higher edge threshold should result in fewer bets."""
        low = Backtester(min_edge=0.01, model="meta_learner").run(synthetic_oof_csv)
        high = Backtester(min_edge=0.10, model="meta_learner").run(synthetic_oof_csv)

        assert low.stats.total_bets >= high.stats.total_bets

    def test_bankroll_curve_tracked(self, synthetic_oof_csv: Path):
        """Bankroll curve has entries for each bet placed."""
        result = Backtester(min_edge=0.03, model="meta_learner").run(synthetic_oof_csv)

        if result.stats.total_bets > 0:
            assert not result.bankroll_curve.empty
            assert "bankroll" in result.bankroll_curve.columns

    def test_by_year_breakdown(self, synthetic_oof_csv: Path):
        """By-year breakdown is produced."""
        result = Backtester(min_edge=0.03, model="meta_learner").run(synthetic_oof_csv)

        if result.stats.total_bets > 0:
            assert not result.by_year.empty
            assert "year" in result.by_year.columns

    def test_sweep_edge_thresholds(self, synthetic_oof_csv: Path):
        """Edge threshold sweep returns comparison table."""
        bt = Backtester(model="meta_learner")
        sweep = bt.sweep_edge_thresholds(
            thresholds=[0.01, 0.05, 0.10],
            oof_path=synthetic_oof_csv,
        )

        assert len(sweep) == 3
        assert "edge_threshold" in sweep.columns
        assert "total_bets" in sweep.columns
        assert "roi_pct" in sweep.columns
        # Lower threshold should have more bets
        assert sweep.iloc[0]["total_bets"] >= sweep.iloc[-1]["total_bets"]

    def test_missing_model_column_raises(self, synthetic_oof_csv: Path):
        """Using a model that doesn't exist in the CSV raises an error."""
        bt = Backtester(model="nonexistent_model")
        with pytest.raises(ValueError, match="not found"):
            bt.run(synthetic_oof_csv)

    def test_missing_oof_file_raises(self, tmp_path: Path):
        """Missing OOF file raises FileNotFoundError."""
        bt = Backtester(model="meta_learner")
        with pytest.raises(FileNotFoundError):
            bt.run(tmp_path / "does_not_exist.csv")

    def test_print_summary(self, synthetic_oof_csv: Path, capsys):
        """print_summary runs without errors."""
        result = Backtester(min_edge=0.03, model="meta_learner").run(synthetic_oof_csv)
        result.print_summary()
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out
