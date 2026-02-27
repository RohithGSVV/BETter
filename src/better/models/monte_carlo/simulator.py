"""Monte Carlo game simulator using Poisson / negative-binomial run models.

Simulates N complete games by sampling runs for each team from a scoring
distribution parameterized by team offensive/pitching strength and park
factors.  ``P(home_win) = wins / N``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from better.config import settings
from better.constants import PARK_FACTORS
from better.models import BasePredictor
from better.utils.logging import get_logger

log = get_logger(__name__)

LEAGUE_AVG_RPG = 4.5


@dataclass
class MCParams:
    """Tunable parameters for the Monte Carlo simulator."""

    n_sims: int = 10_000
    use_negative_binomial: bool = True
    nb_dispersion: float = 4.0
    home_advantage_runs: float = 0.25
    league_avg_rpg: float = LEAGUE_AVG_RPG


class MonteCarloSimulator(BasePredictor):
    """Run-scoring Monte Carlo simulator."""

    def __init__(self, params: MCParams | None = None):
        self.params = params or MCParams(n_sims=settings.monte_carlo_sims)
        self._rng = np.random.default_rng(42)
        self._team_run_rates: dict[str, float] = {}
        self._team_run_allowed_rates: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "monte_carlo"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Estimate team run-scoring and run-allowing rates from training data."""
        teams = set(X["home_team"].unique()) | set(X["away_team"].unique())
        avg_rpg = self.params.league_avg_rpg

        for team in teams:
            home_mask = X["home_team"] == team
            away_mask = X["away_team"] == team

            # Use Pythagorean WP to back into run ratio
            home_pythag = X.loc[home_mask, "home_pythag_win_pct_30"].dropna()
            away_pythag = X.loc[away_mask, "away_pythag_win_pct_30"].dropna()
            all_pythag = pd.concat([home_pythag, away_pythag])

            if len(all_pythag) > 0:
                wp = np.clip(all_pythag.mean(), 0.3, 0.7)
                run_ratio = (wp / (1 - wp)) ** (1 / 1.83)
                self._team_run_rates[team] = avg_rpg * np.sqrt(run_ratio)
                self._team_run_allowed_rates[team] = avg_rpg / np.sqrt(run_ratio)
            else:
                self._team_run_rates[team] = avg_rpg
                self._team_run_allowed_rates[team] = avg_rpg

        log.info("monte_carlo_fitted", teams=len(self._team_run_rates))

    def _expected_runs(
        self,
        batting_team: str,
        pitching_team: str,
        park_team: str,
        is_home: bool,
    ) -> float:
        """Compute expected runs for batting_team vs pitching_team at park."""
        offense = self._team_run_rates.get(batting_team, LEAGUE_AVG_RPG)
        defense = self._team_run_allowed_rates.get(pitching_team, LEAGUE_AVG_RPG)
        avg = self.params.league_avg_rpg

        # Log5-style
        expected = (offense / avg) * (defense / avg) * avg

        # Park factor
        park_factor = PARK_FACTORS.get(park_team, {}).get("runs", 1.0)
        expected *= park_factor

        # Home advantage
        if is_home:
            expected += self.params.home_advantage_runs

        return max(expected, 0.5)

    def _simulate_runs(self, expected: float) -> np.ndarray:
        """Sample N run totals from the scoring distribution."""
        if self.params.use_negative_binomial:
            r = self.params.nb_dispersion
            p = r / (r + expected)
            return self._rng.negative_binomial(r, p, size=self.params.n_sims)
        else:
            return self._rng.poisson(expected, size=self.params.n_sims)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Simulate N games per row, return P(home_win)."""
        probs = np.empty(len(X))

        for i, row in enumerate(X.itertuples(index=False)):
            home_expected = self._expected_runs(
                row.home_team, row.away_team, row.home_team, is_home=True
            )
            away_expected = self._expected_runs(
                row.away_team, row.home_team, row.home_team, is_home=False
            )

            home_runs = self._simulate_runs(home_expected)
            away_runs = self._simulate_runs(away_expected)

            wins = np.sum(home_runs > away_runs)
            ties = np.sum(home_runs == away_runs)
            probs[i] = (wins + 0.5 * ties) / self.params.n_sims

        return probs

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "params": self.params,
                "team_run_rates": self._team_run_rates,
                "team_run_allowed_rates": self._team_run_allowed_rates,
            },
            path / "mc_state.joblib",
        )

    def load(self, path: Path) -> None:
        data = joblib.load(path / "mc_state.joblib")
        self.params = data["params"]
        self._team_run_rates = data["team_run_rates"]
        self._team_run_allowed_rates = data["team_run_allowed_rates"]
