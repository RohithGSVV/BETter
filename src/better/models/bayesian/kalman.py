"""Bayesian state-space model using a Kalman-inspired team-strength filter.

Each team's "true strength" is a latent variable that evolves game-by-game.
After each observed result, the belief about team strength is updated using
a Kalman-like update rule.  For prediction, the strength difference plus a
home-field advantage term is passed through a logistic function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit  # logistic sigmoid

from better.constants import HOME_FIELD_ADVANTAGE_LOGIT
from better.models import BasePredictor
from better.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TeamState:
    """Kalman state for one team: mean strength and variance."""

    mu: float = 0.0
    sigma2: float = 1.0


@dataclass
class KalmanParams:
    """Hyperparameters for the Kalman filter."""

    process_noise: float = 0.002
    observation_noise: float = 1.0
    home_advantage: float = HOME_FIELD_ADVANTAGE_LOGIT
    season_reversion: float = 0.33
    prior_mu: float = 0.0
    prior_sigma2: float = 1.0


class BayesianKalmanModel(BasePredictor):
    """Kalman-filter-based team-strength model."""

    def __init__(self, params: KalmanParams | None = None):
        self.params = params or KalmanParams()
        self.team_states: dict[str, TeamState] = {}

    @property
    def name(self) -> str:
        return "bayesian_kalman"

    def _get_state(self, team: str) -> TeamState:
        if team not in self.team_states:
            self.team_states[team] = TeamState(
                mu=self.params.prior_mu,
                sigma2=self.params.prior_sigma2,
            )
        return self.team_states[team]

    def _predict_game(self, home_team: str, away_team: str) -> float:
        """Predict P(home_win) from current team states."""
        h = self._get_state(home_team)
        a = self._get_state(away_team)
        logit_diff = h.mu - a.mu + self.params.home_advantage
        return float(expit(logit_diff))

    def _update_after_game(
        self, home_team: str, away_team: str, home_win: bool
    ) -> None:
        """Kalman update step: observe game outcome, update beliefs."""
        h = self._get_state(home_team)
        a = self._get_state(away_team)

        # Prediction step: add process noise
        h.sigma2 += self.params.process_noise
        a.sigma2 += self.params.process_noise

        # Predicted probability
        p_home = self._predict_game(home_team, away_team)
        outcome = 1.0 if home_win else 0.0
        surprise = outcome - p_home

        # Kalman gain
        total_var = h.sigma2 + a.sigma2 + self.params.observation_noise
        k_home = h.sigma2 / total_var
        k_away = a.sigma2 / total_var

        # Update means
        h.mu += k_home * surprise
        a.mu -= k_away * surprise

        # Update variances
        h.sigma2 *= 1 - k_home
        a.sigma2 *= 1 - k_away

    def _season_boundary(self) -> None:
        """Regress all team strengths toward zero between seasons."""
        for state in self.team_states.values():
            state.mu *= 1 - self.params.season_reversion
            state.sigma2 = self.params.prior_sigma2

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Process all games chronologically to build team states."""
        self.team_states = {}
        df = X[["game_date", "season", "home_team", "away_team"]].copy()
        df["home_win"] = y.values
        df = df.sort_values("game_date").reset_index(drop=True)

        current_season = None
        for row in df.itertuples(index=False):
            if row.season != current_season:
                if current_season is not None:
                    self._season_boundary()
                current_season = row.season
            self._update_after_game(row.home_team, row.away_team, bool(row.home_win))

        log.info("bayesian_kalman_fitted", teams=len(self.team_states))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict P(home_win) for each row using current team states."""
        probs = np.array([
            self._predict_game(row.home_team, row.away_team)
            for row in X[["home_team", "away_team"]].itertuples(index=False)
        ])
        return probs

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"params": self.params, "team_states": self.team_states},
            path / "kalman_state.joblib",
        )

    def load(self, path: Path) -> None:
        data = joblib.load(path / "kalman_state.joblib")
        self.params = data["params"]
        self.team_states = data["team_states"]

    def get_team_ratings(self) -> pd.DataFrame:
        """Return current team ratings for inspection."""
        rows = [
            {"team": t, "mu": s.mu, "sigma2": s.sigma2}
            for t, s in self.team_states.items()
        ]
        return pd.DataFrame(rows).sort_values("mu", ascending=False)
