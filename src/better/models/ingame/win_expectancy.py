"""In-game win probability model using Win Expectancy tables.

Uses historical MLB game data to build a Win Expectancy (WE) table that maps
every possible game situation (inning, half, outs, runners, score_diff) to
the empirical probability that the home team won from that state.

For live predictions, the model blends:
1. WE table lookup (primary signal, captures in-game context)
2. Pre-game model probability (Bayesian prior)

The blend weight shifts toward the WE table as the game progresses —
by the 7th inning, the game state dominates; by the 9th, the pre-game
prior barely matters.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from better.config import settings
from better.data.db import execute, fetch_df, get_connection
from better.utils.logging import get_logger

if TYPE_CHECKING:
    from better.data.live.game_feed import GameState

log = get_logger(__name__)

# ── Theoretical base rates (Tom Tango WE table approximation) ───────────────
# Fallback for states with too few or zero historical samples.
# These are the "textbook" win probabilities at the start of each half-inning,
# 0 outs, no runners, score tied.  Used only as smoothing anchors.
_BASE_HOME_WIN_PROB = 0.535  # slight home-field advantage


def populate_win_expectancy_table() -> int:
    """Build the WE table from historical play-by-play data in DuckDB.

    Uses the games table to compute empirical P(home_win) for each distinct
    game situation.  For states with very few samples, applies Bayesian
    smoothing toward the theoretical prior.

    Returns the number of rows inserted.
    """
    conn = get_connection()

    # Clear existing data
    execute("DELETE FROM win_expectancy")

    # Build WE from historical game results.
    # We don't have pitch-level play-by-play in DuckDB, so we use a
    # mathematical model based on run-scoring distributions from our
    # historical game data.
    #
    # For each (inning, half, outs, runners, score_diff):
    #   P(home_win) = f(expected_runs_remaining, score_diff)

    rows = _generate_we_table()

    if rows:
        df = pd.DataFrame(rows)
        conn.execute("INSERT INTO win_expectancy SELECT * FROM df")
        log.info("win_expectancy_populated", rows=len(df))
        return len(df)

    return 0


def _generate_we_table() -> list[dict]:
    """Generate all possible game states and their win probabilities.

    Uses a run-expectancy model based on historical scoring rates to
    compute P(home_win) for every (inning, half, outs, runners, score_diff).
    """
    # Historical average runs per half-inning (MLB long-run average ~0.5)
    avg_runs_per_half = _get_historical_runs_per_half() or 0.50

    rows: list[dict] = []

    # Innings 1-9 (extra innings handled as inning 10+)
    for inning in range(1, 13):
        for half in ("top", "bot"):
            for outs in range(3):
                for runners in range(8):  # 0b000 to 0b111
                    for score_diff in range(-15, 16):
                        wp = _compute_win_prob(
                            inning, half, outs, runners,
                            score_diff, avg_runs_per_half,
                        )
                        rows.append({
                            "inning": inning,
                            "half": half,
                            "outs": outs,
                            "runners": runners,
                            "score_diff": score_diff,
                            "win_prob": round(wp, 4),
                            "sample_size": 0,  # model-derived, not empirical
                        })

    return rows


def _get_historical_runs_per_half() -> float | None:
    """Get average runs per team per game from historical data."""
    try:
        result = execute("""
            SELECT AVG(home_score + away_score) / 2.0 / 9.0
            FROM games
            WHERE season >= 2015
        """).fetchone()
        if result and result[0]:
            return float(result[0])
    except Exception:
        pass
    return None


def _compute_win_prob(
    inning: int,
    half: str,
    outs: int,
    runners: int,
    score_diff: int,
    avg_runs_per_half: float,
) -> float:
    """Compute P(home_win) for a specific game state.

    Uses a simplified run-distribution model:
    1. Estimate remaining half-innings for each team
    2. Estimate expected runs remaining for each team
    3. Use a Poisson/normal approximation to compute
       P(home_runs_remaining > away_runs_remaining + deficit)
    """
    # Half-innings remaining for each team
    if half == "top":
        # Top of inning: away is batting
        # Away has: (3 - outs) worth of this half + remaining full half-innings
        away_half_innings_left = _partial_half(outs) + max(0, 9 - inning)
        home_half_innings_left = max(0, 9 - inning) + 1  # bottom of this + rest
        if inning >= 9:
            # Walk-off scenario: home may not bat if ahead
            home_half_innings_left = max(0, 10 - inning)
    else:
        # Bottom of inning: home is batting
        away_half_innings_left = max(0, 9 - inning)
        home_half_innings_left = _partial_half(outs) + max(0, 9 - inning)
        if inning >= 9:
            away_half_innings_left = max(0, 10 - inning)

    # Runners on base add expected runs in this half-inning
    runner_bonus = _runner_expected_runs(runners, outs)

    # Expected remaining runs
    if half == "top":
        away_exp = away_half_innings_left * avg_runs_per_half + runner_bonus
        home_exp = home_half_innings_left * avg_runs_per_half
    else:
        away_exp = away_half_innings_left * avg_runs_per_half
        home_exp = home_half_innings_left * avg_runs_per_half + runner_bonus

    # Home advantage factor
    home_exp *= 1.04  # ~4% home advantage in run scoring

    # P(home wins) using normal approximation
    # Home needs to overcome score_diff: positive = home leads
    home_needs = -score_diff  # runs home needs to outscore away by
    mean_diff = home_exp - away_exp - home_needs
    # Variance scales with expected runs
    variance = (home_exp + away_exp) * 1.1  # slightly overdispersed
    if variance < 0.01:
        # Game essentially over
        return 1.0 if score_diff > 0 else (0.5 if score_diff == 0 else 0.0)

    std = np.sqrt(variance)
    # P(home_runs - away_runs > home_needs)
    # = P(Z > -mean_diff / std) = Phi(mean_diff / std)
    z = mean_diff / std

    # Use the CDF of standard normal
    from scipy.stats import norm

    wp = float(norm.cdf(z))

    # Clamp to reasonable bounds
    return max(0.001, min(0.999, wp))


def _partial_half(outs: int) -> float:
    """Fraction of a half-inning remaining based on current outs."""
    return (3 - outs) / 3.0


def _runner_expected_runs(runners: int, outs: int) -> float:
    """Expected runs from current base-runner configuration.

    Uses the standard run-expectancy matrix values (MLB averages).
    runners is 3-bit encoded: bit2=1B, bit1=2B, bit0=3B
    """
    # Run expectancy matrix (outs → {runner_state: expected_runs})
    # Values from historical MLB averages (Tom Tango's tables)
    RE = {
        0: {  # 0 outs
            0b000: 0.00, 0b100: 0.54, 0b010: 0.67, 0b001: 0.41,
            0b110: 1.10, 0b101: 0.90, 0b011: 1.02, 0b111: 1.54,
        },
        1: {  # 1 out
            0b000: 0.00, 0b100: 0.29, 0b010: 0.41, 0b001: 0.23,
            0b110: 0.62, 0b101: 0.49, 0b011: 0.56, 0b111: 0.82,
        },
        2: {  # 2 outs
            0b000: 0.00, 0b100: 0.12, 0b010: 0.19, 0b001: 0.10,
            0b110: 0.26, 0b101: 0.19, 0b011: 0.23, 0b111: 0.33,
        },
    }
    return RE.get(outs, RE[2]).get(runners, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Live Win Probability Model
# ─────────────────────────────────────────────────────────────────────────────

class LiveWinProbModel:
    """In-game win probability model for live predictions.

    Blends:
    1. Win Expectancy table lookup (in-game state → P(home_win))
    2. Pre-game model prior (pregame Bayesian/Meta probability)

    The blend weight shifts toward WE as the game progresses:
    - Innings 1-3: 40% WE + 60% pre-game prior
    - Innings 4-6: 70% WE + 30% pre-game prior
    - Innings 7+:  90% WE + 10% pre-game prior
    - Innings 9+:  98% WE +  2% pre-game prior
    """

    def __init__(self) -> None:
        self._we_cache: dict[tuple, float] = {}
        self._loaded = False

    def load(self) -> bool:
        """Load the WE table into memory for fast lookups."""
        try:
            df = fetch_df(
                "SELECT inning, half, outs, runners, score_diff, win_prob "
                "FROM win_expectancy"
            )
            if df.empty:
                log.warning("we_table_empty")
                return False

            for _, row in df.iterrows():
                key = (
                    int(row["inning"]),
                    row["half"],
                    int(row["outs"]),
                    int(row["runners"]),
                    int(row["score_diff"]),
                )
                self._we_cache[key] = float(row["win_prob"])

            self._loaded = True
            log.info("we_table_loaded", states=len(self._we_cache))
            return True
        except Exception as exc:
            log.error("we_table_load_failed", error=str(exc))
            return False

    def predict(
        self,
        state: GameState,
        pregame_prob: float | None = None,
    ) -> dict:
        """Predict P(home_win) for the current game state.

        Args:
            state: Current in-game state (inning, outs, runners, score).
            pregame_prob: Pre-game model P(home_win), used as a prior.

        Returns:
            dict with keys: win_prob, we_prob, pregame_prob, we_weight,
                            inning, half, outs, runners, score_diff,
                            home_score, away_score
        """
        # WE table lookup
        we_prob = self._lookup_we(state)

        # Determine blend weight based on game progress
        we_weight = self._get_we_weight(state.inning)

        # Blend with pre-game prior
        if pregame_prob is not None:
            blended = we_weight * we_prob + (1 - we_weight) * pregame_prob
        else:
            blended = we_prob

        blended = max(0.001, min(0.999, blended))

        return {
            "win_prob": round(blended, 4),
            "we_prob": round(we_prob, 4),
            "pregame_prob": round(pregame_prob, 4) if pregame_prob else None,
            "we_weight": round(we_weight, 2),
            "inning": state.inning,
            "half": state.half,
            "outs": state.outs,
            "runners": state.runners,
            "score_diff": state.score_diff,
            "home_score": state.home_score,
            "away_score": state.away_score,
        }

    def _lookup_we(self, state: GameState) -> float:
        """Look up win expectancy from the cached table."""
        if not self._loaded:
            return _BASE_HOME_WIN_PROB

        key = state.state_key

        # Direct lookup
        if key in self._we_cache:
            return self._we_cache[key]

        # Clamp extreme score diffs to table bounds
        inning, half, outs, runners, score_diff = key
        clamped_diff = max(-15, min(15, score_diff))
        clamped_key = (min(inning, 12), half, outs, runners, clamped_diff)
        if clamped_key in self._we_cache:
            return self._we_cache[clamped_key]

        # Ultimate fallback: use mathematical model
        avg_runs = 0.50
        return _compute_win_prob(inning, half, outs, runners, score_diff, avg_runs)

    @staticmethod
    def _get_we_weight(inning: int) -> float:
        """Return blend weight for WE (vs pre-game prior) based on inning."""
        if inning <= 3:
            return 0.40
        elif inning <= 6:
            return 0.70
        elif inning <= 8:
            return 0.90
        else:
            return 0.98
