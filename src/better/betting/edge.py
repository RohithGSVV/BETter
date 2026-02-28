"""Edge analysis and calibration diagnostics.

Answers: Where does the model have edge? Is it well-calibrated?
Which teams/months/odds ranges are most profitable?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from better.utils.logging import get_logger

log = get_logger(__name__)


class EdgeAnalyzer:
    """Analyzes model prediction quality and edge patterns.

    Parameters
    ----------
    oof : pd.DataFrame
        OOF details DataFrame with columns:
        game_date, home_team, away_team, home_win,
        elo_home_win_prob, meta_learner_prob (and other model probs).
    model : str
        Which model column to analyze. Default "meta_learner".
    """

    def __init__(self, oof: pd.DataFrame, model: str = "meta_learner"):
        self.oof = oof.copy()
        self.model = model
        self._prob_col = f"{model}_prob"
        self._market_col = "elo_home_win_prob"

        if self._prob_col not in self.oof.columns:
            raise ValueError(f"Column '{self._prob_col}' not found in data")

    def calibration_report(self, n_bins: int = 10) -> pd.DataFrame:
        """Check if predicted probabilities match actual outcomes.

        Groups predictions into bins and compares predicted vs actual win rate.
        A well-calibrated model: when it says 60%, home wins ~60% of the time.
        """
        df = self.oof.dropna(subset=[self._prob_col, "home_win"]).copy()
        df["prob_bin"] = pd.cut(df[self._prob_col], bins=n_bins)

        cal = (
            df.groupby("prob_bin", observed=True)
            .agg(
                count=("home_win", "count"),
                actual_win_rate=("home_win", "mean"),
                predicted_prob=((self._prob_col), "mean"),
            )
            .assign(
                calibration_error=lambda d: (d["predicted_prob"] - d["actual_win_rate"]).abs(),
            )
            .reset_index()
        )

        # Overall calibration error (ECE — expected calibration error)
        weights = cal["count"] / cal["count"].sum()
        ece = (weights * cal["calibration_error"]).sum()
        log.info("calibration_report", model=self.model, ece=round(ece, 4), n_bins=n_bins)

        return cal

    def edge_by_team(self) -> pd.DataFrame:
        """Analyze model edge broken down by home team.

        Shows which teams the model predicts best/worst relative to market.
        """
        df = self.oof.dropna(subset=[self._prob_col, self._market_col, "home_win"]).copy()
        df["edge"] = df[self._prob_col] - df[self._market_col]
        df["correct"] = ((df[self._prob_col] >= 0.5) == df["home_win"].astype(bool))
        df["market_correct"] = ((df[self._market_col] >= 0.5) == df["home_win"].astype(bool))

        if "home_team" not in df.columns:
            return pd.DataFrame()

        return (
            df.groupby("home_team")
            .agg(
                games=("home_win", "count"),
                model_accuracy=("correct", "mean"),
                market_accuracy=("market_correct", "mean"),
                avg_edge=("edge", "mean"),
                avg_model_prob=((self._prob_col), "mean"),
                actual_win_rate=("home_win", "mean"),
            )
            .assign(
                accuracy_lift=lambda d: d["model_accuracy"] - d["market_accuracy"],
            )
            .sort_values("accuracy_lift", ascending=False)
            .reset_index()
        )

    def edge_by_month(self) -> pd.DataFrame:
        """Analyze model edge by month of the season.

        Shows when during the season the model is strongest.
        """
        df = self.oof.dropna(subset=[self._prob_col, self._market_col, "home_win"]).copy()

        if "game_date" not in df.columns:
            return pd.DataFrame()

        df["month"] = pd.to_datetime(df["game_date"]).dt.month
        df["correct"] = ((df[self._prob_col] >= 0.5) == df["home_win"].astype(bool))
        df["market_correct"] = ((df[self._market_col] >= 0.5) == df["home_win"].astype(bool))
        df["edge"] = df[self._prob_col] - df[self._market_col]

        month_names = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}

        result = (
            df.groupby("month")
            .agg(
                games=("home_win", "count"),
                model_accuracy=("correct", "mean"),
                market_accuracy=("market_correct", "mean"),
                avg_edge=("edge", "mean"),
            )
            .assign(
                accuracy_lift=lambda d: d["model_accuracy"] - d["market_accuracy"],
                month_name=lambda d: d.index.map(month_names),
            )
            .reset_index()
        )
        return result

    def edge_by_probability_range(self) -> pd.DataFrame:
        """Analyze edge by the model's predicted probability range.

        Shows if the model is better at calling strong favorites, toss-ups, or underdogs.
        """
        df = self.oof.dropna(subset=[self._prob_col, self._market_col, "home_win"]).copy()
        df["edge"] = df[self._prob_col] - df[self._market_col]
        df["correct"] = ((df[self._prob_col] >= 0.5) == df["home_win"].astype(bool))

        bins = [0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0]
        labels = ["<40%", "40-45%", "45-50%", "50-55%", "55-60%", ">60%"]
        df["prob_range"] = pd.cut(df[self._prob_col], bins=bins, labels=labels)

        return (
            df.groupby("prob_range", observed=True)
            .agg(
                games=("home_win", "count"),
                accuracy=("correct", "mean"),
                avg_edge=("edge", "mean"),
                actual_win_rate=("home_win", "mean"),
            )
            .reset_index()
        )

    def roi_by_edge_threshold(
        self,
        thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        """Compute hypothetical ROI at various edge thresholds.

        Simple analysis (no compounding): just checks if bets placed at
        each threshold level would have been profitable.
        """
        thresholds = thresholds or [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
        df = self.oof.dropna(subset=[self._prob_col, self._market_col, "home_win"]).copy()
        df["edge_home"] = df[self._prob_col] - df[self._market_col]
        df["edge_away"] = (1 - df[self._prob_col]) - (1 - df[self._market_col])

        rows = []
        for threshold in thresholds:
            # Bet home when model > market + threshold
            home_bets = df[df["edge_home"] > threshold].copy()
            home_wins = home_bets["home_win"].sum()

            # Bet away when model < market - threshold  (i.e., edge_away > threshold)
            away_bets = df[df["edge_away"] > threshold].copy()
            away_wins = (~away_bets["home_win"].astype(bool)).sum()

            total_bets = len(home_bets) + len(away_bets)
            total_wins = home_wins + away_wins
            win_rate = total_wins / total_bets if total_bets > 0 else 0.0

            rows.append({
                "edge_threshold": threshold,
                "total_bets": total_bets,
                "total_wins": int(total_wins),
                "win_rate": round(win_rate, 3),
                "bets_pct_of_games": round(total_bets / len(df) * 100, 1) if len(df) > 0 else 0,
            })

        return pd.DataFrame(rows)

    def model_comparison(self) -> pd.DataFrame:
        """Compare all available models' accuracy and edge vs Elo market."""
        df = self.oof.dropna(subset=[self._market_col, "home_win"]).copy()

        model_cols = [c for c in df.columns if c.endswith("_prob") and c != self._market_col]
        rows = []

        for col in model_cols:
            model_name = col.replace("_prob", "")
            valid = df.dropna(subset=[col])
            correct = ((valid[col] >= 0.5) == valid["home_win"].astype(bool))
            edge = valid[col] - valid[self._market_col]

            rows.append({
                "model": model_name,
                "games": len(valid),
                "accuracy": round(correct.mean(), 4),
                "avg_edge_vs_elo": round(edge.mean(), 4),
                "std_edge": round(edge.std(), 4),
            })

        # Also add Elo (the market baseline)
        elo_correct = ((df[self._market_col] >= 0.5) == df["home_win"].astype(bool))
        rows.append({
            "model": "elo_market",
            "games": len(df),
            "accuracy": round(elo_correct.mean(), 4),
            "avg_edge_vs_elo": 0.0,
            "std_edge": 0.0,
        })

        return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
