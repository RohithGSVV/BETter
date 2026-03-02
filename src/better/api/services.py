"""Prediction service — shared core logic for API and dashboard.

Loads saved models, fetches today's schedule/odds, generates predictions,
and orchestrates the betting engine. Both FastAPI routes and Streamlit
pages import this module directly.
"""

from __future__ import annotations

import functools
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from better.config import settings
from better.utils.logging import get_logger

log = get_logger(__name__)


class PredictionService:
    """Loads models once, generates predictions, caches results.

    Designed as a singleton — call ``get_prediction_service()`` to obtain
    the shared instance.
    """

    def __init__(self) -> None:
        self._models_loaded = False
        self._gbm = None
        self._bayesian = None
        self._mc = None
        self._meta = None
        self._load_errors: dict[str, str] = {}

        # Caches (cleared by refresh_predictions)
        self._cached_predictions: list[dict] | None = None
        self._cached_schedule: list[dict] | None = None
        self._cache_date: date | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> dict[str, bool]:
        """Load all saved models from the models/ directory.

        Returns a dict of ``{model_name: loaded_successfully}``.
        """
        from better.training.persistence import load_model

        results: dict[str, bool] = {}

        # GBM Ensemble
        try:
            from better.models.gbm.ensemble import GBMEnsemble

            self._gbm = GBMEnsemble()
            load_model(self._gbm)
            results["gbm_ensemble"] = True
        except Exception as exc:
            self._load_errors["gbm_ensemble"] = str(exc)
            results["gbm_ensemble"] = False
            log.warning("gbm_load_failed", error=str(exc))

        # Bayesian Kalman
        try:
            from better.models.bayesian.kalman import BayesianKalmanModel

            self._bayesian = BayesianKalmanModel()
            load_model(self._bayesian)
            results["bayesian_kalman"] = True
        except Exception as exc:
            self._load_errors["bayesian_kalman"] = str(exc)
            results["bayesian_kalman"] = False
            log.warning("bayesian_load_failed", error=str(exc))

        # Monte Carlo
        try:
            from better.models.monte_carlo.simulator import MonteCarloSimulator

            self._mc = MonteCarloSimulator()
            load_model(self._mc)
            results["monte_carlo"] = True
        except Exception as exc:
            self._load_errors["monte_carlo"] = str(exc)
            results["monte_carlo"] = False
            log.warning("mc_load_failed", error=str(exc))

        # Meta-learner
        try:
            from better.models.meta.stacker import MetaLearner

            self._meta = MetaLearner()
            load_model(self._meta)
            results["meta_learner"] = True
        except Exception as exc:
            self._load_errors["meta_learner"] = str(exc)
            results["meta_learner"] = False
            log.warning("meta_load_failed", error=str(exc))

        self._models_loaded = any(results.values())
        log.info("models_loaded", results=results)
        return results

    # ------------------------------------------------------------------
    # Model status
    # ------------------------------------------------------------------

    def get_model_status(self) -> dict:
        """Return status of loaded models, file dates, and accuracy metrics."""
        models_loaded = {
            "gbm_ensemble": self._gbm is not None,
            "bayesian_kalman": self._bayesian is not None,
            "monte_carlo": self._mc is not None,
            "meta_learner": self._meta is not None,
        }

        # Try to read summary.csv for accuracy metrics
        details: dict[str, dict] = {}
        summary_path = settings.project_root / "results" / "summary.csv"
        if summary_path.exists():
            try:
                df = pd.read_csv(summary_path)
                for _, row in df.iterrows():
                    name = row.get("model", "")
                    details[name] = {
                        "avg_log_loss": round(row.get("avg_log_loss", 0), 4),
                        "avg_accuracy": round(row.get("avg_accuracy", 0), 4),
                        "avg_brier": round(row.get("avg_brier", 0), 4),
                    }
            except Exception:
                pass

        # Check model file dates
        last_training_date = None
        for model_name in ["gbm_ensemble", "bayesian_kalman", "monte_carlo", "meta_learner"]:
            model_path = settings.models_dir / model_name / "final"
            if model_path.exists():
                try:
                    mtime = max(f.stat().st_mtime for f in model_path.iterdir())
                    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
                    date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                    if model_name in details:
                        details[model_name]["last_modified"] = date_str
                    else:
                        details[model_name] = {"last_modified": date_str}
                    if last_training_date is None or dt.isoformat() > last_training_date:
                        last_training_date = date_str
                except Exception:
                    pass

        return {
            "models_loaded": models_loaded,
            "last_training_date": last_training_date,
            "model_details": details,
        }

    # ------------------------------------------------------------------
    # Today's schedule
    # ------------------------------------------------------------------

    def get_todays_schedule(self) -> list[dict]:
        """Fetch today's games from MLB Stats API."""
        today = date.today()

        # Use cache if same day
        if self._cached_schedule is not None and self._cache_date == today:
            return self._cached_schedule

        try:
            from better.data.ingest.mlb_api import MLBStatsClient

            client = MLBStatsClient()
            try:
                games = client.get_schedule(today)
            finally:
                client.close()

            self._cached_schedule = games
            self._cache_date = today
            return games
        except Exception as exc:
            log.error("schedule_fetch_failed", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Single-game prediction
    # ------------------------------------------------------------------

    def predict_game(self, home_team: str, away_team: str) -> dict[str, float]:
        """Run all loaded models on a single matchup.

        Returns ``{model_name: P(home_win)}``.
        """
        predictions: dict[str, float] = {}

        # Bayesian Kalman — simplest, just needs team names
        if self._bayesian is not None:
            try:
                p = self._bayesian._predict_game(home_team, away_team)
                predictions["bayesian_kalman"] = round(p, 4)
            except Exception as exc:
                log.warning("bayesian_predict_failed", error=str(exc))

        # Monte Carlo — needs team run rates (loaded from saved state)
        if self._mc is not None:
            try:
                home_exp = self._mc._expected_runs(home_team, away_team, home_team, True)
                away_exp = self._mc._expected_runs(away_team, home_team, home_team, False)
                home_runs = self._mc._simulate_runs(home_exp)
                away_runs = self._mc._simulate_runs(away_exp)
                wins = np.sum(home_runs > away_runs)
                ties = np.sum(home_runs == away_runs)
                p_home = (wins + 0.5 * ties) / self._mc.params.n_sims
                predictions["monte_carlo"] = round(float(p_home), 4)
            except Exception as exc:
                log.warning("mc_predict_failed", error=str(exc))

        # GBM requires feature data which may not be available for today.
        # For future: look up latest team_features_daily rows and build
        # a feature row for prediction.

        # Meta-learner from base predictions — pad missing models/features
        # so the feature vector matches what the model was trained on.
        if self._meta is not None and len(predictions) >= 2:
            try:
                expected_names: list[str] = getattr(
                    self._meta, "_meta_feature_names", []
                )

                if expected_names:
                    # Build a feature vector matching the training layout
                    row = np.zeros((1, len(expected_names)))
                    available_avg = sum(predictions.values()) / len(predictions)

                    for i, feat in enumerate(expected_names):
                        if feat.startswith("pred_"):
                            model_key = feat[len("pred_"):]
                            if model_key in predictions:
                                row[0, i] = predictions[model_key]
                            else:
                                # Impute missing base model with available avg
                                row[0, i] = available_avg
                        elif feat.startswith("raw_"):
                            # Raw features not available for live games — use 0
                            row[0, i] = 0.0

                    meta_p = self._meta._model.predict_proba(row)[:, 1]
                else:
                    # No feature-name metadata; fallback to old approach
                    base_preds = {
                        k: np.array([v]) for k, v in predictions.items()
                    }
                    meta_p = self._meta.predict_from_base_predictions(
                        base_preds
                    )

                predictions["meta_learner"] = round(float(meta_p[0]), 4)
            except Exception as exc:
                log.warning("meta_predict_failed", error=str(exc))

        # Consensus: average of all available models
        if predictions:
            predictions["consensus"] = round(
                sum(predictions.values()) / len(predictions), 4
            )

        return predictions

    # ------------------------------------------------------------------
    # Today's predictions (batch)
    # ------------------------------------------------------------------

    def get_todays_predictions(self) -> list[dict]:
        """Generate predictions for all of today's games.

        Includes odds and edge if available.
        """
        today = date.today()

        # Use cache if same day
        if self._cached_predictions is not None and self._cache_date == today:
            return self._cached_predictions

        schedule = self.get_todays_schedule()
        if not schedule:
            return []

        # Try to fetch odds
        odds_by_game = self._fetch_current_odds()

        results = []
        for game in schedule:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            if not home or not away:
                continue

            preds = self.predict_game(home, away)

            entry = {
                "game_pk": game.get("game_pk", 0),
                "game_date": str(game.get("game_date", today)),
                "home_team": home,
                "away_team": away,
                "home_sp_name": game.get("home_sp_name", ""),
                "away_sp_name": game.get("away_sp_name", ""),
                "bayesian_prob": preds.get("bayesian_kalman"),
                "gbm_prob": preds.get("gbm_ensemble"),
                "monte_carlo_prob": preds.get("monte_carlo"),
                "meta_prob": preds.get("meta_learner"),
                "consensus_prob": preds.get("consensus"),
                "market_implied_prob": None,
                "edge": None,
                "confidence": None,
            }

            # Add odds/edge if available
            game_pk = game.get("game_pk")
            if game_pk and game_pk in odds_by_game:
                odds_info = odds_by_game[game_pk]
                market_prob = odds_info.get("home_fair_prob")
                entry["market_implied_prob"] = market_prob
                best_prob = preds.get("meta_learner") or preds.get("consensus")
                if best_prob and market_prob:
                    entry["edge"] = round(best_prob - market_prob, 4)
                    edge = best_prob - market_prob
                    if edge >= 0.07:
                        entry["confidence"] = "high"
                    elif edge >= 0.04:
                        entry["confidence"] = "medium"
                    elif edge >= 0.03:
                        entry["confidence"] = "low"

            results.append(entry)

        self._cached_predictions = results
        self._cache_date = today
        return results

    # ------------------------------------------------------------------
    # Bet recommendations
    # ------------------------------------------------------------------

    def get_bet_recommendations(self) -> list[dict]:
        """Return bet recommendations for games with positive edge."""
        from better.betting.engine import BettingEngine

        predictions = self.get_todays_predictions()
        schedule = self.get_todays_schedule()
        odds_by_game = self._fetch_current_odds()

        engine = BettingEngine(
            bankroll=settings.initial_bankroll,
            min_edge=settings.min_edge_threshold,
            kelly_fraction=settings.kelly_fraction,
            max_bet_pct=settings.max_bet_pct,
        )

        recommendations = []
        for pred in predictions:
            game_pk = pred.get("game_pk")
            if not game_pk or game_pk not in odds_by_game:
                continue

            odds_info = odds_by_game[game_pk]
            best_prob = pred.get("meta_prob") or pred.get("consensus_prob")
            if best_prob is None:
                continue

            home_odds = odds_info.get("home_odds_american", -110)
            away_odds = odds_info.get("away_odds_american", -110)

            rec = engine.evaluate_game(
                model_home_prob=best_prob,
                home_odds_american=home_odds,
                away_odds_american=away_odds,
                home_team=pred["home_team"],
                away_team=pred["away_team"],
                game_date=date.today(),
                game_pk=game_pk,
            )

            if rec is not None:
                recommendations.append({
                    "game_pk": rec.game_pk,
                    "game_date": str(rec.game_date) if rec.game_date else None,
                    "home_team": rec.home_team,
                    "away_team": rec.away_team,
                    "bet_side": rec.bet_side,
                    "model_prob": rec.model_prob,
                    "market_prob": rec.market_prob,
                    "edge": rec.edge,
                    "expected_value": rec.expected_value,
                    "kelly_fraction": rec.kelly_fraction,
                    "bet_amount": rec.bet_amount,
                    "odds_american": rec.odds_american,
                    "bookmaker": rec.bookmaker,
                    "confidence": rec.confidence,
                })

        return recommendations

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def get_backtest_summary(
        self,
        edge_threshold: float = 0.03,
        kelly_fraction: float = 0.25,
        model: str = "meta_learner",
    ) -> dict:
        """Run backtest and return summary statistics."""
        result = self._run_backtest(edge_threshold, kelly_fraction, model)
        s = result.stats
        return {
            "initial_bankroll": s.initial_bankroll,
            "final_bankroll": round(s.current_bankroll, 2),
            "total_bets": s.total_bets,
            "wins": s.bets_won,
            "losses": s.bets_lost,
            "win_rate": round(s.win_rate, 4),
            "roi_pct": round(s.roi_pct, 2),
            "yield_pct": round(s.yield_pct, 2),
            "max_drawdown_pct": round(s.max_drawdown_pct, 2),
            "avg_edge": round(s.avg_edge, 4),
            "sharpe_ratio": round(s.sharpe_ratio, 2),
            "longest_losing_streak": s.longest_losing_streak,
            "edge_threshold": edge_threshold,
            "kelly_fraction": kelly_fraction,
            "model_used": model,
        }

    def get_backtest_bankroll_curve(
        self,
        edge_threshold: float = 0.03,
        kelly_fraction: float = 0.25,
        model: str = "meta_learner",
    ) -> list[dict]:
        """Return bankroll curve data for charting."""
        result = self._run_backtest(edge_threshold, kelly_fraction, model)
        curve = result.bankroll_curve

        if curve.empty:
            return []

        points = []
        for _, row in curve.iterrows():
            points.append({
                "game_date": str(row.get("game_date", "")),
                "season": int(row["season"]) if "season" in row else None,
                "bankroll": round(float(row["bankroll"]), 2),
            })
        return points

    @functools.lru_cache(maxsize=16)
    def _run_backtest(
        self,
        edge_threshold: float,
        kelly_fraction: float,
        model: str,
    ):
        """Run backtest with given params (cached)."""
        from better.betting.backtest import Backtester

        bt = Backtester(
            initial_bankroll=settings.initial_bankroll,
            min_edge=edge_threshold,
            kelly_fraction=kelly_fraction,
            model=model,
        )
        return bt.run()

    # ------------------------------------------------------------------
    # Edge / calibration analysis
    # ------------------------------------------------------------------

    def get_calibration_data(self, model: str = "meta_learner") -> dict:
        """Load OOF details and return calibration analysis data."""
        oof_path = settings.project_root / "results" / "oof_details.csv"
        if not oof_path.exists():
            return {"error": "OOF details not found. Run `better model train` first."}

        from better.betting.edge import EdgeAnalyzer

        oof = pd.read_csv(oof_path)
        analyzer = EdgeAnalyzer(oof, model=model)

        result: dict = {}

        try:
            cal = analyzer.calibration_report()
            result["calibration"] = cal.to_dict(orient="records")
        except Exception:
            result["calibration"] = []

        try:
            comparison = analyzer.model_comparison()
            result["model_comparison"] = comparison.to_dict(orient="records")
        except Exception:
            result["model_comparison"] = []

        try:
            by_month = analyzer.edge_by_month()
            result["edge_by_month"] = by_month.to_dict(orient="records") if not by_month.empty else []
        except Exception:
            result["edge_by_month"] = []

        try:
            roi = analyzer.roi_by_edge_threshold()
            result["roi_by_edge_threshold"] = roi.to_dict(orient="records")
        except Exception:
            result["roi_by_edge_threshold"] = []

        try:
            by_prob = analyzer.edge_by_probability_range()
            result["edge_by_probability_range"] = by_prob.to_dict(orient="records")
        except Exception:
            result["edge_by_probability_range"] = []

        return result

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def refresh_predictions(self) -> None:
        """Clear cached predictions (called by daily scheduler)."""
        self._cached_predictions = None
        self._cached_schedule = None
        self._cache_date = None
        self._run_backtest.cache_clear()
        log.info("prediction_cache_cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_current_odds(self) -> dict[int, dict]:
        """Fetch current odds, returning a dict keyed by game_pk."""
        try:
            from better.data.ingest.odds import OddsClient, parse_odds_response

            if not settings.odds_api_key:
                return {}

            client = OddsClient()
            try:
                games = client.get_odds()
            finally:
                client.close()

            if not games:
                return {}

            df = parse_odds_response(games)
            if df.empty:
                return {}

            # Group by game_pk, take median odds across bookmakers
            grouped = df.groupby("game_pk").agg({
                "home_fair_prob": "median",
                "away_fair_prob": "median",
                "home_odds_american": "median",
                "away_odds_american": "median",
            }).to_dict(orient="index")

            # Convert median odds to int
            result = {}
            for gpk, vals in grouped.items():
                result[gpk] = {
                    "home_fair_prob": round(vals["home_fair_prob"], 4),
                    "away_fair_prob": round(vals["away_fair_prob"], 4),
                    "home_odds_american": int(round(vals["home_odds_american"])),
                    "away_odds_american": int(round(vals["away_odds_american"])),
                }
            return result

        except Exception as exc:
            log.warning("odds_fetch_failed", error=str(exc))
            return {}


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """Get or create the singleton PredictionService."""
    global _service
    if _service is None:
        _service = PredictionService()
        _service.load_models()
    return _service
