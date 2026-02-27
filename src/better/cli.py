"""BETter CLI — powered by Typer.

Usage::

    uv run better features build
    uv run better features training-set
    uv run better features elo
    uv run better model train [--skip-tuning] [--n-trials 50]
    uv run better model evaluate
    uv run better model predict NYY BOS
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(name="better", help="BETter MLB Prediction System")
console = Console()

features_app = typer.Typer(help="Feature engineering commands")
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build(
    start_year: int = typer.Option(None, help="Start year (default: config)"),
    end_year: int = typer.Option(None, help="End year (default: config)"),
    skip_pitcher: bool = typer.Option(False, help="Skip pitcher features"),
) -> None:
    """Run the full feature engineering pipeline."""
    from better.features.pipeline import run_feature_pipeline

    results = run_feature_pipeline(start_year, end_year, skip_pitcher)
    console.print(f"[green]Feature pipeline complete:[/green] {results}")


@features_app.command("training-set")
def features_training_set(
    start_year: int = typer.Option(None, help="Start year"),
    end_year: int = typer.Option(None, help="End year"),
    output: str = typer.Option("data/training_set.parquet", help="Output path"),
) -> None:
    """Build and export the training DataFrame as Parquet."""
    from better.features.pipeline import build_training_set

    df = build_training_set(start_year, end_year)
    df.to_parquet(output, index=False)
    console.print(f"[green]Training set exported:[/green] {len(df)} rows → {output}")


@features_app.command("elo")
def features_elo(
    start_year: int = typer.Option(None, help="Start year"),
    end_year: int = typer.Option(None, help="End year"),
    top: int = typer.Option(10, help="Show top N teams"),
) -> None:
    """Compute Elo ratings and display the leaderboard."""
    from better.features.elo import build_elo_ratings

    elo_df = build_elo_ratings(start_year or 2010, end_year or 2025)
    latest = (
        elo_df.sort_values("as_of_date")
        .drop_duplicates("team", keep="last")
        .sort_values("elo_rating", ascending=False)
        .head(top)
    )
    console.print(latest[["team", "as_of_date", "elo_rating"]].to_string(index=False))


# ── Model commands ──────────────────────────────────────────────────────

model_app = typer.Typer(help="Model training and prediction commands")
app.add_typer(model_app, name="model")


@model_app.command("train")
def model_train(
    skip_tuning: bool = typer.Option(False, help="Skip Optuna hyperparameter tuning"),
    n_trials: int = typer.Option(50, help="Number of Optuna trials per model"),
) -> None:
    """Train all models (GBM ensemble, Bayesian, Monte Carlo, Meta-learner)."""
    from better.training.orchestrator import run_full_training_pipeline

    summary = run_full_training_pipeline(
        skip_tuning=skip_tuning,
        n_tuning_trials=n_trials,
    )
    console.print("\n[bold green]Training Complete[/bold green]")
    console.print(summary.to_string(index=False))


@model_app.command("evaluate")
def model_evaluate() -> None:
    """Show evaluation results from saved OOF predictions."""
    from better.training.persistence import load_oof_predictions

    oof_data = load_oof_predictions()
    console.print(f"[green]Loaded OOF predictions for:[/green] {list(oof_data.keys())}")
    for model_name, fold_preds in oof_data.items():
        for year, preds in fold_preds.items():
            console.print(f"  {model_name} fold {year}: {len(preds)} predictions")


@model_app.command("predict")
def model_predict(
    home_team: str = typer.Argument(..., help="Home team abbreviation (e.g. NYY)"),
    away_team: str = typer.Argument(..., help="Away team abbreviation (e.g. BOS)"),
) -> None:
    """Generate a prediction for a specific matchup using saved models."""
    from better.models.bayesian.kalman import BayesianKalmanModel
    from better.models.monte_carlo.simulator import MonteCarloSimulator
    from better.training.persistence import load_model

    console.print(f"[bold]{away_team} @ {home_team}[/bold]")

    bayesian = BayesianKalmanModel()
    load_model(bayesian)
    p_bayesian = bayesian._predict_game(home_team, away_team)
    console.print(f"  Bayesian Kalman:  P(home) = {p_bayesian:.3f}")

    mc = MonteCarloSimulator()
    load_model(mc)
    home_exp = mc._expected_runs(home_team, away_team, home_team, True)
    away_exp = mc._expected_runs(away_team, home_team, home_team, False)
    console.print(f"  Monte Carlo:      Expected runs {home_team} {home_exp:.1f} - {away_team} {away_exp:.1f}")


if __name__ == "__main__":
    app()
