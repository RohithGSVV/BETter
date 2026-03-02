"""BETter CLI — powered by Typer.

Usage::

    uv run better features build
    uv run better features training-set
    uv run better features elo
    uv run better model train [--skip-tuning] [--n-trials 50]
    uv run better model evaluate
    uv run better model predict NYY BOS
    uv run better bet backtest [--edge-threshold 0.03] [--model meta_learner]
    uv run better bet sweep [--model meta_learner]
    uv run better bet edge [--model meta_learner]
    uv run better api serve [--port 8000]
    uv run better dashboard [--port 8501]
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

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
    console.print(f"[green]Training set exported:[/green] {len(df)} rows -> {output}")


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
    """Generate a prediction for a specific matchup using all saved models."""
    from better.models.bayesian.kalman import BayesianKalmanModel
    from better.models.monte_carlo.simulator import MonteCarloSimulator
    from better.training.persistence import load_model

    console.print(f"\n[bold]{away_team} @ {home_team}[/bold]\n")

    # Bayesian Kalman
    bayesian = BayesianKalmanModel()
    load_model(bayesian)
    p_bayesian = bayesian._predict_game(home_team, away_team)

    # Monte Carlo
    mc = MonteCarloSimulator()
    load_model(mc)
    p_mc_home = 0.5
    try:
        home_exp = mc._expected_runs(home_team, away_team, home_team, True)
        away_exp = mc._expected_runs(away_team, home_team, home_team, False)
        # Run quick simulation to get probability
        import numpy as np
        rng = np.random.default_rng(42)
        wins = 0
        n_sims = 10_000
        for _ in range(n_sims):
            h_runs = rng.negative_binomial(4, 4 / (4 + home_exp))
            a_runs = rng.negative_binomial(4, 4 / (4 + away_exp))
            if h_runs > a_runs:
                wins += 1
            elif h_runs == a_runs:
                wins += 0.5
        p_mc_home = wins / n_sims
    except Exception:
        pass

    # Results table
    table = Table(title="Model Predictions")
    table.add_column("Model", style="cyan")
    table.add_column("P(Home Win)", justify="right")
    table.add_column("P(Away Win)", justify="right")

    table.add_row("Bayesian Kalman", f"{p_bayesian:.1%}", f"{1-p_bayesian:.1%}")
    table.add_row("Monte Carlo", f"{p_mc_home:.1%}", f"{1-p_mc_home:.1%}")
    console.print(table)

    # Try to fetch odds (if API key configured)
    try:
        from better.config import settings
        if settings.odds_api_key:
            console.print("\n[dim]Fetching live odds...[/dim]")
            from better.data.ingest.odds import OddsClient
            client = OddsClient()
            games = client.get_odds()
            client.close()
            console.print(f"[green]Found {len(games)} upcoming games with odds[/green]")
    except Exception:
        pass


# ── Betting commands ──────────────────────────────────────────────────────

bet_app = typer.Typer(help="Betting engine and backtesting commands")
app.add_typer(bet_app, name="bet")


@bet_app.command("backtest")
def bet_backtest(
    edge_threshold: float = typer.Option(0.03, help="Minimum edge to place a bet"),
    kelly_fraction: float = typer.Option(0.25, help="Fraction of Kelly criterion"),
    max_bet_pct: float = typer.Option(0.05, help="Max bet as fraction of bankroll"),
    bankroll: float = typer.Option(1000.0, help="Initial bankroll"),
    model: str = typer.Option("meta_learner", help="Model to use for predictions"),
) -> None:
    """Run historical backtest using OOF predictions vs Elo market proxy."""
    from better.betting.backtest import Backtester

    bt = Backtester(
        initial_bankroll=bankroll,
        min_edge=edge_threshold,
        kelly_fraction=kelly_fraction,
        max_bet_pct=max_bet_pct,
        model=model,
    )
    result = bt.run()
    result.print_summary()


@bet_app.command("sweep")
def bet_sweep(
    model: str = typer.Option("meta_learner", help="Model to use"),
    bankroll: float = typer.Option(1000.0, help="Initial bankroll"),
    kelly_fraction: float = typer.Option(0.25, help="Fraction of Kelly criterion"),
) -> None:
    """Sweep edge thresholds to find optimal betting parameters."""
    from better.betting.backtest import Backtester

    bt = Backtester(
        initial_bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        model=model,
    )
    sweep = bt.sweep_edge_thresholds()
    console.print(f"\n[bold]Edge Threshold Sweep — {model}[/bold]\n")
    console.print(sweep.to_string(index=False))


@bet_app.command("edge")
def bet_edge(
    model: str = typer.Option("meta_learner", help="Model to analyze"),
) -> None:
    """Analyze model edge: calibration, by-team, by-month, model comparison."""
    import pandas as pd
    from better.config import settings
    from better.betting.edge import EdgeAnalyzer

    oof_path = settings.project_root / "results" / "oof_details.csv"
    if not oof_path.exists():
        console.print("[red]OOF details not found. Run `better model train` first.[/red]")
        raise typer.Exit(1)

    oof = pd.read_csv(oof_path)
    analyzer = EdgeAnalyzer(oof, model=model)

    # Model comparison
    console.print("\n[bold]Model Comparison vs Elo Market[/bold]")
    console.print(analyzer.model_comparison().to_string(index=False))

    # Calibration
    console.print(f"\n[bold]Calibration Report — {model}[/bold]")
    console.print(analyzer.calibration_report().to_string(index=False))

    # ROI by edge threshold
    console.print(f"\n[bold]Win Rate by Edge Threshold[/bold]")
    console.print(analyzer.roi_by_edge_threshold().to_string(index=False))

    # Edge by probability range
    console.print(f"\n[bold]Edge by Probability Range[/bold]")
    console.print(analyzer.edge_by_probability_range().to_string(index=False))

    # Edge by month
    by_month = analyzer.edge_by_month()
    if not by_month.empty:
        console.print(f"\n[bold]Edge by Month[/bold]")
        console.print(by_month.to_string(index=False))


@bet_app.command("generate-oof")
def bet_generate_oof() -> None:
    """Regenerate OOF details CSV from saved predictions (no retraining)."""
    import numpy as np
    import pandas as pd
    from better.config import settings
    from better.features.pipeline import build_training_set
    from better.models import IDENTIFIER_COLS, TARGET_COL
    from better.training.persistence import load_oof_predictions
    from better.training.splits import generate_walk_forward_splits
    from better.models.meta.stacker import MetaLearner
    from better.training.persistence import load_model

    console.print("[bold]Rebuilding OOF details from saved predictions...[/bold]")

    # Load training set and OOF predictions
    df = build_training_set()
    folds = generate_walk_forward_splits(df)
    oof_data = load_oof_predictions()

    # Reconstruct aligned arrays
    all_test_indices = []
    for fold in folds:
        all_test_indices.extend(fold.test_idx.tolist())

    oof_df = df.loc[all_test_indices].copy()

    # Build OOF output
    result = pd.DataFrame()
    for col in IDENTIFIER_COLS:
        if col in oof_df.columns:
            result[col] = oof_df[col].values
    result["home_win"] = oof_df[TARGET_COL].astype(int).values

    if "elo_home_win_prob" in oof_df.columns:
        result["elo_home_win_prob"] = oof_df["elo_home_win_prob"].values

    # Concatenate per-fold predictions for each model
    for model_name, fold_preds in sorted(oof_data.items()):
        all_preds = []
        for fold in folds:
            if fold.test_year in fold_preds:
                all_preds.append(fold_preds[fold.test_year])
        if all_preds:
            result[f"{model_name}_prob"] = np.concatenate(all_preds)

    # Meta-learner predictions from base model OOF
    base_preds = {}
    for model_name in ["gbm_ensemble", "bayesian_kalman", "monte_carlo"]:
        col = f"{model_name}_prob"
        if col in result.columns:
            base_preds[model_name] = result[col].values

    if base_preds:
        meta = MetaLearner()
        load_model(meta)
        meta_preds = meta.predict_from_base_predictions(base_preds, oof_df.reset_index(drop=True))
        result["meta_learner_prob"] = meta_preds

    results_dir = settings.project_root / "results"
    results_dir.mkdir(exist_ok=True)
    result.to_csv(results_dir / "oof_details.csv", index=False)
    console.print(f"[green]Saved {len(result)} rows to results/oof_details.csv[/green]")


# ── API commands ──────────────────────────────────────────────────────

api_app = typer.Typer(help="API server commands")
app.add_typer(api_app, name="api")


@api_app.command("serve")
def api_serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Auto-reload on file changes"),
) -> None:
    """Start the FastAPI prediction server."""
    import uvicorn

    console.print(f"\n[bold green]Starting BETter API server[/bold green]")
    console.print(f"  Host: {host}:{port}")
    console.print(f"  Docs: http://{host}:{port}/docs\n")

    uvicorn.run(
        "better.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


# ── Dashboard command ──────────────────────────────────────────────────

@app.command("dashboard")
def dashboard_run(
    port: int = typer.Option(8501, help="Dashboard port"),
) -> None:
    """Launch the NiceGUI dashboard."""
    from better.dashboard.app import run

    console.print(f"\n[bold green]Starting BETter Dashboard[/bold green]")
    console.print(f"  URL: http://localhost:{port}\n")

    run(port=port)


if __name__ == "__main__":
    app()
