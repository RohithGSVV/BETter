"""BETter CLI — powered by Typer.

Usage::

    uv run better features build
    uv run better features training-set
    uv run better features elo
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


if __name__ == "__main__":
    app()
