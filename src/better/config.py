"""Application configuration via Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration loaded from environment variables and .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent.parent / "data"
    models_dir: Path = Path(__file__).resolve().parent.parent.parent / "models"
    raw_data_dir: Path = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

    # DuckDB
    duckdb_path: Path = Path(__file__).resolve().parent.parent.parent / "data" / "better.duckdb"

    # The Odds API
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # MLB Stats API
    mlb_api_base_url: str = "https://statsapi.mlb.com/api/v1"

    # Training
    train_start_year: int = 2010
    train_end_year: int = 2025
    statcast_start_year: int = 2015

    # Feature engineering
    rolling_windows: list[int] = [7, 14, 30]
    ewma_decay: float = 0.1
    pythagorean_exponent: float = 1.83

    # Model
    monte_carlo_sims: int = 10_000
    walk_forward_test_start: int = 2015
    min_edge_threshold: float = 0.03
    kelly_fraction: float = 0.25
    max_bet_pct: float = 0.05

    @property
    def db_path_str(self) -> str:
        return str(self.duckdb_path)

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        for d in [self.data_dir, self.raw_data_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
