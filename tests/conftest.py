"""Shared pytest fixtures for BETter tests."""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from better.data.schema import create_all_tables


@pytest.fixture
def db(tmp_path):
    """Create a temporary DuckDB database with all tables."""
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))

    # Monkey-patch the db module to use our test connection
    import better.data.db as db_module

    original = db_module._connection
    db_module._connection = conn

    # Create all tables
    create_all_tables()

    yield conn

    # Restore original connection
    db_module._connection = original
    conn.close()


@pytest.fixture
def sample_game():
    """A sample game dict for testing."""
    return {
        "game_pk": 717001,
        "game_date": date(2024, 7, 15),
        "season": 2024,
        "home_team": "NYY",
        "away_team": "BOS",
        "home_score": 5,
        "away_score": 3,
        "home_win": True,
    }
