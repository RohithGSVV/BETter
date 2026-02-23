"""DuckDB connection manager."""

from __future__ import annotations

import duckdb

from better.config import settings


_connection: duckdb.DuckDBPyConnection | None = None


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Get or create the singleton DuckDB connection.

    For write operations (ingestion, feature computation), use read_only=False.
    For read operations (predictions, API), use read_only=True.
    """
    global _connection
    if _connection is None:
        settings.ensure_dirs()
        _connection = duckdb.connect(settings.db_path_str, read_only=read_only)
    return _connection


def execute(sql: str, params: list | None = None) -> duckdb.DuckDBPyRelation:
    """Execute SQL on the default connection."""
    conn = get_connection()
    if params:
        return conn.execute(sql, params)
    return conn.execute(sql)


def fetch_df(sql: str, params: list | None = None):
    """Execute SQL and return a pandas DataFrame."""
    result = execute(sql, params)
    return result.fetchdf()


def fetch_pl(sql: str, params: list | None = None):
    """Execute SQL and return a Polars DataFrame."""
    result = execute(sql, params)
    return result.pl()


def close() -> None:
    """Close the database connection."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
