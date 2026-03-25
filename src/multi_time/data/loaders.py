"""
Data loading utilities for multi-time.

Provides CSV readers with automatic date parsing, frequency inference,
and column selection for time series workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv_series(
    path: str | Path,
    column: str | None = None,
    date_column: str | int | None = 0,
    freq: str | None = None,
) -> pd.Series:
    """Load a single time series from a CSV file.

    Args:
        path: Path to CSV file.
        column: Column name to extract. Uses first numeric column if None.
        date_column: Column to use as index (name or position). Default 0.
        freq: Force a specific frequency after loading.

    Returns:
        Numeric pandas Series with DatetimeIndex.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If no numeric columns found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, parse_dates=True, index_col=date_column)

    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
        series = df[column]
    else:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in input file")
        series = df[numeric_cols[0]]

    if freq:
        series = series.asfreq(freq)

    logger.info("Loaded series: %d observations from %s (column=%s)", len(series), path, column or series.name)
    return series


def load_csv_dataframe(
    path: str | Path,
    date_column: str | int | None = 0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a multivariate time series from a CSV file.

    Args:
        path: Path to CSV file.
        date_column: Column to use as index.
        columns: Subset of columns to load. All if None.

    Returns:
        DataFrame with DatetimeIndex.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, parse_dates=True, index_col=date_column)

    if columns:
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        df = df[columns]

    logger.info("Loaded DataFrame: %s from %s", df.shape, path)
    return df
