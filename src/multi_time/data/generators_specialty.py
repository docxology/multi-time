"""
Specialty time series generators.

Generators for irregular, patchy, random walk, multi-seasonal,
multivariate, and fully configurable synthetic series.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_patchy_series(
    n: int = 100,
    start: str = "2023-01-01",
    gap_ranges: list[tuple[int, int]] | None = None,
    seed: int | None = 45,
) -> pd.Series:
    """Generate a daily series with deliberate gaps (NaN values).

    Args:
        n: Number of observations.
        start: Start date.
        gap_ranges: List of (start_idx, end_idx) tuples for NaN gaps.
                    Default creates 3 gaps at indices 10-15, 40-42, 70-78.
        seed: Random seed.

    Returns:
        Daily pd.Series with NaN gaps.
    """
    if seed is not None:
        np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    values = np.random.normal(50, 5, n).astype(float)

    if gap_ranges is None:
        gap_ranges = [(10, 15), (40, 42), (70, 78)]

    for start_idx, end_idx in gap_ranges:
        values[start_idx:end_idx] = np.nan

    total_missing = sum(end - start for start, end in gap_ranges)
    logger.info("Generated patchy series: n=%d, gaps=%d, missing=%d", n, len(gap_ranges), total_missing)
    return pd.Series(values, index=dates, name="patchy")


def generate_irregular_series(
    n: int = 50,
    year: int = 2023,
    seed: int | None = 46,
) -> pd.Series:
    """Generate an irregularly spaced time series.

    Args:
        n: Number of observations.
        year: Year for date range.
        seed: Random seed.

    Returns:
        pd.Series with irregular DatetimeIndex.
    """
    if seed is not None:
        np.random.seed(seed)
    base = pd.Timestamp(f"{year}-01-01")
    offsets = sorted(np.random.choice(range(365), size=n, replace=False))
    dates = pd.DatetimeIndex([base + pd.Timedelta(days=int(d)) for d in offsets])
    values = np.random.normal(100, 10, n)
    logger.info("Generated irregular series: n=%d, year=%d", n, year)
    return pd.Series(values, index=dates, name="irregular")


def generate_random_walk(
    n: int = 200,
    start: str = "2023-01-01",
    freq: str = "D",
    drift: float = 0.0,
    volatility: float = 1.0,
    initial_value: float = 100.0,
    seed: int | None = 47,
) -> pd.Series:
    """Generate a random walk (non-stationary) series.

    Args:
        n: Number of observations.
        start: Start date.
        freq: Frequency (D, H, W, MS, etc.).
        drift: Constant drift per step.
        volatility: Standard deviation of innovations.
        initial_value: Starting value.
        seed: Random seed.

    Returns:
        pd.Series with cumulative random walk.
    """
    if seed is not None:
        np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq=freq)
    innovations = np.random.normal(drift, volatility, n)
    values = initial_value + np.cumsum(innovations)
    logger.info("Generated random walk: n=%d, drift=%.2f, vol=%.2f", n, drift, volatility)
    return pd.Series(values, index=dates, name="random_walk")


def generate_multi_seasonal_series(
    n: int = 720,
    start: str = "2023-01-01",
    freq: str = "h",
    periods: list[int] | None = None,
    amplitudes: list[float] | None = None,
    trend: float = 0.01,
    noise_std: float = 1.0,
    seed: int | None = 51,
) -> pd.Series:
    """Generate a series with multiple seasonal components.

    Args:
        n: Number of observations.
        start: Start datetime.
        freq: Frequency string.
        periods: List of seasonal period lengths (default: [24, 168] for hourly data).
        amplitudes: Corresponding amplitudes (default: [5.0, 10.0]).
        trend: Linear trend per step.
        noise_std: Noise standard deviation.
        seed: Random seed.

    Returns:
        pd.Series with multiple overlapping seasonal patterns.
    """
    if seed is not None:
        np.random.seed(seed)
    if periods is None:
        periods = [24, 168]  # daily + weekly cycles
    if amplitudes is None:
        amplitudes = [5.0, 10.0]

    dates = pd.date_range(start, periods=n, freq=freq)
    t = np.arange(n)
    values = 100 + trend * t + np.random.normal(0, noise_std, n)
    for period, amplitude in zip(periods, amplitudes):
        values += amplitude * np.sin(2 * np.pi * t / period)

    logger.info("Generated multi-seasonal series: n=%d, periods=%s", n, periods)
    return pd.Series(values, index=dates, name="multi_seasonal")


def generate_multivariate_series(
    n: int = 200,
    start: str = "2023-01-01",
    freq: str = "D",
    n_series: int = 3,
    correlation: float = 0.6,
    seed: int | None = 52,
) -> pd.DataFrame:
    """Generate correlated multivariate time series.

    Args:
        n: Number of observations.
        start: Start date.
        freq: Frequency.
        n_series: Number of series to generate.
        correlation: Pairwise correlation between series (0 to 1).
        seed: Random seed.

    Returns:
        pd.DataFrame with correlated time series columns.
    """
    if seed is not None:
        np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq=freq)

    # Build correlation matrix
    cov = np.full((n_series, n_series), correlation)
    np.fill_diagonal(cov, 1.0)
    data = np.random.multivariate_normal(np.zeros(n_series), cov, size=n)

    # Add trends
    for i in range(n_series):
        data[:, i] += np.linspace(50 + i * 10, 60 + i * 10, n)

    columns = [f"series_{i + 1}" for i in range(n_series)]
    logger.info("Generated multivariate: n=%d, n_series=%d, corr=%.2f", n, n_series, correlation)
    return pd.DataFrame(data, index=dates, columns=columns)


def generate_configurable_series(
    n: int = 365,
    start: str = "2023-01-01",
    freq: str = "D",
    baseline: float = 100.0,
    trend: float = 0.1,
    seasonal_period: int | None = None,
    seasonal_amplitude: float = 0.0,
    noise_std: float = 1.0,
    outlier_fraction: float = 0.0,
    outlier_magnitude: float = 5.0,
    gap_fraction: float = 0.0,
    seed: int | None = 53,
) -> pd.Series:
    """Generate a fully configurable synthetic time series.

    Combines any mix of trend, seasonality, noise, outliers, and missing data.

    Args:
        n: Number of observations.
        start: Start date.
        freq: Frequency string.
        baseline: Starting level.
        trend: Linear trend per step.
        seasonal_period: Period for seasonal component (None = no seasonality).
        seasonal_amplitude: Amplitude of seasonal component.
        noise_std: Gaussian noise standard deviation.
        outlier_fraction: Fraction of points to make outliers (0–1).
        outlier_magnitude: Std deviations for outliers.
        gap_fraction: Fraction of points to set as NaN (0–1).
        seed: Random seed.

    Returns:
        pd.Series with configurable properties.
    """
    if seed is not None:
        np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq=freq)
    t = np.arange(n, dtype=float)

    # Base + trend
    values = baseline + trend * t

    # Seasonality
    if seasonal_period is not None and seasonal_amplitude > 0:
        values += seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)

    # Noise
    values += np.random.normal(0, noise_std, n)

    # Outliers
    if outlier_fraction > 0:
        n_outliers = max(1, int(n * outlier_fraction))
        outlier_idx = np.random.choice(n, n_outliers, replace=False)
        values[outlier_idx] += np.random.choice([-1, 1], n_outliers) * outlier_magnitude * noise_std

    # Gaps
    if gap_fraction > 0:
        n_gaps = max(1, int(n * gap_fraction))
        gap_idx = np.random.choice(n, n_gaps, replace=False)
        values[gap_idx] = np.nan

    logger.info(
        "Generated configurable series: n=%d, freq=%s, trend=%.2f, "
        "seasonal_period=%s, outliers=%.1f%%, gaps=%.1f%%",
        n, freq, trend, seasonal_period,
        outlier_fraction * 100, gap_fraction * 100,
    )
    return pd.Series(values, index=dates, name="configurable")
