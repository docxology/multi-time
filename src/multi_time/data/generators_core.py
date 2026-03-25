"""
Core time series generators: daily, hourly, weekly, monthly.

Regular-frequency generators with trend, seasonality, and noise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_daily_series(
    n: int = 365,
    start: str = "2023-01-01",
    trend: float = 0.1,
    noise_std: float = 2.0,
    seed: int | None = 42,
) -> pd.Series:
    """Generate a daily time series with trend and noise.

    Args:
        n: Number of observations.
        start: Start date string.
        trend: Linear trend slope per period.
        noise_std: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        Daily pd.Series with DatetimeIndex.
    """
    if seed is not None:
        np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    values = np.linspace(10, 10 + trend * n, n) + np.random.normal(0, noise_std, n)
    logger.info("Generated daily series: n=%d, start=%s", n, start)
    return pd.Series(values, index=dates, name="daily")


def generate_hourly_series(
    n: int = 168,
    start: str = "2023-06-01",
    daily_amplitude: float = 5.0,
    baseline: float = 50.0,
    noise_std: float = 0.5,
    seed: int | None = 43,
) -> pd.Series:
    """Generate an hourly time series with daily seasonality.

    Args:
        n: Number of hours (168 = 1 week).
        start: Start datetime string.
        daily_amplitude: Amplitude of daily cycle.
        baseline: Baseline level.
        noise_std: Noise standard deviation.
        seed: Random seed.

    Returns:
        Hourly pd.Series with DatetimeIndex.
    """
    if seed is not None:
        np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq="h")
    hour_of_day = np.array([d.hour for d in dates])
    seasonal = daily_amplitude * np.sin(2 * np.pi * hour_of_day / 24)
    values = baseline + seasonal + np.random.normal(0, noise_std, n)
    logger.info("Generated hourly series: n=%d, start=%s", n, start)
    return pd.Series(values, index=dates, name="hourly")


def generate_weekly_series(
    n: int = 104,
    start: str = "2021-01-04",
    trend: float = 0.3,
    annual_amplitude: float = 8.0,
    noise_std: float = 1.5,
    seed: int | None = 50,
) -> pd.Series:
    """Generate a weekly time series with annual seasonality.

    Args:
        n: Number of weeks (104 = 2 years).
        start: Start date (should be a Monday).
        trend: Linear trend per week.
        annual_amplitude: Amplitude of 52-week seasonal cycle.
        noise_std: Noise standard deviation.
        seed: Random seed.

    Returns:
        Weekly pd.Series with DatetimeIndex.
    """
    if seed is not None:
        np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq="W-MON")
    weeks = np.arange(n)
    values = (
        100 + trend * weeks
        + annual_amplitude * np.sin(2 * np.pi * weeks / 52)
        + np.random.normal(0, noise_std, n)
    )
    logger.info("Generated weekly series: n=%d, start=%s", n, start)
    return pd.Series(values, index=dates, name="weekly")


def generate_monthly_series(
    n: int = 60,
    start: str = "2019-01-01",
    seasonal_amplitude: float = 10.0,
    trend: float = 0.5,
    noise_std: float = 1.0,
    seed: int | None = 44,
) -> pd.Series:
    """Generate a monthly series with annual seasonality.

    Args:
        n: Number of months.
        start: Start date string.
        seasonal_amplitude: Amplitude of seasonal component.
        trend: Linear trend per month.
        noise_std: Noise standard deviation.
        seed: Random seed.

    Returns:
        Monthly pd.Series with DatetimeIndex.
    """
    if seed is not None:
        np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq="MS")
    months = np.arange(n)
    values = (
        100 + trend * months
        + seasonal_amplitude * np.sin(2 * np.pi * months / 12)
        + np.random.normal(0, noise_std, n)
    )
    logger.info("Generated monthly series: n=%d, start=%s", n, start)
    return pd.Series(values, index=dates, name="monthly")
