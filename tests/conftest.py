"""
Shared pytest fixtures for multi-time test suite.

Provides time series at various frequencies (daily, hourly, monthly),
with different characteristics (regular, irregular, patchy, trending, seasonal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_series() -> pd.Series:
    """Regular daily time series with trend and noise (100 points)."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    trend = np.linspace(10, 50, 100)
    noise = np.random.normal(0, 2, 100)
    return pd.Series(trend + noise, index=dates, name="daily")


@pytest.fixture
def hourly_series() -> pd.Series:
    """Regular hourly time series with daily seasonality (168 points = 1 week)."""
    np.random.seed(43)
    dates = pd.date_range("2023-06-01", periods=168, freq="h")
    hour_of_day = np.array([d.hour for d in dates])
    seasonal = 5 * np.sin(2 * np.pi * hour_of_day / 24)
    noise = np.random.normal(0, 0.5, 168)
    return pd.Series(50 + seasonal + noise, index=dates, name="hourly")


@pytest.fixture
def monthly_series() -> pd.Series:
    """Regular monthly series with annual seasonality (60 months = 5 years)."""
    np.random.seed(44)
    dates = pd.date_range("2019-01-01", periods=60, freq="MS")
    months = np.arange(60)
    trend = 0.5 * months
    seasonal = 10 * np.sin(2 * np.pi * months / 12)
    noise = np.random.normal(0, 1, 60)
    return pd.Series(100 + trend + seasonal + noise, index=dates, name="monthly")


@pytest.fixture
def patchy_series() -> pd.Series:
    """Daily series with deliberate gaps (missing values)."""
    np.random.seed(45)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    values = np.random.normal(50, 5, 100).astype(float)
    # Create gaps
    values[10:15] = np.nan  # 5-day gap
    values[40:42] = np.nan  # 2-day gap
    values[70:78] = np.nan  # 8-day gap
    return pd.Series(values, index=dates, name="patchy")


@pytest.fixture
def irregular_series() -> pd.Series:
    """Irregularly spaced time series."""
    np.random.seed(46)
    # Random dates within 2023
    base = pd.Timestamp("2023-01-01")
    offsets = sorted(np.random.choice(range(365), size=50, replace=False))
    dates = pd.DatetimeIndex([base + pd.Timedelta(days=int(d)) for d in offsets])
    values = np.random.normal(100, 10, 50)
    return pd.Series(values, index=dates, name="irregular")


@pytest.fixture
def stationary_series() -> pd.Series:
    """White noise (stationary) series."""
    np.random.seed(47)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    return pd.Series(np.random.normal(0, 1, 200), index=dates, name="stationary")


@pytest.fixture
def nonstationary_series() -> pd.Series:
    """Random walk (non-stationary) series."""
    np.random.seed(48)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    return pd.Series(np.cumsum(np.random.normal(0, 1, 200)), index=dates, name="random_walk")


@pytest.fixture
def short_series() -> pd.Series:
    """Very short series (10 points)."""
    np.random.seed(49)
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.Series(np.random.normal(50, 5, 10), index=dates, name="short")


@pytest.fixture
def multivariate_df() -> pd.DataFrame:
    """Multi-column DataFrame for Granger causality tests."""
    np.random.seed(50)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    x = np.random.normal(0, 1, n)
    # y is caused by x with lag 1
    y = np.zeros(n)
    y[0] = np.random.normal()
    for i in range(1, n):
        y[i] = 0.5 * x[i - 1] + np.random.normal(0, 0.5)
    return pd.DataFrame({"x": x, "y": y}, index=dates)
