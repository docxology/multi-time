"""
multi_time.stats — Descriptive and inferential statistics.

Provides descriptive statistics, ACF/PACF, seasonal decomposition,
and statistical tests (stationarity, normality, seasonality, causality).
"""

from multi_time.stats.descriptive import (
    compute_descriptive_stats,
    compute_acf_pacf,
    compute_rolling_stats,
    compute_seasonal_decomposition,
    summarize_series,
)
from multi_time.stats.tests import (
    test_stationarity,
    test_normality,
    test_seasonality,
    test_heteroscedasticity,
    test_granger_causality,
    StatTestResult,
)

__all__ = [
    "compute_descriptive_stats",
    "compute_acf_pacf",
    "compute_rolling_stats",
    "compute_seasonal_decomposition",
    "summarize_series",
    "test_stationarity",
    "test_normality",
    "test_seasonality",
    "test_heteroscedasticity",
    "test_granger_causality",
    "StatTestResult",
]
