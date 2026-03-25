"""
Statistical tests for time series analysis.

Facade module re-exporting from focused sub-modules for backward compatibility.
"""

from multi_time.stats.result import StatTestResult
from multi_time.stats.stationarity import test_stationarity
from multi_time.stats.normality import test_normality
from multi_time.stats.seasonality import test_seasonality, test_heteroscedasticity
from multi_time.stats.causality import test_granger_causality

__all__ = [
    "StatTestResult",
    "test_stationarity",
    "test_normality",
    "test_seasonality",
    "test_heteroscedasticity",
    "test_granger_causality",
]
