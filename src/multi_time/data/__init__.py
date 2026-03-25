"""
multi_time.data — Data loading, I/O, and sample datasets.

Provides utilities for loading time series from CSV files and generating
sample datasets for testing and demonstration.
"""

from multi_time.data.loaders import load_csv_series, load_csv_dataframe
from multi_time.data.generators import (
    generate_daily_series,
    generate_hourly_series,
    generate_weekly_series,
    generate_monthly_series,
    generate_patchy_series,
    generate_irregular_series,
    generate_random_walk,
    generate_multi_seasonal_series,
    generate_multivariate_series,
    generate_configurable_series,
    list_generators,
    GENERATOR_REGISTRY,
)

__all__ = [
    "load_csv_series",
    "load_csv_dataframe",
    "generate_daily_series",
    "generate_hourly_series",
    "generate_weekly_series",
    "generate_monthly_series",
    "generate_patchy_series",
    "generate_irregular_series",
    "generate_random_walk",
    "generate_multi_seasonal_series",
    "generate_multivariate_series",
    "generate_configurable_series",
    "list_generators",
    "GENERATOR_REGISTRY",
]
