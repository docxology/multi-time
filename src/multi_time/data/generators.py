"""
Sample data generators for testing and demonstration.

This module acts as a facade re-exporting all generators from their
dedicated sub-modules for backward compatibility.
"""

from __future__ import annotations

from multi_time.data.generators_core import (
    generate_daily_series,
    generate_hourly_series,
    generate_weekly_series,
    generate_monthly_series,
)
from multi_time.data.generators_specialty import (
    generate_patchy_series,
    generate_irregular_series,
    generate_random_walk,
    generate_multi_seasonal_series,
    generate_multivariate_series,
    generate_configurable_series,
)

# ── Registry ────────────────────────────────────────────────────────────────────

GENERATOR_REGISTRY = {
    "daily": generate_daily_series,
    "hourly": generate_hourly_series,
    "weekly": generate_weekly_series,
    "monthly": generate_monthly_series,
    "patchy": generate_patchy_series,
    "irregular": generate_irregular_series,
    "random_walk": generate_random_walk,
    "multi_seasonal": generate_multi_seasonal_series,
    "multivariate": generate_multivariate_series,
    "configurable": generate_configurable_series,
}


def list_generators() -> list[str]:
    """Return list of available generator names."""
    return sorted(GENERATOR_REGISTRY.keys())


__all__ = [
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
