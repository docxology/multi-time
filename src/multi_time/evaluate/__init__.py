"""
multi_time.evaluate — Forecast evaluation metrics.

Provides a unified interface for computing single and multiple metrics
on forecast results using sktime performance metrics.
"""

from multi_time.evaluate.metrics import (
    compute_metric,
    evaluate_forecast,
    compute_rmse,
    list_available_metrics,
    METRIC_REGISTRY,
)

__all__ = [
    "compute_metric",
    "evaluate_forecast",
    "compute_rmse",
    "list_available_metrics",
    "METRIC_REGISTRY",
]
