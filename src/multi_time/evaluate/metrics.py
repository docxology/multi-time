"""
Forecast evaluation metrics wrapping sktime performance metrics.

Provides a unified interface for computing single and multiple metrics
on forecast results.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
    MedianAbsolutePercentageError,
)

logger = logging.getLogger(__name__)


# ── Metric registry ─────────────────────────────────────────────────────────────

METRIC_REGISTRY: dict[str, type] = {
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "mape": MeanAbsolutePercentageError,
    "mdape": MedianAbsolutePercentageError,
}

# Try importing optional metrics
try:
    from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

    METRIC_REGISTRY["mase"] = MeanAbsoluteScaledError
except ImportError:
    pass

try:
    from sktime.performance_metrics.forecasting import MeanSquaredScaledError

    METRIC_REGISTRY["rmsse"] = MeanSquaredScaledError
except ImportError:
    pass


def compute_metric(
    y_true: pd.Series,
    y_pred: pd.Series,
    metric_name: str,
    y_train: pd.Series | None = None,
    **kwargs: Any,
) -> float:
    """Compute a single forecast metric.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        metric_name: Metric name (key in METRIC_REGISTRY).
        y_train: Training data (required for scaled metrics like MASE).
        **kwargs: Additional metric parameters.

    Returns:
        Computed metric value.

    Raises:
        ValueError: If metric_name not in registry.
    """
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{metric_name}'. Available: {list(METRIC_REGISTRY.keys())}"
        )

    metric_class = METRIC_REGISTRY[metric_name]
    metric = metric_class(**kwargs)

    # Scaled metrics require y_train
    if metric_name in ("mase", "rmsse"):
        if y_train is None:
            raise ValueError(f"Metric '{metric_name}' requires y_train")
        value = metric(y_true, y_pred, y_train=y_train)
    else:
        value = metric(y_true, y_pred)

    logger.info("Metric %s = %.6f", metric_name, value)
    return float(value)


def evaluate_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    metrics_list: list[str] | None = None,
    y_train: pd.Series | None = None,
) -> dict[str, float]:
    """Compute multiple forecast evaluation metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        metrics_list: List of metric names. Defaults to ['mae', 'mse', 'mape'].
        y_train: Training data (needed for scaled metrics).

    Returns:
        Dict mapping metric name to computed value.
    """
    if metrics_list is None:
        metrics_list = ["mae", "mse", "mape"]

    results: dict[str, float] = {}
    for name in metrics_list:
        try:
            results[name] = compute_metric(
                y_true, y_pred, name, y_train=y_train
            )
        except Exception as e:
            logger.warning("Failed to compute metric '%s': %s", name, e)
            results[name] = float("nan")

    logger.info("Evaluation results: %s", {k: f"{v:.4f}" for k, v in results.items()})
    return results


def compute_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        RMSE value.
    """
    mse = MeanSquaredError(square_root=True)
    value = float(mse(y_true, y_pred))
    logger.info("RMSE = %.6f", value)
    return value


def list_available_metrics() -> list[str]:
    """List all available metric names.

    Returns:
        Sorted list of metric name strings.
    """
    return sorted(METRIC_REGISTRY.keys())
