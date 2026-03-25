"""
Forecasting interface wrapping sktime forecasters.

Facade module re-exporting from focused sub-modules for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster

from multi_time.modeling.registry import (
    FORECASTER_REGISTRY,
    create_forecaster,
)
from multi_time.modeling.ensemble import create_ensemble
from multi_time.modeling.evaluation import evaluate_forecaster, tune_forecaster

logger = logging.getLogger(__name__)


def run_forecast(
    forecaster: BaseForecaster,
    y_train: pd.Series,
    fh: int | list[int] | np.ndarray = 12,
    X: pd.DataFrame | None = None,
) -> pd.Series:
    """Fit a forecaster and generate predictions.

    Args:
        forecaster: Configured sktime forecaster.
        y_train: Training time series.
        fh: Forecast horizon (int for range, list for specific steps).
        X: Optional exogenous variables.

    Returns:
        Predicted values as pandas Series.
    """
    if isinstance(fh, int):
        fh_array = np.arange(1, fh + 1)
    else:
        fh_array = np.array(fh)

    forecaster.fit(y_train, X=X, fh=fh_array)
    predictions = forecaster.predict(fh=fh_array, X=X)

    logger.info(
        "Forecast complete: model=%s, horizon=%d, predictions=%d",
        type(forecaster).__name__,
        len(fh_array),
        len(predictions),
    )
    return predictions


__all__ = [
    "FORECASTER_REGISTRY",
    "create_forecaster",
    "create_ensemble",
    "run_forecast",
    "evaluate_forecaster",
    "tune_forecaster",
]
