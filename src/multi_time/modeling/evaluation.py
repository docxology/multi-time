"""
Model evaluation and hyperparameter tuning.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.split import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)
from sktime.performance_metrics.forecasting import MeanAbsoluteError

logger = logging.getLogger(__name__)


def evaluate_forecaster(
    forecaster: BaseForecaster,
    y: pd.Series,
    cv_strategy: str = "expanding",
    initial_window: int | None = None,
    step_length: int = 1,
    fh: int | list[int] = 1,
) -> pd.DataFrame:
    """Evaluate a forecaster using temporal cross-validation.

    Args:
        forecaster: Configured sktime forecaster.
        y: Full time series.
        cv_strategy: 'expanding' or 'sliding'.
        initial_window: Initial training window size. Defaults to 70% of data.
        step_length: Step between successive windows.
        fh: Forecast horizon.

    Returns:
        DataFrame with cross-validation results.
    """
    if initial_window is None:
        initial_window = max(10, int(len(y) * 0.7))

    if isinstance(fh, int):
        fh_array = np.arange(1, fh + 1)
    else:
        fh_array = np.array(fh)

    if cv_strategy == "sliding":
        cv = SlidingWindowSplitter(
            window_length=initial_window,
            step_length=step_length,
            fh=fh_array,
        )
    else:
        cv = ExpandingWindowSplitter(
            initial_window=initial_window,
            step_length=step_length,
            fh=fh_array,
        )

    scoring = MeanAbsoluteError()
    results = evaluate(
        forecaster=forecaster,
        y=y,
        cv=cv,
        scoring=scoring,
        return_data=True,
    )

    logger.info(
        "Evaluation complete: model=%s, cv=%s, folds=%d",
        type(forecaster).__name__,
        cv_strategy,
        len(results),
    )
    return results


def tune_forecaster(
    forecaster: BaseForecaster,
    param_grid: dict[str, list[Any]],
    y: pd.Series,
    cv_strategy: str = "expanding",
    initial_window: int | None = None,
    fh: int | list[int] = 1,
) -> ForecastingGridSearchCV:
    """Tune forecaster hyperparameters via grid search.

    Args:
        forecaster: Base forecaster to tune.
        param_grid: Dictionary of parameter names to candidate values.
        y: Training time series.
        cv_strategy: 'expanding' or 'sliding'.
        initial_window: Initial window size for CV.
        fh: Forecast horizon.

    Returns:
        Fitted ForecastingGridSearchCV with best parameters.
    """
    if initial_window is None:
        initial_window = max(10, int(len(y) * 0.7))

    if isinstance(fh, int):
        fh_array = np.arange(1, fh + 1)
    else:
        fh_array = np.array(fh)

    cv = ExpandingWindowSplitter(initial_window=initial_window, fh=fh_array)
    scoring = MeanAbsoluteError()

    gscv = ForecastingGridSearchCV(
        forecaster=forecaster,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
    )

    gscv.fit(y)

    logger.info(
        "Tuning complete: best_params=%s, best_score=%.4f",
        gscv.best_params_,
        gscv.best_score_,
    )
    return gscv
