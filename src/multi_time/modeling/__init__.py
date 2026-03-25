"""
multi_time.modeling — Forecasting and probabilistic modeling.

Provides forecaster factories, ensemble methods, temporal cross-validation,
hyperparameter tuning, and probabilistic prediction interfaces.
"""

from multi_time.modeling.forecasters import (
    create_forecaster,
    create_ensemble,
    run_forecast,
    evaluate_forecaster,
    tune_forecaster,
    FORECASTER_REGISTRY,
)
from multi_time.modeling.probabilistic import (
    predict_intervals,
    predict_quantiles,
    predict_variance,
    create_probabilistic_forecaster,
    fit_distribution,
)

__all__ = [
    "create_forecaster",
    "create_ensemble",
    "run_forecast",
    "evaluate_forecaster",
    "tune_forecaster",
    "FORECASTER_REGISTRY",
    "predict_intervals",
    "predict_quantiles",
    "predict_variance",
    "create_probabilistic_forecaster",
    "fit_distribution",
]
