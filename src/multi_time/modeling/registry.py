"""
Forecaster registry and factory.
"""

from __future__ import annotations

import logging
from typing import Any

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster

logger = logging.getLogger(__name__)


# ── Factory functions ─────────────────────────────────────────────────────────


def _create_naive(**kwargs: Any) -> NaiveForecaster:
    return NaiveForecaster(strategy=kwargs.get("strategy", "last"), sp=kwargs.get("sp", 1))


def _create_exp_smoothing(**kwargs: Any) -> ExponentialSmoothing:
    return ExponentialSmoothing(
        trend=kwargs.get("trend", "add"),
        seasonal=kwargs.get("seasonal", "add"),
        sp=kwargs.get("sp", 12),
    )


def _create_theta(**kwargs: Any) -> ThetaForecaster:
    return ThetaForecaster(sp=kwargs.get("sp", 1))


def _create_poly_trend(**kwargs: Any) -> PolynomialTrendForecaster:
    return PolynomialTrendForecaster(degree=kwargs.get("degree", 1))


def _create_auto_arima(**kwargs: Any) -> BaseForecaster:
    """Create AutoARIMA (requires pmdarima)."""
    try:
        from sktime.forecasting.arima import AutoARIMA
        return AutoARIMA(
            sp=kwargs.get("sp", 1),
            suppress_warnings=kwargs.get("suppress_warnings", True),
            max_p=kwargs.get("max_p", 5),
            max_q=kwargs.get("max_q", 5),
        )
    except ImportError:
        logger.warning("pmdarima not installed; falling back to ExponentialSmoothing")
        return _create_exp_smoothing(**kwargs)


def _create_sarimax(**kwargs: Any) -> BaseForecaster:
    """Create SARIMAX (requires statsmodels)."""
    from sktime.forecasting.sarimax import SARIMAX
    return SARIMAX(
        order=kwargs.get("order", (1, 0, 0)),
        seasonal_order=kwargs.get("seasonal_order", (0, 0, 0, 0)),
        enforce_stationarity=kwargs.get("enforce_stationarity", False),
        enforce_invertibility=kwargs.get("enforce_invertibility", False),
    )


FORECASTER_REGISTRY: dict[str, callable] = {
    "naive": _create_naive,
    "exp_smoothing": _create_exp_smoothing,
    "theta": _create_theta,
    "poly_trend": _create_poly_trend,
    "auto_arima": _create_auto_arima,
    "sarimax": _create_sarimax,
}


def create_forecaster(name: str, **params: Any) -> BaseForecaster:
    """Create a forecaster by name with given parameters.

    Args:
        name: Forecaster name (key in FORECASTER_REGISTRY).
        **params: Model-specific parameters.

    Returns:
        Configured sktime forecaster.

    Raises:
        ValueError: If name not in registry.
    """
    if name not in FORECASTER_REGISTRY:
        raise ValueError(
            f"Unknown forecaster '{name}'. Available: {list(FORECASTER_REGISTRY.keys())}"
        )

    forecaster = FORECASTER_REGISTRY[name](**params)
    logger.info("Created forecaster: %s(%s)", name, params)
    return forecaster
