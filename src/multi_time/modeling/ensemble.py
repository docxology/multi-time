"""
Ensemble forecaster builder.
"""

from __future__ import annotations

import logging
from typing import Any

from sktime.forecasting.compose import EnsembleForecaster

from multi_time.modeling.registry import create_forecaster

logger = logging.getLogger(__name__)


def create_ensemble(
    forecaster_specs: list[str | tuple[str, dict[str, Any]]],
    aggfunc: str = "mean",
) -> EnsembleForecaster:
    """Create an ensemble of forecasters.

    Args:
        forecaster_specs: List of forecaster names or (name, params) tuples.
        aggfunc: Aggregation function ('mean', 'median', 'min', 'max').

    Returns:
        Configured EnsembleForecaster.
    """
    forecasters = []
    for spec in forecaster_specs:
        if isinstance(spec, str):
            name, params = spec, {}
        else:
            name, params = spec[0], spec[1]
        f = create_forecaster(name, **params)
        forecasters.append((name, f))

    ensemble = EnsembleForecaster(forecasters=forecasters, aggfunc=aggfunc)
    logger.info("Created ensemble with %d forecasters, aggfunc=%s", len(forecasters), aggfunc)
    return ensemble
