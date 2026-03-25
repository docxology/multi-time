"""
Time series transformers wrapping sktime's transformation classes.

Provides factory functions for common transformations and a pipeline builder
for chaining multiple transforms in sequence.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.lag import Lag
from sktime.transformations.compose import TransformerPipeline

logger = logging.getLogger(__name__)


def create_imputer(strategy: str = "drift", **kwargs: Any) -> Imputer:
    """Create a missing-value imputer.

    Args:
        strategy: Imputation strategy. One of 'drift', 'mean', 'median',
                  'ffill', 'bfill', 'nearest', 'constant'.
        **kwargs: Additional Imputer parameters.

    Returns:
        Configured sktime Imputer.
    """
    imputer = Imputer(method=strategy, **kwargs)
    logger.info("Created Imputer with strategy='%s'", strategy)
    return imputer


def create_detrender(**kwargs: Any) -> Detrender:
    """Create a trend removal transformer.

    Uses PolynomialTrendForecaster internally to estimate and remove trend.

    Args:
        **kwargs: Additional Detrender parameters.

    Returns:
        Configured sktime Detrender.
    """
    detrender = Detrender(**kwargs)
    logger.info("Created Detrender")
    return detrender


def create_deseasonalizer(sp: int = 12, model: str = "additive") -> Any:
    """Create a seasonal removal transformer.

    Args:
        sp: Seasonal period.
        model: 'additive' or 'multiplicative'.

    Returns:
        Configured sktime Deseasonalizer.
    """
    from sktime.transformations.series.detrend import Deseasonalizer

    deseasonalizer = Deseasonalizer(sp=sp, model=model)
    logger.info("Created Deseasonalizer: sp=%d, model=%s", sp, model)
    return deseasonalizer


def create_box_cox(**kwargs: Any) -> BoxCoxTransformer:
    """Create a Box-Cox power transformer.

    Args:
        **kwargs: Additional BoxCoxTransformer parameters.

    Returns:
        Configured BoxCoxTransformer.
    """
    transformer = BoxCoxTransformer(**kwargs)
    logger.info("Created BoxCoxTransformer")
    return transformer


def create_differencer(lags: int | list[int] = 1, **kwargs: Any) -> Differencer:
    """Create a differencing transformer.

    Args:
        lags: Differencing lag(s). Int for single, list for multiple.
        **kwargs: Additional Differencer parameters.

    Returns:
        Configured Differencer.
    """
    if isinstance(lags, int):
        lags = [lags]
    differencer = Differencer(lags=lags, **kwargs)
    logger.info("Created Differencer: lags=%s", lags)
    return differencer


def create_lag_transformer(lags: int | list[int] = 1, **kwargs: Any) -> Lag:
    """Create a lag feature transformer.

    Args:
        lags: Number of lag(s) to create.
        **kwargs: Additional Lag parameters.

    Returns:
        Configured Lag transformer.
    """
    lag_transformer = Lag(lags=lags, **kwargs)
    logger.info("Created Lag transformer: lags=%s", lags)
    return lag_transformer


# ── Transformer registry ────────────────────────────────────────────────────────

TRANSFORMER_REGISTRY: dict[str, callable] = {
    "impute": create_imputer,
    "detrend": create_detrender,
    "deseasonalize": create_deseasonalizer,
    "box_cox": create_box_cox,
    "difference": create_differencer,
    "lag": create_lag_transformer,
}


def build_transform_pipeline(
    steps: list[str | tuple[str, dict[str, Any]]],
) -> TransformerPipeline:
    """Build a transformation pipeline from a list of step specifications.

    Args:
        steps: List of transformer names (str) or tuples of (name, params_dict).
               Names must be keys in TRANSFORMER_REGISTRY.

    Returns:
        Configured TransformerPipeline.

    Raises:
        ValueError: If an unknown transformer name is specified.

    Example:
        >>> pipeline = build_transform_pipeline([
        ...     "impute",
        ...     ("deseasonalize", {"sp": 7}),
        ...     "detrend",
        ... ])
    """
    transformers = []
    for step in steps:
        if isinstance(step, str):
            name, params = step, {}
        else:
            name, params = step[0], step[1]

        if name not in TRANSFORMER_REGISTRY:
            raise ValueError(
                f"Unknown transformer '{name}'. Available: {list(TRANSFORMER_REGISTRY.keys())}"
            )

        transformer = TRANSFORMER_REGISTRY[name](**params)
        transformers.append(transformer)
        logger.info("Added transform step: %s(%s)", name, params)

    pipeline = TransformerPipeline(steps=transformers)
    logger.info("Built transform pipeline with %d steps", len(transformers))
    return pipeline


def apply_transform(
    transformer: Any,
    data: pd.Series | pd.DataFrame,
    fit: bool = True,
) -> pd.Series | pd.DataFrame:
    """Apply a transformer to data.

    Args:
        transformer: An sktime transformer instance.
        data: Input time series.
        fit: Whether to fit before transforming.

    Returns:
        Transformed data.
    """
    if fit:
        result = transformer.fit_transform(data)
    else:
        result = transformer.transform(data)

    logger.info(
        "Applied %s: input_shape=%s, output_shape=%s",
        type(transformer).__name__,
        getattr(data, "shape", "?"),
        getattr(result, "shape", "?"),
    )
    return result
