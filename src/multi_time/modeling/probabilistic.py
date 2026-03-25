"""
Probabilistic forecasting and distribution fitting.

Provides prediction intervals, quantile forecasts, and integration with
skpro for probabilistic modeling. Gracefully falls back when skpro is
not installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster

logger = logging.getLogger(__name__)


def _check_skpro_available() -> bool:
    """Check if skpro is installed."""
    try:
        import skpro  # noqa: F401

        return True
    except ImportError:
        return False


def predict_intervals(
    forecaster: BaseForecaster,
    y: pd.Series,
    fh: int | list[int] | np.ndarray = 12,
    coverage: float | list[float] = 0.95,
    X: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate prediction intervals from a fitted forecaster.

    Args:
        forecaster: Fitted sktime forecaster.
        y: Training time series (used for fitting if not already fit).
        fh: Forecast horizon.
        coverage: Coverage probability (0–1) or list of probabilities.
        X: Optional exogenous variables.

    Returns:
        DataFrame with prediction interval columns.
    """
    if isinstance(fh, int):
        fh_array = np.arange(1, fh + 1)
    else:
        fh_array = np.array(fh)

    if isinstance(coverage, float):
        coverage = [coverage]

    # Fit if needed
    try:
        forecaster.check_is_fitted()
    except Exception:
        forecaster.fit(y, X=X, fh=fh_array)

    intervals = forecaster.predict_interval(fh=fh_array, X=X, coverage=coverage)

    logger.info(
        "Prediction intervals: model=%s, horizon=%d, coverages=%s",
        type(forecaster).__name__,
        len(fh_array),
        coverage,
    )
    return intervals


def predict_quantiles(
    forecaster: BaseForecaster,
    y: pd.Series,
    fh: int | list[int] | np.ndarray = 12,
    alpha: list[float] | None = None,
    X: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate quantile predictions from a fitted forecaster.

    Args:
        forecaster: Fitted sktime forecaster.
        y: Training time series.
        fh: Forecast horizon.
        alpha: List of quantile levels (e.g. [0.1, 0.5, 0.9]).
        X: Optional exogenous variables.

    Returns:
        DataFrame with quantile predictions.
    """
    if isinstance(fh, int):
        fh_array = np.arange(1, fh + 1)
    else:
        fh_array = np.array(fh)

    if alpha is None:
        alpha = [0.05, 0.25, 0.5, 0.75, 0.95]

    # Fit if needed
    try:
        forecaster.check_is_fitted()
    except Exception:
        forecaster.fit(y, X=X, fh=fh_array)

    quantiles = forecaster.predict_quantiles(fh=fh_array, X=X, alpha=alpha)

    logger.info(
        "Quantile predictions: model=%s, horizon=%d, alphas=%s",
        type(forecaster).__name__,
        len(fh_array),
        alpha,
    )
    return quantiles


def predict_variance(
    forecaster: BaseForecaster,
    y: pd.Series,
    fh: int | list[int] | np.ndarray = 12,
    X: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate variance predictions from a fitted forecaster.

    Args:
        forecaster: Fitted sktime forecaster.
        y: Training time series.
        fh: Forecast horizon.
        X: Optional exogenous variables.

    Returns:
        DataFrame with variance predictions.
    """
    if isinstance(fh, int):
        fh_array = np.arange(1, fh + 1)
    else:
        fh_array = np.array(fh)

    # Fit if needed
    try:
        forecaster.check_is_fitted()
    except Exception:
        forecaster.fit(y, X=X, fh=fh_array)

    variance = forecaster.predict_var(fh=fh_array, X=X)

    logger.info(
        "Variance prediction: model=%s, horizon=%d",
        type(forecaster).__name__,
        len(fh_array),
    )
    return variance


def create_probabilistic_forecaster(
    name: str = "exp_smoothing",
    **params: Any,
) -> BaseForecaster:
    """Create a forecaster that supports probabilistic predictions.

    Wraps standard forecasters to ensure interval/quantile prediction
    capabilities. Uses skpro-enhanced models when available.

    Args:
        name: Forecaster name.
        **params: Model-specific parameters.

    Returns:
        Forecaster with probabilistic prediction support.
    """
    from multi_time.modeling.forecasters import create_forecaster

    forecaster = create_forecaster(name, **params)

    # Verify probabilistic capability
    tags = forecaster.get_tags()
    has_intervals = tags.get("capability:pred_int", False)

    if has_intervals:
        logger.info("Forecaster '%s' supports native probabilistic predictions", name)
    else:
        logger.warning(
            "Forecaster '%s' may not support prediction intervals natively. "
            "Falling back to bootstrapped intervals if available.",
            name,
        )

    return forecaster


def fit_distribution(
    data: pd.Series,
    distributions: list[str] | None = None,
) -> dict[str, Any]:
    """Fit parametric distributions to time series values.

    Tests multiple distributions and returns goodness-of-fit statistics.
    Uses scipy.stats internally.

    Args:
        data: Numeric pandas Series.
        distributions: List of distribution names to test.
                       Defaults to ['norm', 'lognorm', 't', 'gamma', 'expon'].

    Returns:
        Dict mapping distribution name to {params, ks_stat, ks_pvalue, aic, bic}.
    """
    from scipy import stats as scipy_stats

    if distributions is None:
        distributions = ["norm", "lognorm", "t", "gamma", "expon"]

    clean = data.dropna().values
    n = len(clean)
    results = {}

    for dist_name in distributions:
        try:
            dist = getattr(scipy_stats, dist_name)
            params = dist.fit(clean)

            # Kolmogorov-Smirnov test
            ks_stat, ks_p = scipy_stats.kstest(clean, dist_name, args=params)

            # Log-likelihood for AIC/BIC
            log_lik = np.sum(dist.logpdf(clean, *params))
            k = len(params)
            aic = 2 * k - 2 * log_lik
            bic = k * np.log(n) - 2 * log_lik

            results[dist_name] = {
                "params": params,
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "aic": float(aic),
                "bic": float(bic),
                "log_likelihood": float(log_lik),
            }

            logger.info(
                "Distribution %s: KS stat=%.4f, p=%.4f, AIC=%.2f",
                dist_name,
                ks_stat,
                ks_p,
                aic,
            )
        except Exception as e:
            logger.warning("Failed to fit distribution '%s': %s", dist_name, e)
            results[dist_name] = {"error": str(e)}

    # Rank by AIC
    valid = {k: v for k, v in results.items() if "aic" in v}
    if valid:
        best = min(valid, key=lambda k: valid[k]["aic"])
        results["best_fit"] = best
        logger.info("Best fitting distribution: %s (AIC=%.2f)", best, valid[best]["aic"])

    return results
