"""
Descriptive statistics for time series.

Computes comprehensive summary statistics, autocorrelation functions,
rolling statistics, and feature extraction using sktime transformers.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import acf, pacf

logger = logging.getLogger(__name__)


def compute_descriptive_stats(data: pd.Series) -> dict[str, Any]:
    """Compute comprehensive descriptive statistics for a time series.

    Args:
        data: Numeric pandas Series.

    Returns:
        Dict with keys: count, mean, std, var, min, max, median,
        skewness, kurtosis, q25, q50, q75, iqr, range, cv.
    """
    clean = data.dropna()

    if len(clean) == 0:
        logger.warning("No non-null values for descriptive stats")
        return {"count": 0, "error": "No non-null values"}

    mean_val = float(clean.mean())
    std_val = float(clean.std())

    result = {
        "count": int(len(clean)),
        "mean": mean_val,
        "std": std_val,
        "var": float(clean.var()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "median": float(clean.median()),
        "skewness": float(scipy_stats.skew(clean, nan_policy="omit")),
        "kurtosis": float(scipy_stats.kurtosis(clean, nan_policy="omit")),
        "q25": float(clean.quantile(0.25)),
        "q50": float(clean.quantile(0.50)),
        "q75": float(clean.quantile(0.75)),
        "iqr": float(clean.quantile(0.75) - clean.quantile(0.25)),
        "range": float(clean.max() - clean.min()),
        "cv": float(std_val / abs(mean_val)) if mean_val != 0 else float("inf"),
    }

    logger.info(
        "Descriptive stats: n=%d, mean=%.4f, std=%.4f, skew=%.4f",
        result["count"],
        result["mean"],
        result["std"],
        result["skewness"],
    )
    return result


def compute_acf_pacf(
    data: pd.Series,
    nlags: int = 40,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute autocorrelation and partial autocorrelation functions.

    Args:
        data: Numeric pandas Series (NaN values are dropped).
        nlags: Number of lags to compute.
        alpha: Significance level for confidence intervals.

    Returns:
        Dict with keys: acf_values, pacf_values, acf_ci, pacf_ci,
        significant_acf_lags, significant_pacf_lags.
    """
    clean = data.dropna().values

    if len(clean) < nlags + 1:
        nlags = max(1, len(clean) // 2 - 1)
        logger.warning("Adjusted nlags to %d due to short series", nlags)

    acf_values, acf_ci = acf(clean, nlags=nlags, alpha=alpha)
    pacf_values, pacf_ci = pacf(clean, nlags=nlags, alpha=alpha)

    # Identify significant lags (outside confidence interval)
    ci_width = 1.96 / np.sqrt(len(clean))
    sig_acf = [int(i) for i in range(1, len(acf_values)) if abs(acf_values[i]) > ci_width]
    sig_pacf = [int(i) for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > ci_width]

    result = {
        "acf_values": acf_values.tolist(),
        "pacf_values": pacf_values.tolist(),
        "acf_ci": acf_ci.tolist(),
        "pacf_ci": pacf_ci.tolist(),
        "significant_acf_lags": sig_acf,
        "significant_pacf_lags": sig_pacf,
        "nlags": nlags,
    }

    logger.info(
        "ACF/PACF: nlags=%d, significant_acf=%d, significant_pacf=%d",
        nlags,
        len(sig_acf),
        len(sig_pacf),
    )
    return result


def compute_rolling_stats(
    data: pd.Series,
    window: int = 12,
) -> pd.DataFrame:
    """Compute rolling statistics over a sliding window.

    Args:
        data: Numeric pandas Series.
        window: Rolling window size.

    Returns:
        DataFrame with columns: rolling_mean, rolling_std, rolling_min,
        rolling_max, rolling_median.
    """
    result = pd.DataFrame(index=data.index)
    result["rolling_mean"] = data.rolling(window=window, min_periods=1).mean()
    result["rolling_std"] = data.rolling(window=window, min_periods=1).std()
    result["rolling_min"] = data.rolling(window=window, min_periods=1).min()
    result["rolling_max"] = data.rolling(window=window, min_periods=1).max()
    result["rolling_median"] = data.rolling(window=window, min_periods=1).median()

    logger.info("Rolling stats computed: window=%d, shape=%s", window, result.shape)
    return result


def compute_seasonal_decomposition(
    data: pd.Series,
    period: int | None = None,
    model: str = "additive",
) -> dict[str, pd.Series]:
    """Decompose a time series into trend, seasonal, and residual components.

    Args:
        data: Numeric pandas Series with regular frequency.
        period: Seasonal period (auto-detected if None).
        model: 'additive' or 'multiplicative'.

    Returns:
        Dict with keys: trend, seasonal, residual, observed.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    clean = data.dropna()

    if period is None:
        # Heuristic period detection
        freq = pd.infer_freq(clean.index)
        period_map = {"H": 24, "D": 7, "B": 5, "W": 52, "MS": 12, "M": 12, "QS": 4, "Q": 4}
        period = period_map.get(freq, 12) if freq else 12
        logger.info("Auto-detected seasonal period: %d", period)

    if len(clean) < 2 * period:
        logger.warning("Series too short for period=%d, using period=2", period)
        period = 2

    decomposition = seasonal_decompose(clean, model=model, period=period)

    result = {
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid,
        "observed": decomposition.observed,
    }

    logger.info(
        "Seasonal decomposition: model=%s, period=%d",
        model,
        period,
    )
    return result


def summarize_series(
    data: pd.Series,
    nlags: int = 40,
    rolling_window: int = 12,
) -> dict[str, Any]:
    """Produce a full summary of a time series.

    Combines descriptive stats, ACF/PACF, and rolling stats into one report.

    Args:
        data: Numeric pandas Series.
        nlags: Lags for ACF/PACF.
        rolling_window: Window for rolling stats.

    Returns:
        Dict with keys: descriptive, acf_pacf, rolling (as dict summary).
    """
    summary: dict[str, Any] = {}

    summary["descriptive"] = compute_descriptive_stats(data)
    summary["acf_pacf"] = compute_acf_pacf(data, nlags=min(nlags, max(1, len(data.dropna()) // 2 - 1)))

    rolling = compute_rolling_stats(data, window=rolling_window)
    summary["rolling"] = {
        "window": rolling_window,
        "final_rolling_mean": float(rolling["rolling_mean"].iloc[-1])
        if not rolling["rolling_mean"].empty
        else None,
        "final_rolling_std": float(rolling["rolling_std"].iloc[-1])
        if not rolling["rolling_std"].empty
        else None,
    }

    logger.info("Full series summary computed")
    return summary
