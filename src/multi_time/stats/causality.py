"""
Granger causality test.
"""

from __future__ import annotations

import logging

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from multi_time.stats.result import StatTestResult

logger = logging.getLogger(__name__)


def test_granger_causality(
    data_x: pd.Series,
    data_y: pd.Series,
    maxlag: int = 4,
    alpha: float = 0.05,
) -> StatTestResult:
    """Test Granger causality from x to y.

    Tests whether x Granger-causes y (past values of x help predict y).

    Args:
        data_x: Potential cause series.
        data_y: Potential effect series.
        maxlag: Maximum number of lags to test.
        alpha: Significance level.

    Returns:
        StatTestResult with minimum p-value across lags.
    """
    # Align series
    combined = pd.DataFrame({"y": data_y, "x": data_x}).dropna()

    if len(combined) < maxlag + 2:
        return StatTestResult(
            test_name="Granger Causality",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            alpha=alpha,
            interpretation="Insufficient data for Granger test",
        )

    gc_results = grangercausalitytests(combined[["y", "x"]], maxlag=maxlag)

    # Find minimum p-value across all lags (using F-test)
    min_p = 1.0
    best_lag = 1
    lag_results = {}
    for lag in range(1, maxlag + 1):
        f_stat = gc_results[lag][0]["ssr_ftest"][0]
        p_val = gc_results[lag][0]["ssr_ftest"][1]
        lag_results[lag] = {"f_stat": round(f_stat, 4), "p_value": round(p_val, 6)}
        if p_val < min_p:
            min_p = p_val
            best_lag = lag

    is_significant = min_p < alpha

    result = StatTestResult(
        test_name="Granger Causality",
        statistic=float(gc_results[best_lag][0]["ssr_ftest"][0]),
        p_value=float(min_p),
        is_significant=is_significant,
        alpha=alpha,
        interpretation=(
            f"x Granger-causes y (best lag={best_lag}, p={min_p:.6f})"
            if is_significant
            else f"No Granger causality detected (min p={min_p:.6f})"
        ),
        details={"best_lag": best_lag, "lag_results": lag_results},
    )

    logger.info("Granger causality: best_lag=%d, min_p=%.6f", best_lag, min_p)
    return result
