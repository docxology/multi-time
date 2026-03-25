"""
Stationarity tests: ADF and KPSS.
"""

from __future__ import annotations

import logging

import warnings
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

from multi_time.stats.result import StatTestResult

logger = logging.getLogger(__name__)


def test_stationarity(data: pd.Series, alpha: float = 0.05) -> dict[str, StatTestResult]:
    """Test stationarity using ADF and KPSS tests.

    ADF: H0 = unit root (non-stationary). Reject H0 → stationary.
    KPSS: H0 = stationary. Reject H0 → non-stationary.

    Args:
        data: Numeric pandas Series.
        alpha: Significance level.

    Returns:
        Dict with 'adf' and 'kpss' StatTestResult objects.
    """
    clean = data.dropna().values
    results: dict[str, StatTestResult] = {}

    # ADF test
    adf_stat, adf_p, adf_used_lag, adf_nobs, adf_critical, adf_icbest = adfuller(clean)
    adf_significant = adf_p < alpha
    results["adf"] = StatTestResult(
        test_name="Augmented Dickey-Fuller",
        statistic=float(adf_stat),
        p_value=float(adf_p),
        is_significant=adf_significant,
        alpha=alpha,
        interpretation=(
            "Series is stationary (reject unit root H0)"
            if adf_significant
            else "Series is non-stationary (cannot reject unit root H0)"
        ),
        details={
            "used_lag": int(adf_used_lag),
            "nobs": int(adf_nobs),
            "critical_values": {k: round(v, 4) for k, v in adf_critical.items()},
        },
    )

    # KPSS test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_stat, kpss_p, kpss_lags, kpss_critical = kpss(clean, regression="c", nlags="auto")
    kpss_significant = kpss_p < alpha
    results["kpss"] = StatTestResult(
        test_name="KPSS",
        statistic=float(kpss_stat),
        p_value=float(kpss_p),
        is_significant=kpss_significant,
        alpha=alpha,
        interpretation=(
            "Series is non-stationary (reject stationarity H0)"
            if kpss_significant
            else "Series is stationary (cannot reject stationarity H0)"
        ),
        details={
            "lags_used": int(kpss_lags),
            "critical_values": {k: round(v, 4) for k, v in kpss_critical.items()},
        },
    )

    logger.info(
        "Stationarity tests: ADF p=%.4f (%s), KPSS p=%.4f (%s)",
        adf_p,
        "stationary" if adf_significant else "non-stationary",
        kpss_p,
        "non-stationary" if kpss_significant else "stationary",
    )
    return results
