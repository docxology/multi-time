"""
Normality tests: Shapiro-Wilk and Jarque-Bera.
"""

from __future__ import annotations

import logging

import pandas as pd
from scipy import stats as scipy_stats

from multi_time.stats.result import StatTestResult

logger = logging.getLogger(__name__)


def test_normality(data: pd.Series, alpha: float = 0.05) -> dict[str, StatTestResult]:
    """Test normality using Shapiro-Wilk and Jarque-Bera tests.

    Args:
        data: Numeric pandas Series.
        alpha: Significance level.

    Returns:
        Dict with 'shapiro' and 'jarque_bera' StatTestResult objects.
    """
    clean = data.dropna().values
    results: dict[str, StatTestResult] = {}

    # Shapiro-Wilk (limited to 5000 observations)
    sample = clean[:5000] if len(clean) > 5000 else clean
    sw_stat, sw_p = scipy_stats.shapiro(sample)
    sw_significant = sw_p < alpha
    results["shapiro"] = StatTestResult(
        test_name="Shapiro-Wilk",
        statistic=float(sw_stat),
        p_value=float(sw_p),
        is_significant=sw_significant,
        alpha=alpha,
        interpretation=(
            "Data is NOT normally distributed (reject H0)"
            if sw_significant
            else "Data is consistent with normal distribution (cannot reject H0)"
        ),
    )

    # Jarque-Bera
    jb_stat, jb_p = scipy_stats.jarque_bera(clean)
    jb_significant = jb_p < alpha
    results["jarque_bera"] = StatTestResult(
        test_name="Jarque-Bera",
        statistic=float(jb_stat),
        p_value=float(jb_p),
        is_significant=jb_significant,
        alpha=alpha,
        interpretation=(
            "Data is NOT normally distributed (reject H0)"
            if jb_significant
            else "Data is consistent with normal distribution (cannot reject H0)"
        ),
    )

    logger.info("Normality tests: Shapiro p=%.4f, Jarque-Bera p=%.4f", sw_p, jb_p)
    return results
