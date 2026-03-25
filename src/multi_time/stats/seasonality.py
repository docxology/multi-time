"""
Seasonality and heteroscedasticity tests.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_arch

from multi_time.stats.result import StatTestResult

logger = logging.getLogger(__name__)


def test_seasonality(
    data: pd.Series,
    period: int | None = None,
    alpha: float = 0.05,
) -> StatTestResult:
    """Test for seasonality strength via decomposition.

    Uses the ratio of seasonal variance to residual+seasonal variance.
    A value above 0.64 indicates strong seasonality (Wang et al., 2006).

    Args:
        data: Numeric pandas Series.
        period: Seasonal period. Auto-detected if None.
        alpha: Not directly used but kept for API consistency.

    Returns:
        StatTestResult with seasonality strength metric.
    """
    clean = data.dropna()

    if period is None:
        freq = pd.infer_freq(clean.index)
        period_map = {"H": 24, "D": 7, "B": 5, "W": 52, "MS": 12, "M": 12, "QS": 4, "Q": 4}
        period = period_map.get(freq, 12) if freq else 12

    if len(clean) < 2 * period:
        return StatTestResult(
            test_name="Seasonality Strength",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            alpha=alpha,
            interpretation="Series too short for seasonality detection",
        )

    decomp = seasonal_decompose(clean, model="additive", period=period)
    seasonal_var = np.nanvar(decomp.seasonal)
    residual_var = np.nanvar(decomp.resid)

    # Seasonality strength: 1 - Var(R) / Var(S+R)
    total_var = seasonal_var + residual_var
    strength = 1.0 - (residual_var / total_var) if total_var > 0 else 0.0
    is_seasonal = strength > 0.64

    result = StatTestResult(
        test_name="Seasonality Strength",
        statistic=float(strength),
        p_value=float(1.0 - strength),
        is_significant=is_seasonal,
        alpha=alpha,
        interpretation=(
            f"Strong seasonality detected (strength={strength:.4f} > 0.64)"
            if is_seasonal
            else f"Weak/no seasonality (strength={strength:.4f} <= 0.64)"
        ),
        details={"period": period, "seasonal_var": float(seasonal_var), "residual_var": float(residual_var)},
    )

    logger.info("Seasonality test: strength=%.4f, period=%d", strength, period)
    return result


def test_heteroscedasticity(data: pd.Series, nlags: int = 5, alpha: float = 0.05) -> StatTestResult:
    """Test for heteroscedasticity using Engle's ARCH test.

    Args:
        data: Numeric pandas Series.
        nlags: Number of lags for ARCH test.
        alpha: Significance level.

    Returns:
        StatTestResult for ARCH effect presence.
    """
    clean = data.dropna().values

    arch_stat, arch_p, _, _ = het_arch(clean, nlags=nlags)
    is_significant = arch_p < alpha

    result = StatTestResult(
        test_name="ARCH Heteroscedasticity",
        statistic=float(arch_stat),
        p_value=float(arch_p),
        is_significant=is_significant,
        alpha=alpha,
        interpretation=(
            "ARCH effects detected (heteroscedastic)"
            if is_significant
            else "No ARCH effects (homoscedastic)"
        ),
        details={"nlags": nlags},
    )

    logger.info("ARCH test: stat=%.4f, p=%.4f", arch_stat, arch_p)
    return result
