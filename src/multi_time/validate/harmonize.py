"""
Multi-series frequency harmonization.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def harmonize_frequencies(
    series_list: list[pd.Series],
    target_freq: str,
    method: str = "ffill",
) -> list[pd.Series]:
    """Resample multiple series to a common frequency.

    Args:
        series_list: List of time series with DatetimeIndex.
        target_freq: Target frequency string (e.g., 'D', 'H', 'M').
        method: Fill method after resampling ('ffill', 'bfill', 'interpolate').

    Returns:
        List of resampled series at the target frequency.
    """
    harmonized = []
    for i, s in enumerate(series_list):
        if not isinstance(s.index, pd.DatetimeIndex):
            logger.warning("Series %d does not have DatetimeIndex, skipping", i)
            harmonized.append(s)
            continue

        resampled = s.resample(target_freq)
        if method == "interpolate":
            result = resampled.mean().interpolate(method="time")
        elif method == "bfill":
            result = resampled.mean().bfill()
        else:
            result = resampled.mean().ffill()

        harmonized.append(result)
        logger.info("Harmonized series %d to freq=%s (%d -> %d pts)", i, target_freq, len(s), len(result))

    return harmonized
