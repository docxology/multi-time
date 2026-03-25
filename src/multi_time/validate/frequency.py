"""
Time series frequency detection.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def detect_frequency(data: pd.Series | pd.DataFrame) -> dict[str, Any]:
    """Detect the frequency of a time series.

    Uses pandas infer_freq with fallback heuristics for irregular series.

    Args:
        data: Time series with DatetimeIndex or PeriodIndex.

    Returns:
        Dict with keys: 'inferred_freq', 'is_regular', 'median_delta',
        'min_delta', 'max_delta', 'delta_std'.
    """
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    idx = data.index
    result: dict[str, Any] = {
        "inferred_freq": None,
        "is_regular": False,
        "median_delta": None,
        "min_delta": None,
        "max_delta": None,
        "delta_std": None,
    }

    # Try pandas infer_freq
    try:
        freq = pd.infer_freq(idx)
        if freq is not None:
            result["inferred_freq"] = freq
            result["is_regular"] = True
            logger.info("Inferred frequency: %s", freq)
    except (TypeError, ValueError) as e:
        logger.warning("pd.infer_freq failed: %s", e)

    # Compute delta statistics for DatetimeIndex
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        deltas = pd.Series(idx[1:] - idx[:-1])
        result["median_delta"] = str(deltas.median())
        result["min_delta"] = str(deltas.min())
        result["max_delta"] = str(deltas.max())
        result["delta_std"] = str(deltas.std())

        # Heuristic frequency detection if infer_freq failed
        if result["inferred_freq"] is None:
            median_td = deltas.median()
            if median_td <= pd.Timedelta(hours=2):
                result["inferred_freq"] = "H"
            elif median_td <= pd.Timedelta(days=2):
                result["inferred_freq"] = "D"
            elif median_td <= pd.Timedelta(days=8):
                result["inferred_freq"] = "W"
            elif median_td <= pd.Timedelta(days=35):
                result["inferred_freq"] = "MS"
            elif median_td <= pd.Timedelta(days=100):
                result["inferred_freq"] = "QS"
            else:
                result["inferred_freq"] = "YS"
            logger.info("Heuristic frequency estimate: %s", result["inferred_freq"])

    elif isinstance(idx, pd.PeriodIndex):
        result["inferred_freq"] = str(idx.freqstr) if idx.freqstr else result["inferred_freq"]
        result["is_regular"] = True

    return result
