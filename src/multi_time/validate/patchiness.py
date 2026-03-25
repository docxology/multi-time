"""
Patchiness and gap analysis for time series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from multi_time.validate.frequency import detect_frequency

logger = logging.getLogger(__name__)


@dataclass
class PatchinessResult:
    """Result of patchiness/gap analysis.

    Attributes:
        n_gaps: Number of gaps found.
        total_missing_periods: Total number of missing time periods.
        gap_sizes: List of individual gap sizes (in periods).
        longest_gap: Size of the longest gap.
        mean_gap_size: Average gap size.
        gap_locations: List of (start, end) tuples for each gap.
        patchiness_score: 0-1 score (0 = no gaps, 1 = entirely gaps).
    """

    n_gaps: int = 0
    total_missing_periods: int = 0
    gap_sizes: list[int] = field(default_factory=list)
    longest_gap: int = 0
    mean_gap_size: float = 0.0
    gap_locations: list[tuple[Any, Any]] = field(default_factory=list)
    patchiness_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_gaps": self.n_gaps,
            "total_missing_periods": self.total_missing_periods,
            "gap_sizes": self.gap_sizes,
            "longest_gap": self.longest_gap,
            "mean_gap_size": round(self.mean_gap_size, 4),
            "gap_locations": [(str(s), str(e)) for s, e in self.gap_locations],
            "patchiness_score": round(self.patchiness_score, 4),
        }


def assess_patchiness(
    data: pd.Series | pd.DataFrame, freq: str | None = None
) -> PatchinessResult:
    """Analyze gaps and patchiness in a time series.

    Identifies contiguous gaps by reindexing to a regular grid.

    Args:
        data: Time series with DatetimeIndex.
        freq: Frequency to use for regular grid. Auto-detected if None.

    Returns:
        PatchinessResult with gap analysis.
    """
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    result = PatchinessResult()

    if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning("Patchiness analysis requires DatetimeIndex")
        return result

    if len(data) < 2:
        return result

    # Detect frequency if not provided
    if freq is None:
        freq_info = detect_frequency(data)
        freq = freq_info.get("inferred_freq")
        if freq is None:
            logger.warning("Could not detect frequency for patchiness analysis")
            return result

    # Create regular grid and find gaps
    try:
        full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)
    except ValueError as e:
        logger.error("Failed to create regular grid: %s", e)
        return result

    reindexed = data.reindex(full_index)
    is_missing = reindexed.isna()

    # Find contiguous gaps
    gap_groups = (is_missing != is_missing.shift()).cumsum()
    gaps = []
    for group_id, group in reindexed[is_missing].groupby(gap_groups[is_missing]):
        gap_size = len(group)
        gap_start = group.index[0]
        gap_end = group.index[-1]
        gaps.append((gap_start, gap_end, gap_size))

    result.n_gaps = len(gaps)
    result.gap_sizes = [g[2] for g in gaps]
    result.gap_locations = [(g[0], g[1]) for g in gaps]
    result.total_missing_periods = sum(result.gap_sizes)
    result.longest_gap = max(result.gap_sizes) if result.gap_sizes else 0
    result.mean_gap_size = float(np.mean(result.gap_sizes)) if result.gap_sizes else 0.0
    result.patchiness_score = result.total_missing_periods / len(full_index) if len(full_index) > 0 else 0.0

    logger.info(
        "Patchiness: %d gaps, longest=%d, score=%.4f",
        result.n_gaps,
        result.longest_gap,
        result.patchiness_score,
    )
    return result
