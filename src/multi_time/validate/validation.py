"""
Validation result dataclass and core series validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of time series validation.

    Attributes:
        is_valid: Whether the series passes all validation checks.
        n_observations: Number of data points.
        n_missing: Count of NaN/null values.
        missing_pct: Percentage of missing values.
        dtype: Data type of values.
        index_type: Type of the index.
        is_monotonic: Whether index is monotonically increasing.
        has_duplicates: Whether index has duplicate entries.
        warnings: List of warning messages.
        errors: List of error messages.
    """

    is_valid: bool = True
    n_observations: int = 0
    n_missing: int = 0
    missing_pct: float = 0.0
    dtype: str = ""
    index_type: str = ""
    is_monotonic: bool = True
    has_duplicates: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "n_observations": self.n_observations,
            "n_missing": self.n_missing,
            "missing_pct": round(self.missing_pct, 4),
            "dtype": self.dtype,
            "index_type": self.index_type,
            "is_monotonic": self.is_monotonic,
            "has_duplicates": self.has_duplicates,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def validate_series(data: pd.Series | pd.DataFrame) -> ValidationResult:
    """Validate a time series for correctness and quality.

    Checks type, missing values, index monotonicity, duplicates, and numeric dtype.

    Args:
        data: Time series as pandas Series or single-column DataFrame.

    Returns:
        ValidationResult with detailed diagnostics.
    """
    result = ValidationResult()

    # Convert DataFrame to Series if single-column
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            data = data.iloc[:, 0]
            logger.info("Converted single-column DataFrame to Series")
        else:
            result.errors.append(
                f"Expected single-column DataFrame, got {data.shape[1]} columns"
            )
            result.is_valid = False
            return result

    if not isinstance(data, pd.Series):
        result.errors.append(f"Expected pd.Series, got {type(data).__name__}")
        result.is_valid = False
        return result

    result.n_observations = len(data)
    if result.n_observations == 0:
        result.errors.append("Series is empty")
        result.is_valid = False
        return result

    # Dtype check
    result.dtype = str(data.dtype)
    if not pd.api.types.is_numeric_dtype(data):
        result.errors.append(f"Series dtype '{data.dtype}' is not numeric")
        result.is_valid = False

    # Missing values
    result.n_missing = int(data.isna().sum())
    result.missing_pct = result.n_missing / result.n_observations * 100

    if result.missing_pct > 50:
        result.warnings.append(f"High missing rate: {result.missing_pct:.1f}%")
    elif result.n_missing > 0:
        logger.info("Found %d missing values (%.2f%%)", result.n_missing, result.missing_pct)

    # Index checks
    result.index_type = type(data.index).__name__
    result.is_monotonic = bool(data.index.is_monotonic_increasing)
    result.has_duplicates = bool(data.index.has_duplicates)

    if not result.is_monotonic:
        result.warnings.append("Index is not monotonically increasing")

    if result.has_duplicates:
        n_dups = int(data.index.duplicated().sum())
        result.warnings.append(f"Index has {n_dups} duplicate entries")

    # Final validity
    if result.errors:
        result.is_valid = False

    logger.info(
        "Validation complete: valid=%s, n=%d, missing=%.2f%%",
        result.is_valid,
        result.n_observations,
        result.missing_pct,
    )
    return result
