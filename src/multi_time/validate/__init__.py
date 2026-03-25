"""
multi_time.validate — Time series validation subsystem.

Provides validation, frequency detection, patchiness analysis,
and multi-series frequency harmonization.
"""

from multi_time.validate.validators import (
    validate_series,
    detect_frequency,
    assess_patchiness,
    harmonize_frequencies,
    ValidationResult,
    PatchinessResult,
)

__all__ = [
    "validate_series",
    "detect_frequency",
    "assess_patchiness",
    "harmonize_frequencies",
    "ValidationResult",
    "PatchinessResult",
]
