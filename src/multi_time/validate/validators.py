"""
Time series validation, frequency detection, and patchiness analysis.

Facade module re-exporting from focused sub-modules for backward compatibility.
"""

from multi_time.validate.validation import ValidationResult, validate_series
from multi_time.validate.frequency import detect_frequency
from multi_time.validate.patchiness import PatchinessResult, assess_patchiness
from multi_time.validate.harmonize import harmonize_frequencies

__all__ = [
    "ValidationResult",
    "PatchinessResult",
    "validate_series",
    "detect_frequency",
    "assess_patchiness",
    "harmonize_frequencies",
]
