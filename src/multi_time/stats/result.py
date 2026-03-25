"""
Result dataclass for statistical tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StatTestResult:
    """Result of a statistical test.

    Attributes:
        test_name: Name of the test.
        statistic: Test statistic value.
        p_value: p-value of the test.
        is_significant: Whether result is significant at the given alpha.
        alpha: Significance level used.
        interpretation: Human-readable interpretation.
        details: Additional test-specific information.
    """

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float = 0.05
    interpretation: str = ""
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "interpretation": self.interpretation,
            "details": self.details,
        }
