"""
Shared visualization utilities: matplotlib setup, save logic, style constants.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/CI use
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    logger.warning("matplotlib not installed — visualization functions unavailable")

# Consistent color palette
COLORS = {
    "primary": "#2196F3",
    "secondary": "#4CAF50",
    "accent": "#FF5722",
    "highlight": "#9C27B0",
    "warning": "#FF9800",
}


def check_matplotlib() -> None:
    """Raise ImportError if matplotlib is not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


def save_or_show(fig: Any, save_path: str | Path | None = None) -> Any:
    """Save figure to disk or return it."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", path)
    if HAS_MATPLOTLIB:
        plt.close(fig)
    return fig
