"""
Forecast and residual visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from multi_time.visualization.core import check_matplotlib, save_or_show, plt, COLORS

logger = logging.getLogger(__name__)


def plot_forecast(
    y_train: pd.Series,
    y_pred: pd.Series,
    y_test: pd.Series | None = None,
    intervals: pd.DataFrame | None = None,
    title: str = "Forecast",
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Plot forecast with optional confidence intervals and actuals.

    Args:
        y_train: Historical training data.
        y_pred: Forecast predictions.
        y_test: Optional actual values for comparison.
        intervals: Optional prediction intervals DataFrame.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(y_train.index, y_train.values, label="Historical", color=COLORS["primary"], linewidth=1.2)
    ax.plot(y_pred.index, y_pred.values, label="Forecast", color=COLORS["accent"], linewidth=2, linestyle="--")

    if y_test is not None:
        ax.plot(y_test.index, y_test.values, label="Actual", color=COLORS["secondary"], linewidth=1.2)

    if intervals is not None and len(intervals.columns) >= 2:
        lower = intervals.iloc[:, 0]
        upper = intervals.iloc[:, -1]
        ax.fill_between(intervals.index, lower, upper, alpha=0.2, color=COLORS["accent"], label="Prediction Interval")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    logger.info("Forecast plot: train=%d, pred=%d", len(y_train), len(y_pred))
    return save_or_show(fig, save_path)


def plot_residuals(
    residuals: pd.Series,
    title: str = "Residual Diagnostics",
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Plot residual analysis: residuals over time and histogram.

    Args:
        residuals: Residual series (y_true - y_pred).
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    clean = residuals.dropna()

    ax1.plot(clean.index, clean.values, color=COLORS["primary"], linewidth=0.8)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=0.5)
    ax1.set_title("Residuals Over Time")
    ax1.set_ylabel("Residual")
    ax1.grid(True, alpha=0.3)

    ax2.hist(clean.values, bins=30, color=COLORS["secondary"], alpha=0.7, edgecolor="white")
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=0.5)
    ax2.set_title("Residual Distribution")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Residual plot rendered: n=%d, mean=%.4f", len(clean), clean.mean())
    return save_or_show(fig, save_path)
