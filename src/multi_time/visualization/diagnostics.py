"""
Diagnostic and analytical visualization: ACF/PACF, decomposition, diagnostics panel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_time.visualization.core import check_matplotlib, save_or_show, plt, COLORS

logger = logging.getLogger(__name__)


def plot_acf_pacf(
    acf_values: list[float],
    pacf_values: list[float],
    nlags: int | None = None,
    title: str = "ACF and PACF",
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Plot ACF and PACF side by side.

    Args:
        acf_values: Autocorrelation values (from compute_acf_pacf).
        pacf_values: Partial autocorrelation values.
        nlags: Number of lags (auto from data length if None).
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    nlags = nlags or len(acf_values) - 1
    lags = range(nlags + 1)

    ax1.bar(lags, acf_values[:nlags + 1], width=0.3, color=COLORS["primary"], alpha=0.7)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_title("ACF", fontsize=12)
    ax1.set_xlabel("Lag")
    ax1.grid(True, alpha=0.3)

    ax2.bar(lags, pacf_values[:nlags + 1], width=0.3, color=COLORS["accent"], alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_title("PACF", fontsize=12)
    ax2.set_xlabel("Lag")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("ACF/PACF plot: nlags=%d", nlags)
    return save_or_show(fig, save_path)


def plot_decomposition(
    decomposition: dict[str, pd.Series],
    title: str = "Seasonal Decomposition",
    figsize: tuple[int, int] = (14, 10),
    save_path: str | Path | None = None,
) -> Any:
    """Plot seasonal decomposition components.

    Args:
        decomposition: Dict with keys 'observed', 'trend', 'seasonal', 'residual'.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    components = ["observed", "trend", "seasonal", "residual"]
    fig, axes = plt.subplots(len(components), 1, figsize=figsize, sharex=True)

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["warning"], COLORS["highlight"]]
    for ax, name, color in zip(axes, components, colors):
        if name in decomposition and decomposition[name] is not None:
            ax.plot(decomposition[name], color=color, linewidth=1.0)
        ax.set_ylabel(name.capitalize())
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Decomposition plot rendered")
    return save_or_show(fig, save_path)


def plot_diagnostics(
    data: pd.Series,
    title: str = "Series Diagnostics",
    figsize: tuple[int, int] = (14, 10),
    save_path: str | Path | None = None,
) -> Any:
    """Plot diagnostic panel: time series, histogram, Q-Q plot, rolling stats.

    Args:
        data: Numeric pandas Series.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    clean = data.dropna()

    # Time series
    axes[0, 0].plot(clean.index, clean.values, color=COLORS["primary"], linewidth=0.8)
    axes[0, 0].set_title("Time Series")
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram
    axes[0, 1].hist(clean.values, bins=30, color=COLORS["secondary"], alpha=0.7, edgecolor="white")
    axes[0, 1].set_title("Distribution")

    # Q-Q plot
    scipy_stats.probplot(clean.values, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)

    # Rolling mean & std
    window = min(12, len(clean) // 4)
    if window >= 2:
        rolling_mean = clean.rolling(window).mean()
        rolling_std = clean.rolling(window).std()
        axes[1, 1].plot(rolling_mean.index, rolling_mean.values, label="Rolling Mean", color=COLORS["accent"])
        axes[1, 1].plot(rolling_std.index, rolling_std.values, label="Rolling Std", color=COLORS["highlight"])
        axes[1, 1].legend()
    axes[1, 1].set_title("Rolling Statistics")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Diagnostics plot rendered")
    return save_or_show(fig, save_path)
