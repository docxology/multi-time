"""
Time series and overlay plot functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from multi_time.visualization.core import check_matplotlib, save_or_show, plt, COLORS

logger = logging.getLogger(__name__)


def plot_series(
    *series: pd.Series,
    title: str = "Time Series",
    ylabel: str = "Value",
    labels: list[str] | None = None,
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Plot one or more time series on the same axes.

    Args:
        *series: One or more pandas Series to plot.
        title: Plot title.
        ylabel: Y-axis label.
        labels: Legend labels. Uses series names if None.
        figsize: Figure size.
        save_path: Optional path to save the plot.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)

    for i, s in enumerate(series):
        label = labels[i] if labels and i < len(labels) else getattr(s, "name", f"Series {i}")
        ax.plot(s.index, s.values, label=label, linewidth=1.2)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    logger.info("Plotted %d series: title='%s'", len(series), title)
    return save_or_show(fig, save_path)


def plot_multi_series_panel(
    series_dict: dict[str, pd.Series],
    title: str = "Multi-Series Panel",
    figsize: tuple[int, int] = (14, None),
    show_overlap: bool = True,
    save_path: str | Path | None = None,
) -> Any:
    """Plot multiple series in stacked panels with shared x-axis.

    Each series gets its own y-axis panel, with frequency annotation.
    Overlap regions between series are highlighted.

    Args:
        series_dict: Dict mapping series names to Series objects.
        title: Overall figure title.
        figsize: Figure size. Height auto-calculated if None.
        show_overlap: Whether to highlight overlapping time regions.
        save_path: Optional path to save the plot.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    n = len(series_dict)
    if n == 0:
        raise ValueError("series_dict must contain at least one series")

    height = figsize[1] if figsize[1] else max(4, n * 2.5)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], height), sharex=True)
    if n == 1:
        axes = [axes]

    color_cycle = list(COLORS.values())
    names = list(series_dict.keys())
    all_series = list(series_dict.values())

    for i, (name, s) in enumerate(series_dict.items()):
        ax = axes[i]
        color = color_cycle[i % len(color_cycle)]
        ax.plot(s.index, s.values, color=color, linewidth=1.2, label=name)

        # Annotate frequency
        freq_str = "unknown"
        if isinstance(s.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(s.index)
            if inferred:
                freq_str = inferred
        ax.set_ylabel(name, fontsize=10)
        ax.text(0.98, 0.95, f"freq={freq_str}, n={len(s)}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="gray", style="italic")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)

        # Highlight overlap regions
        if show_overlap and isinstance(s.index, pd.DatetimeIndex):
            for j, other_s in enumerate(all_series):
                if j != i and isinstance(other_s.index, pd.DatetimeIndex):
                    overlap_start = max(s.index.min(), other_s.index.min())
                    overlap_end = min(s.index.max(), other_s.index.max())
                    if overlap_start < overlap_end:
                        ax.axvspan(overlap_start, overlap_end, alpha=0.08,
                                   color=color_cycle[j % len(color_cycle)])

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Multi-series panel: %d series, title='%s'", n, title)
    return save_or_show(fig, save_path)


def plot_series_correlation(
    series_dict: dict[str, pd.Series],
    title: str = "Cross-Series Correlation",
    method: str = "pearson",
    figsize: tuple[int, int] = (8, 6),
    save_path: str | Path | None = None,
) -> Any:
    """Plot pairwise cross-correlation matrix for multiple series.

    Aligns all series to their overlapping time range before computing
    correlation.

    Args:
        series_dict: Dict mapping series names to Series objects.
        title: Plot title.
        method: Correlation method ('pearson', 'spearman', 'kendall').
        figsize: Figure size.
        save_path: Optional path to save the plot.

    Returns:
        matplotlib Figure object.
    """
    import numpy as np

    check_matplotlib()

    # Build aligned DataFrame from overlapping regions
    df = pd.DataFrame(series_dict)
    # Only keep rows where all series have values
    df_aligned = df.dropna()

    if len(df_aligned) < 3:
        logger.warning("Insufficient overlapping data for correlation (%d rows)", len(df_aligned))
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient overlapping data\nfor correlation analysis",
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title(title)
        return save_or_show(fig, save_path)

    corr = df_aligned.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=color)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    fig.colorbar(im, ax=ax, label=f"{method.title()} Correlation")
    ax.set_title(f"{title}\n({len(df_aligned)} overlapping observations, {method})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    logger.info("Cross-correlation: %d series, %d overlap pts, method=%s",
                len(series_dict), len(df_aligned), method)
    return save_or_show(fig, save_path)
