"""
Model comparison and multi-forecast visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_time.visualization.core import check_matplotlib, save_or_show, plt, COLORS

logger = logging.getLogger(__name__)


def plot_model_comparison(
    y_test: pd.Series,
    predictions: dict[str, pd.Series],
    metrics: dict[str, dict[str, float]] | None = None,
    title: str = "Model Comparison",
    figsize: tuple[int, int] = (14, 8),
    save_path: str | Path | None = None,
) -> Any:
    """Plot multiple model forecasts against actuals with optional metric annotations.

    Args:
        y_test: Actual test values.
        predictions: Dict mapping model name to prediction Series.
        metrics: Dict mapping model name to metric dict (e.g. {'mae': 1.2}).
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

    # Forecast comparison
    ax1.plot(y_test.index, y_test.values, color="black", linewidth=2, label="Actual", zorder=10)

    palette = [COLORS["primary"], COLORS["accent"], COLORS["secondary"],
               COLORS["highlight"], COLORS["warning"]]
    for i, (name, preds) in enumerate(predictions.items()):
        color = palette[i % len(palette)]
        label = name
        if metrics and name in metrics:
            mae = metrics[name].get("mae", metrics[name].get("MAE", "?"))
            label = f"{name} (MAE={float(mae):.3f})" if isinstance(mae, (int, float)) else name
        ax1.plot(preds.index, preds.values, color=color, linewidth=1.5, linestyle="--", label=label)

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Value")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Error comparison (bar chart)
    if metrics:
        model_names = list(predictions.keys())
        mae_values = [float(metrics.get(m, {}).get("mae", metrics.get(m, {}).get("MAE", 0)))
                      for m in model_names]
        colors = [palette[i % len(palette)] for i in range(len(model_names))]
        ax2.barh(model_names, mae_values, color=colors, alpha=0.7)
        ax2.set_xlabel("MAE")
        ax2.set_title("Error Comparison", fontsize=12)
        ax2.grid(True, alpha=0.3, axis="x")
    else:
        ax2.text(0.5, 0.5, "No metrics provided", ha="center", va="center", fontsize=12)
        ax2.set_axis_off()

    fig.tight_layout()
    logger.info("Model comparison: %d models", len(predictions))
    return save_or_show(fig, save_path)


def plot_error_distribution(
    y_test: pd.Series,
    predictions: dict[str, pd.Series],
    title: str = "Error Distributions",
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Plot error (residual) distributions for multiple models.

    Args:
        y_test: Actual values.
        predictions: Dict mapping model name to predictions.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
    if n_models == 1:
        axes = [axes]

    palette = [COLORS["primary"], COLORS["accent"], COLORS["secondary"],
               COLORS["highlight"], COLORS["warning"]]

    for i, (name, preds) in enumerate(predictions.items()):
        ax = axes[i]
        common_idx = y_test.index.intersection(preds.index)
        errors = y_test.loc[common_idx] - preds.loc[common_idx]
        color = palette[i % len(palette)]

        ax.hist(errors.values, bins=20, color=color, alpha=0.7, edgecolor="white", density=True)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(x=errors.mean(), color="black", linestyle="-", linewidth=1.5, label=f"μ={errors.mean():.3f}")
        ax.set_title(f"{name}\nσ={errors.std():.3f}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    logger.info("Error distribution: %d models", n_models)
    return save_or_show(fig, save_path)


def plot_cumulative_error(
    y_test: pd.Series,
    predictions: dict[str, pd.Series],
    title: str = "Cumulative Absolute Error",
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Plot cumulative absolute error over time for multiple models.

    Args:
        y_test: Actual values.
        predictions: Dict mapping model name to predictions.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)

    palette = [COLORS["primary"], COLORS["accent"], COLORS["secondary"],
               COLORS["highlight"], COLORS["warning"]]

    for i, (name, preds) in enumerate(predictions.items()):
        common_idx = y_test.index.intersection(preds.index)
        abs_errors = (y_test.loc[common_idx] - preds.loc[common_idx]).abs()
        cumulative = abs_errors.cumsum()
        color = palette[i % len(palette)]
        ax.plot(cumulative.index, cumulative.values, color=color, linewidth=1.5,
                label=f"{name} (total={cumulative.iloc[-1]:.2f})")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative |Error|")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    logger.info("Cumulative error: %d models", len(predictions))
    return save_or_show(fig, save_path)
