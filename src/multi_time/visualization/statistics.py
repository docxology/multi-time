"""
Statistical visualization: rolling stats, boxplots, heatmaps, distribution, lag plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_time.visualization.core import check_matplotlib, save_or_show, plt, COLORS

logger = logging.getLogger(__name__)


def plot_rolling_statistics(
    data: pd.Series,
    window: int = 12,
    title: str = "Rolling Statistics",
    figsize: tuple[int, int] = (14, 8),
    save_path: str | Path | None = None,
) -> Any:
    """Plot rolling mean, std, min, max with confidence bands.

    Args:
        data: Numeric pandas Series.
        window: Rolling window size.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    clean = data.dropna()

    rolling_mean = clean.rolling(window).mean()
    rolling_std = clean.rolling(window).std()
    rolling_min = clean.rolling(window).min()
    rolling_max = clean.rolling(window).max()

    # Top: series + rolling mean + confidence band
    ax1.plot(clean.index, clean.values, color=COLORS["primary"], alpha=0.4, linewidth=0.8, label="Raw")
    ax1.plot(rolling_mean.index, rolling_mean.values, color=COLORS["accent"], linewidth=2, label=f"Rolling Mean ({window})")
    ax1.fill_between(
        clean.index,
        (rolling_mean - 2 * rolling_std).values,
        (rolling_mean + 2 * rolling_std).values,
        alpha=0.15, color=COLORS["accent"], label="±2σ band",
    )
    ax1.set_title("Series with Rolling Mean ± 2σ", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bottom: rolling std + range
    ax2.plot(rolling_std.index, rolling_std.values, color=COLORS["highlight"], linewidth=1.5, label="Rolling Std")
    ax2.fill_between(
        clean.index, rolling_min.values, rolling_max.values,
        alpha=0.15, color=COLORS["secondary"], label="Rolling Range",
    )
    ax2.set_title("Rolling Volatility & Range", fontsize=12)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Rolling statistics plot: window=%d", window)
    return save_or_show(fig, save_path)


def plot_distribution(
    data: pd.Series,
    bins: int = 40,
    title: str = "Distribution Analysis",
    show_stats: bool = True,
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Plot histogram with KDE, normal overlay, and descriptive statistics.

    Args:
        data: Numeric pandas Series.
        bins: Number of histogram bins.
        title: Plot title.
        show_stats: Annotate with mean, std, skew, kurtosis.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    from scipy import stats as scipy_stats

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    clean = data.dropna().values

    # Histogram + KDE
    ax1.hist(clean, bins=bins, density=True, color=COLORS["primary"], alpha=0.6, edgecolor="white", label="Data")
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    kde = scipy_stats.gaussian_kde(clean)
    ax1.plot(x, kde(x), color=COLORS["accent"], linewidth=2, label="KDE")

    # Normal overlay
    mu, sigma = clean.mean(), clean.std()
    normal_pdf = scipy_stats.norm.pdf(x, mu, sigma)
    ax1.plot(x, normal_pdf, color=COLORS["secondary"], linewidth=1.5, linestyle="--", label="Normal fit")
    ax1.set_title("Histogram + KDE")
    ax1.legend()

    if show_stats:
        stats_text = (
            f"n = {len(clean)}\n"
            f"μ = {mu:.4f}\n"
            f"σ = {sigma:.4f}\n"
            f"skew = {scipy_stats.skew(clean):.4f}\n"
            f"kurt = {scipy_stats.kurtosis(clean):.4f}"
        )
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 verticalalignment="top", fontsize=9, fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Box + Violin
    parts = ax2.violinplot(clean, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(COLORS["primary"])
        pc.set_alpha(0.4)
    ax2.set_title("Violin Plot")
    ax2.set_ylabel("Value")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Distribution plot: n=%d, mean=%.4f, std=%.4f", len(clean), mu, sigma)
    return save_or_show(fig, save_path)


def plot_lag_scatter(
    data: pd.Series,
    lags: list[int] | None = None,
    title: str = "Lag Scatter Plots",
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Plot scatter of y(t) vs y(t-lag) for multiple lags.

    Args:
        data: Numeric pandas Series.
        lags: List of lag values. Default: [1, 7, 12].
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    if lags is None:
        lags = [1, 7, 12]
    clean = data.dropna()

    n_plots = len(lags)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    palette = [COLORS["primary"], COLORS["accent"], COLORS["secondary"], COLORS["highlight"]]
    for i, lag in enumerate(lags):
        ax = axes[i]
        x = clean.iloc[lag:]
        y = clean.iloc[:-lag]
        color = palette[i % len(palette)]
        ax.scatter(y.values, x.values, alpha=0.4, s=10, color=color)

        # Correlation annotation
        corr = np.corrcoef(y.values, x.values)[0, 1]
        ax.set_title(f"Lag {lag} (r={corr:.3f})", fontsize=11)
        ax.set_xlabel(f"y(t)")
        ax.set_ylabel(f"y(t+{lag})")
        ax.grid(True, alpha=0.3)

        # 45-degree line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Lag scatter plot: lags=%s", lags)
    return save_or_show(fig, save_path)


def plot_boxplot_by_period(
    data: pd.Series,
    period: str = "month",
    title: str = "Seasonal Boxplot",
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Plot boxplots grouped by time period (month, day of week, hour, quarter).

    Args:
        data: Series with DatetimeIndex.
        period: Grouping period — 'month', 'dayofweek', 'hour', 'quarter'.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    clean = data.dropna()

    if not isinstance(clean.index, pd.DatetimeIndex):
        raise ValueError("boxplot_by_period requires DatetimeIndex")

    period_map = {
        "month": (clean.index.month, [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]),
        "dayofweek": (clean.index.dayofweek, [
            "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
        ]),
        "hour": (clean.index.hour, [str(h) for h in range(24)]),
        "quarter": (clean.index.quarter, ["Q1", "Q2", "Q3", "Q4"]),
    }

    if period not in period_map:
        raise ValueError(f"Unknown period: {period}. Available: {list(period_map.keys())}")

    groups, group_labels = period_map[period]
    df = pd.DataFrame({"value": clean.values, "group": groups})

    fig, ax = plt.subplots(figsize=figsize)
    unique_groups = sorted(df["group"].unique())
    box_data = [df[df["group"] == g]["value"].values for g in unique_groups]
    bp = ax.boxplot(box_data, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["primary"])
        patch.set_alpha(0.5)
    for median in bp["medians"]:
        median.set_color(COLORS["accent"])
        median.set_linewidth(2)

    labels = [group_labels[int(g) - 1] if period in ("month", "quarter")
              else group_labels[int(g)] for g in unique_groups]
    ax.set_xticklabels(labels, rotation=45 if period == "hour" else 0)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    logger.info("Boxplot by %s: %d groups", period, len(unique_groups))
    return save_or_show(fig, save_path)


def plot_correlation_heatmap(
    data: pd.DataFrame,
    method: str = "pearson",
    title: str = "Correlation Heatmap",
    figsize: tuple[int, int] = (10, 8),
    save_path: str | Path | None = None,
) -> Any:
    """Plot correlation matrix as a heatmap.

    Args:
        data: DataFrame with numeric columns.
        method: Correlation method ('pearson', 'spearman', 'kendall').
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()

    corr = data.corr(method=method)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    # Annotate cells
    for i in range(len(corr)):
        for j in range(len(corr)):
            color = "white" if abs(corr.values[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Correlation heatmap: %d×%d, method=%s", *corr.shape, method)
    return save_or_show(fig, save_path)


def plot_stationarity_summary(
    data: pd.Series,
    window: int = 12,
    title: str = "Stationarity Visual Check",
    figsize: tuple[int, int] = (14, 10),
    save_path: str | Path | None = None,
) -> Any:
    """Visual stationarity check: series, rolling mean/std, first differences, histogram.

    Args:
        data: Numeric pandas Series.
        window: Rolling window.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    clean = data.dropna()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Original + rolling mean
    axes[0, 0].plot(clean.index, clean.values, color=COLORS["primary"], alpha=0.5, linewidth=0.8)
    rm = clean.rolling(window).mean()
    axes[0, 0].plot(rm.index, rm.values, color=COLORS["accent"], linewidth=2)
    axes[0, 0].set_title(f"Series + Rolling Mean ({window})")
    axes[0, 0].grid(True, alpha=0.3)

    # Rolling std
    rs = clean.rolling(window).std()
    axes[0, 1].plot(rs.index, rs.values, color=COLORS["highlight"], linewidth=1.5)
    axes[0, 1].set_title(f"Rolling Std ({window})")
    axes[0, 1].grid(True, alpha=0.3)

    # First difference
    diff1 = clean.diff().dropna()
    axes[1, 0].plot(diff1.index, diff1.values, color=COLORS["secondary"], linewidth=0.8)
    axes[1, 0].axhline(y=0, color="red", linestyle="--", linewidth=0.5)
    axes[1, 0].set_title("First Difference")
    axes[1, 0].grid(True, alpha=0.3)

    # Histogram of differences
    axes[1, 1].hist(diff1.values, bins=30, color=COLORS["warning"], alpha=0.7, edgecolor="white")
    axes[1, 1].axvline(x=0, color="red", linestyle="--", linewidth=0.5)
    mean_diff = diff1.mean()
    axes[1, 1].set_title(f"Diff Distribution (μ={mean_diff:.4f})")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Stationarity summary: window=%d, diff_mean=%.4f", window, mean_diff)
    return save_or_show(fig, save_path)


def plot_validation_summary(
    data: pd.Series,
    validation_result: dict | None = None,
    freq_result: dict | None = None,
    title: str = "Data Quality Summary",
    figsize: tuple[int, int] = (16, 10),
    save_path: str | Path | None = None,
) -> Any:
    """Plot data quality dashboard: series overview, missing pattern, frequency, stats.

    Args:
        data: Input time series.
        validation_result: Dict from validate_series().to_dict().
        freq_result: Dict from detect_frequency().
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Series with NaN markers
    ax = axes[0, 0]
    ax.plot(data.index, data.values, color=COLORS["primary"], linewidth=0.8, label="Series")
    if data.isna().any():
        nan_idx = data[data.isna()].index
        ymin, ymax = data.dropna().min(), data.dropna().max()
        for idx in nan_idx:
            ax.axvline(x=idx, color="red", alpha=0.3, linewidth=0.5)
        ax.set_title(f"Series ({data.isna().sum()} missing values)", fontsize=11)
    else:
        ax.set_title("Series (no missing values)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # 2. Data availability heatmap (chunk the series into blocks)
    ax = axes[0, 1]
    n = len(data)
    n_blocks = min(50, n)
    block_size = max(1, n // n_blocks)
    availability = []
    for i in range(0, n, block_size):
        chunk = data.iloc[i:i + block_size]
        pct_valid = 1.0 - chunk.isna().mean()
        availability.append(pct_valid)
    avail_arr = np.array(availability).reshape(1, -1)
    ax.imshow(avail_arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Data Availability (green=complete, red=missing)", fontsize=11)
    ax.set_yticks([])
    ax.set_xlabel("Block index")

    # 3. Consecutive differences (frequency regularity)
    ax = axes[1, 0]
    if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
        deltas = pd.Series(data.index[1:] - data.index[:-1])
        delta_hours = deltas.dt.total_seconds() / 3600
        ax.bar(range(len(delta_hours)), delta_hours, color=COLORS["accent"], alpha=0.5, width=1.0)
        median_h = delta_hours.median()
        ax.axhline(y=median_h, color=COLORS["secondary"], linestyle="--", linewidth=1.5,
                   label=f"Median: {median_h:.1f}h")
        ax.set_title("Time Deltas Between Observations", fontsize=11)
        ax.set_ylabel("Hours")
        ax.set_xlabel("Observation index")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No DatetimeIndex", ha="center", va="center")

    # 4. Stats annotation panel
    ax = axes[1, 1]
    ax.axis("off")
    lines = [f"{'─'*40}", f" DATA QUALITY REPORT", f"{'─'*40}"]
    lines.append(f" Observations:  {len(data)}")
    lines.append(f" Missing:       {data.isna().sum()} ({data.isna().mean()*100:.1f}%)")
    lines.append(f" Dtype:         {data.dtype}")
    if isinstance(data.index, pd.DatetimeIndex):
        lines.append(f" Start:         {data.index.min()}")
        lines.append(f" End:           {data.index.max()}")
    if validation_result:
        lines.append(f" Valid:          {validation_result.get('is_valid', '?')}")
        lines.append(f" Monotonic:     {validation_result.get('is_monotonic', '?')}")
        lines.append(f" Duplicates:    {validation_result.get('has_duplicates', '?')}")
    if freq_result:
        lines.append(f" Frequency:     {freq_result.get('inferred_freq', '?')}")
        lines.append(f" Regular:       {freq_result.get('is_regular', '?')}")
    clean = data.dropna()
    if len(clean) > 0:
        lines.append(f"{'─'*40}")
        lines.append(f" Mean:          {clean.mean():.4f}")
        lines.append(f" Std:           {clean.std():.4f}")
        lines.append(f" Min:           {clean.min():.4f}")
        lines.append(f" Max:           {clean.max():.4f}")
    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, fontfamily="monospace",
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Validation summary plot: n=%d, missing=%d", len(data), data.isna().sum())
    return save_or_show(fig, save_path)


def plot_missing_data(
    data: pd.Series,
    title: str = "Missing Data Analysis",
    figsize: tuple[int, int] = (14, 8),
    save_path: str | Path | None = None,
) -> Any:
    """Visualize missing data patterns: gap timeline, gap sizes, availability.

    Args:
        data: Time series with potential NaN values.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    check_matplotlib()

    n_missing = data.isna().sum()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Series with gaps highlighted
    ax = axes[0, 0]
    clean = data.dropna()
    ax.plot(clean.index, clean.values, color=COLORS["primary"], linewidth=0.8)
    if n_missing > 0:
        # Highlight gap regions
        is_nan = data.isna()
        gap_starts = []
        gap_ends = []
        in_gap = False
        for i, (idx, val) in enumerate(is_nan.items()):
            if val and not in_gap:
                gap_starts.append(idx)
                in_gap = True
            elif not val and in_gap:
                gap_ends.append(idx)
                in_gap = False
        if in_gap:
            gap_ends.append(data.index[-1])
        for gs, ge in zip(gap_starts, gap_ends):
            ax.axvspan(gs, ge, alpha=0.3, color="red")
    ax.set_title(f"Series with {n_missing} Missing Values", fontsize=11)
    ax.grid(True, alpha=0.3)

    # 2. Missing indicator timeline
    ax = axes[0, 1]
    is_present = (~data.isna()).astype(int)
    ax.fill_between(data.index, 0, is_present, color=COLORS["secondary"], alpha=0.5, step="mid")
    ax.set_title("Data Presence (1=present, 0=missing)", fontsize=11)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Missing", "Present"])
    ax.grid(True, alpha=0.3)

    # 3. Gap size distribution
    ax = axes[1, 0]
    if n_missing > 0:
        gap_sizes = []
        current_gap = 0
        for val in data.isna():
            if val:
                current_gap += 1
            elif current_gap > 0:
                gap_sizes.append(current_gap)
                current_gap = 0
        if current_gap > 0:
            gap_sizes.append(current_gap)

        if gap_sizes:
            ax.bar(range(1, len(gap_sizes) + 1), gap_sizes, color=COLORS["warning"], alpha=0.7)
            ax.set_xlabel("Gap #")
            ax.set_ylabel("Gap Size (observations)")
            ax.set_title(f"{len(gap_sizes)} Gaps (mean={np.mean(gap_sizes):.1f})", fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No gaps found", ha="center", va="center", fontsize=14, color="green")
        ax.set_title("Gap Analysis", fontsize=11)

    # 4. Rolling completeness
    ax = axes[1, 1]
    window = max(7, len(data) // 30)
    completeness = (~data.isna()).astype(float).rolling(window).mean() * 100
    ax.plot(completeness.index, completeness.values, color=COLORS["accent"], linewidth=1.5)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="100%")
    ax.axhline(y=90, color="orange", linestyle="--", alpha=0.5, label="90%")
    ax.set_title(f"Rolling Completeness (window={window})", fontsize=11)
    ax.set_ylabel("% Complete")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    logger.info("Missing data plot: n=%d, missing=%d, gaps=%d",
                len(data), n_missing, len(gap_sizes) if n_missing > 0 else 0)
    return save_or_show(fig, save_path)

