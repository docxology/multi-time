"""
Matplotlib-based plotting functions for time series analysis.

Facade module re-exporting from focused sub-modules for backward compatibility.
"""

from multi_time.visualization.series import plot_series, plot_multi_series_panel, plot_series_correlation
from multi_time.visualization.forecast import plot_forecast, plot_residuals
from multi_time.visualization.diagnostics import (
    plot_acf_pacf,
    plot_decomposition,
    plot_diagnostics,
)
from multi_time.visualization.statistics import (
    plot_rolling_statistics,
    plot_distribution,
    plot_lag_scatter,
    plot_boxplot_by_period,
    plot_correlation_heatmap,
    plot_stationarity_summary,
    plot_validation_summary,
    plot_missing_data,
)
from multi_time.visualization.comparison import (
    plot_model_comparison,
    plot_error_distribution,
    plot_cumulative_error,
)

__all__ = [
    # Series
    "plot_series",
    "plot_multi_series_panel",
    "plot_series_correlation",
    # Forecast
    "plot_forecast",
    "plot_residuals",
    # Diagnostics
    "plot_acf_pacf",
    "plot_decomposition",
    "plot_diagnostics",
    # Statistics
    "plot_rolling_statistics",
    "plot_distribution",
    "plot_lag_scatter",
    "plot_boxplot_by_period",
    "plot_correlation_heatmap",
    "plot_stationarity_summary",
    "plot_validation_summary",
    "plot_missing_data",
    # Comparison
    "plot_model_comparison",
    "plot_error_distribution",
    "plot_cumulative_error",
]

