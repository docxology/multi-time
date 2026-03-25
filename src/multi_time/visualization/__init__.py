"""
multi_time.visualization — Time series plotting, diagnostics, and analysis.

Provides matplotlib-based visualization functions for time series analysis
including series, forecasts, decompositions, ACF/PACF, diagnostics,
statistical analysis, and model comparison.
"""

from multi_time.visualization.plots import (
    # Series
    plot_series,
    plot_multi_series_panel,
    plot_series_correlation,
    # Forecast
    plot_forecast,
    plot_residuals,
    # Diagnostics
    plot_acf_pacf,
    plot_decomposition,
    plot_diagnostics,
    # Statistics
    plot_rolling_statistics,
    plot_distribution,
    plot_lag_scatter,
    plot_boxplot_by_period,
    plot_correlation_heatmap,
    plot_stationarity_summary,
    plot_validation_summary,
    plot_missing_data,
    # Comparison
    plot_model_comparison,
    plot_error_distribution,
    plot_cumulative_error,
)

__all__ = [
    "plot_series",
    "plot_multi_series_panel",
    "plot_series_correlation",
    "plot_forecast",
    "plot_residuals",
    "plot_acf_pacf",
    "plot_decomposition",
    "plot_diagnostics",
    "plot_rolling_statistics",
    "plot_distribution",
    "plot_lag_scatter",
    "plot_boxplot_by_period",
    "plot_correlation_heatmap",
    "plot_stationarity_summary",
    "plot_validation_summary",
    "plot_missing_data",
    "plot_model_comparison",
    "plot_error_distribution",
    "plot_cumulative_error",
]

