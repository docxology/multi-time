# multi_time.visualization

**17 matplotlib-based plot functions across 5 sub-modules**

## Sub-modules

| Module | Functions | Description |
| --- | --- | --- |
| `core.py` | 2 | Agg backend, `COLORS` palette, `save_or_show()` |
| `series.py` | 1 | `plot_series()` — multi-series overlay |
| `forecast.py` | 2 | `plot_forecast()`, `plot_residuals()` |
| `diagnostics.py` | 3 | `plot_acf_pacf()`, `plot_decomposition()`, `plot_diagnostics()` |
| `statistics.py` | 8 | Rolling stats, distribution, lag scatter, boxplot, heatmap, stationarity, validation summary, missing data |
| `comparison.py` | 3 | `plot_model_comparison()`, `plot_error_distribution()`, `plot_cumulative_error()` |
| `plots.py` | — | Facade re-exporting all 17 functions |

## All 17 Plot Functions

| Function | Signature (key args) | Description |
| --- | --- | --- |
| `plot_series` | `(*series, title, labels, save_path)` | Multi-series overlay |
| `plot_forecast` | `(y_train, y_pred, y_test?, intervals?, save_path)` | Forecast vs actuals |
| `plot_residuals` | `(residuals, save_path)` | Residual time + histogram |
| `plot_acf_pacf` | `(acf_values, pacf_values, nlags, save_path)` | ACF + PACF bars |
| `plot_decomposition` | `(decomposition: dict, save_path)` | 4-panel STL |
| `plot_diagnostics` | `(data, save_path)` | 4-panel: series, hist, Q-Q, rolling |
| `plot_rolling_statistics` | `(data, window, save_path)` | Mean±2σ + volatility |
| `plot_distribution` | `(data, bins, show_stats, save_path)` | Histogram+KDE+violin |
| `plot_lag_scatter` | `(data, lags, save_path)` | y(t) vs y(t-k) scatter |
| `plot_boxplot_by_period` | `(data, period, save_path)` | month/dayofweek/hour/quarter |
| `plot_correlation_heatmap` | `(data: DataFrame, method, save_path)` | Annotated heatmap |
| `plot_stationarity_summary` | `(data, window, save_path)` | 4-panel stationarity check |
| `plot_validation_summary` | `(data, validation_result?, freq_result?, save_path)` | Data quality dashboard |
| `plot_missing_data` | `(data, save_path)` | 4-panel gap analysis |
| `plot_model_comparison` | `(y_test, predictions: dict, metrics?, save_path)` | Multi-model overlay |
| `plot_error_distribution` | `(y_test, predictions: dict, save_path)` | Per-model error histograms |
| `plot_cumulative_error` | `(y_test, predictions: dict, save_path)` | Cumulative |error| over time |

## Color Palette

```python
COLORS = {
    "primary":   "#2196F3",   # Blue
    "secondary": "#4CAF50",   # Green
    "accent":    "#FF5722",   # Deep Orange
    "highlight": "#9C27B0",   # Purple
    "warning":   "#FF9800",   # Orange
}
```

## Common Pattern

All plot functions follow the same signature pattern:

```python
def plot_XXX(
    data: pd.Series | pd.DataFrame,
    *,
    title: str = "Default Title",
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """..."""
    check_matplotlib()  # Guard against missing matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    # ... plotting logic ...
    return save_or_show(fig, save_path)  # Auto-saves PNG + plt.close()
```

## Usage

```python
from multi_time.visualization import (
    plot_series, plot_diagnostics, plot_forecast,
    plot_rolling_statistics, plot_model_comparison,
)

# All functions accept save_path for PNG output
plot_series(data, title="My Series", save_path="output/series.png")
plot_diagnostics(data, save_path="output/diag.png")
plot_rolling_statistics(data, window=14, save_path="output/rolling.png")
plot_forecast(y_train, preds, y_test=y_test, save_path="output/forecast.png")
plot_model_comparison(y_test, {"theta": preds1, "naive": preds2},
                      metrics={"theta": {"mae": 1.2}, "naive": {"mae": 2.3}},
                      save_path="output/comparison.png")
```
