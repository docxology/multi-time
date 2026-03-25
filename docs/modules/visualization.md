# 🎯 Module: visualization

> [!NOTE]
> 19 `matplotlib`-based plot functions across 5 sub-modules. See [Notation and Glossary](../technical/notation.md) for plotting variable meanings (e.g. `lags`, `alpha`).

## Files

| File | Purpose |
| --- | --- |
| `core.py` | Shared utilities: `COLORS`, `check_matplotlib()`, `save_or_show()`, Agg backend |
| `series.py` | `plot_series()`, `plot_multi_series_panel()`, `plot_series_correlation()` |
| `forecast.py` | `plot_forecast()`, `plot_residuals()` — forecast vs actuals |
| `diagnostics.py` | `plot_acf_pacf()`, `plot_decomposition()`, `plot_diagnostics()` — analytical panels |
| `statistics.py` | `plot_rolling_statistics()`, `plot_distribution()`, `plot_lag_scatter()`, `plot_boxplot_by_period()`, `plot_correlation_heatmap()`, `plot_stationarity_summary()`, `plot_validation_summary()`, `plot_missing_data()` |
| `comparison.py` | `plot_model_comparison()`, `plot_error_distribution()`, `plot_cumulative_error()` |
| `plots.py` | **Facade** — re-exports all 19 functions |

## Function Reference

### Series

- `plot_series(*series, title, labels, figsize, save_path)` — Overlay multiple time series
- `plot_multi_series_panel(series_dict, title, figsize, show_overlap, save_path)` — Stacked panels with shared x-axis
- `plot_series_correlation(series_dict, title, method, figsize, save_path)` — Cross-correlation matrix

### Forecast

- `plot_forecast(y_train, y_pred, y_test, intervals, save_path)` — Forecast vs actuals with prediction intervals
- `plot_residuals(residuals, save_path)` — Residuals over time + histogram

### Diagnostics

- `plot_acf_pacf(acf_values, pacf_values, nlags, save_path)` — Side-by-side ACF/PACF bars
- `plot_decomposition(decomposition, save_path)` — 4-panel: observed, trend, seasonal, residual
- `plot_diagnostics(data, save_path)` — 4-panel: series, histogram, Q-Q plot, rolling stats

### Statistics

- `plot_rolling_statistics(data, window, save_path)` — Rolling mean ± 2σ confidence bands + volatility
- `plot_distribution(data, bins, show_stats, save_path)` — Histogram + KDE + normal overlay + violin + stats annotation
- `plot_lag_scatter(data, lags, save_path)` — Scatter y(t) vs y(t-lag) with Pearson r
- `plot_boxplot_by_period(data, period, save_path)` — Seasonal boxplots grouped by month, dayofweek, hour, or quarter
- `plot_correlation_heatmap(data, method, save_path)` — Annotated Pearson/Spearman/Kendall correlation matrix
- `plot_stationarity_summary(data, window, save_path)` — 4-panel stationarity visual check
- `plot_validation_summary(data, save_path)` — 4-panel data quality dashboard
- `plot_missing_data(data, save_path)` — Gap analysis (NaN segments)

### Comparison

- `plot_model_comparison(y_test, predictions, metrics, save_path)` — Multi-model forecast overlay + horizontal error bar chart
- `plot_error_distribution(y_test, predictions, save_path)` — Per-model error histograms with mean/std
- `plot_cumulative_error(y_test, predictions, save_path)` — Cumulative |error| over time for model selection

## Tests

`tests/visualization/test_visualization.py` — 27 tests covering all 19 plot functions.
