# Guide: Visualization & Analysis

## Overview

`multi_time.visualization` provides 19 plot functions across 6 categories. All accept `save_path` for file output and return `matplotlib.Figure`.

## Series Plots

```python
from multi_time.visualization import plot_series

# Overlay multiple series
plot_series(train, test, labels=["Train", "Test"], save_path="output/series.png")

# Multi-series stacked panels with overlap highlighting
from multi_time.visualization import plot_multi_series_panel, plot_series_correlation
series_dict = {"Series A": train, "Series B": test}
plot_multi_series_panel(series_dict, title="Dataset Comparison", save_path="output/panel.png")

# Cross-correlation matrix between series
plot_series_correlation(series_dict, method="pearson", save_path="output/cross_corr.png")
```

## Forecast Plots

```python
from multi_time.visualization import plot_forecast, plot_residuals

# Forecast vs actuals with prediction intervals
plot_forecast(y_train, predictions, y_test=test, intervals=intervals,
              save_path="output/forecast.png")

# Residual analysis
residuals = test - predictions
plot_residuals(residuals, save_path="output/residuals.png")
```

## Diagnostics

```python
from multi_time.visualization import plot_diagnostics, plot_acf_pacf, plot_decomposition

# 4-panel diagnostic: series, histogram, Q-Q, rolling stats
plot_diagnostics(data, save_path="output/diagnostics.png")

# Data quality dashboards and missing data
from multi_time.visualization import plot_validation_summary, plot_missing_data
plot_validation_summary(data, save_path="output/validation_summary.png")
plot_missing_data(data, save_path="output/missing_data.png")

# ACF/PACF from pre-computed values
from multi_time.stats import summarize_series
summary = summarize_series(data, nlags=30)
acf_data = summary["acf_pacf"]
plot_acf_pacf(acf_data["acf_values"], acf_data["pacf_values"],
              nlags=acf_data["nlags"], save_path="output/acf_pacf.png")

# Seasonal decomposition
from multi_time.stats import compute_seasonal_decomposition
decomp = compute_seasonal_decomposition(data, period=7)
plot_decomposition(decomp, save_path="output/decomposition.png")
```

## Statistical Visualizations

```python
from multi_time.visualization import (
    plot_rolling_statistics, plot_distribution, plot_lag_scatter,
    plot_boxplot_by_period, plot_correlation_heatmap, plot_stationarity_summary,
)

# Rolling mean ± 2σ confidence bands + volatility & range
plot_rolling_statistics(data, window=12, save_path="output/rolling.png")

# Distribution: histogram + KDE + normal overlay + violin + descriptive stats
plot_distribution(data, bins=40, save_path="output/distribution.png")

# Lag scatter plots with correlation coefficients
plot_lag_scatter(data, lags=[1, 7, 12], save_path="output/lags.png")

# Seasonal boxplots by month, day of week, hour, or quarter
plot_boxplot_by_period(data, period="month", save_path="output/seasonal_box.png")
plot_boxplot_by_period(hourly_data, period="hour", save_path="output/hourly_box.png")

# Correlation heatmap (requires DataFrame input)
import pandas as pd
df = pd.DataFrame({"A": series_a, "B": series_b, "C": series_c})
plot_correlation_heatmap(df, method="spearman", save_path="output/corr.png")

# Stationarity visual check: series+rolling mean, rolling std, first diff, diff histogram
plot_stationarity_summary(data, window=12, save_path="output/stationarity.png")
```

## Model Comparison

```python
from multi_time.visualization import (
    plot_model_comparison, plot_error_distribution, plot_cumulative_error,
)

predictions = {"naive": naive_preds, "theta": theta_preds, "ets": ets_preds}
metrics = {"naive": {"mae": 2.1}, "theta": {"mae": 1.3}, "ets": {"mae": 1.5}}

# Multi-model overlay (actual vs forecasts) + horizontal MAE bar chart
plot_model_comparison(y_test, predictions, metrics=metrics,
                      save_path="output/comparison.png")

# Per-model error histograms with mean/std annotation
plot_error_distribution(y_test, predictions, save_path="output/errors.png")

# Cumulative absolute error over time (which model drifts least?)
plot_cumulative_error(y_test, predictions, save_path="output/cumulative.png")
```

## Integration with Scripts

The `run_end_to_end.py` script automatically generates all applicable plots:

```bash
uv run python scripts/run_end_to_end.py --type daily --n 365 \
    --models naive theta exp_smoothing -o output/e2e
# Generates: series.png, diagnostics.png, acf_pacf.png,
#            decomposition.png, forecast.png, residuals.png
```

Use `--no-plots` to skip visualization in CI or headless environments.
