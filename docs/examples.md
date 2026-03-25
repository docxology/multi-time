# 🎯 Examples

> [!NOTE]
> Code snippets demonstrating end-to-end `multi-time` usage. Variables comply with definitions in [Notation and Glossary](technical/notation.md).

```python
from multi_time.data import (
    generate_daily_series, generate_hourly_series, generate_monthly_series,
    generate_configurable_series, generate_multivariate_series,
    list_generators, GENERATOR_REGISTRY,
)

# Daily with trend
data = generate_daily_series(n=365, trend=0.1, seed=42)

# Configurable: complex synthetic series
complex = generate_configurable_series(
    n=500, freq="D", trend=0.2,
    seasonal_period=7, seasonal_amplitude=5.0,
    noise_std=1.5, outlier_fraction=0.02, gap_fraction=0.05,
)

# Multivariate correlated series
multi = generate_multivariate_series(n=200, n_series=4, correlation=0.7)

# List all available generators
print(list_generators())  # ['configurable', 'daily', 'hourly', ...]
```

## 2. Validate Data

```python
from multi_time.validate import validate_series, detect_frequency, assess_patchiness

result = validate_series(data)
print(f"Valid: {result.is_valid}, Missing: {result.missing_pct:.1f}%")

freq = detect_frequency(data)
print(f"Frequency: {freq['inferred_freq']}, Regular: {freq['is_regular']}")

gaps = assess_patchiness(data)
print(f"Gaps: {gaps.n_gaps}, Max size: {gaps.max_gap_size}")

# Multi-series harmonization
from multi_time.validate import harmonize_frequencies
s1 = generate_daily_series(n=100)
s2 = generate_hourly_series(n=240)
harmonized = harmonize_frequencies([s1, s2], target_freq="D")
print(f"Harmonized lengths: {[len(h) for h in harmonized]}")
```

## 3. Descriptive Statistics

```python
from multi_time.stats import summarize_series, compute_seasonal_decomposition

summary = summarize_series(data, nlags=30, rolling_window=7)
stats = summary["descriptive"]
print(f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
print(f"Significant ACF lags: {summary['acf_pacf']['significant_acf_lags']}")
```

## 4. Statistical Testing

```python
from multi_time.stats import (
    test_stationarity, test_normality, test_seasonality,
    test_heteroscedasticity, test_granger_causality,
)

station = test_stationarity(data)
print(f"ADF: {station['adf'].interpretation}")
print(f"KPSS: {station['kpss'].interpretation}")

normal = test_normality(data)
print(f"Shapiro: {normal['shapiro'].interpretation}")

seasonal = test_seasonality(data, period=12)
print(f"Seasonality: {seasonal.interpretation}")

# ARCH test for heteroscedasticity
arch = test_heteroscedasticity(data, nlags=5)
print(f"ARCH: {arch.interpretation}")

# Granger causality (bivariate)
gc = test_granger_causality(series_x, series_y, maxlag=4)
print(f"Granger: {gc.interpretation}")
```

## 5. Transformations

```python
from multi_time.transform import build_transform_pipeline, apply_transform, TRANSFORMER_REGISTRY

# Single transform
from multi_time.transform import create_imputer
imputer = create_imputer(strategy="mean")
clean = apply_transform(imputer, patchy_data)

# Pipeline
pipeline = build_transform_pipeline([
    ("impute", {"strategy": "drift"}),
    ("deseasonalize", {"sp": 7}),
    "detrend",
])
transformed = apply_transform(pipeline, data)

# Available transforms
print(list(TRANSFORMER_REGISTRY.keys()))  # ['impute', 'detrend', ...]
```

## 6. Forecasting

```python
from multi_time.modeling import create_forecaster, run_forecast, create_ensemble, FORECASTER_REGISTRY
from multi_time.evaluate import evaluate_forecast

train, test = data.iloc[:-30], data.iloc[-30:]

# Single model
forecaster = create_forecaster("theta")
predictions = run_forecast(forecaster, train, fh=30)

# Evaluate
metrics = evaluate_forecast(test, predictions)
print(f"MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.4f}")

# Ensemble
ensemble = create_ensemble(["naive", "theta", "exp_smoothing"])
preds = run_forecast(ensemble, train, fh=30)
```

## 7. Probabilistic Forecasting

```python
from multi_time.modeling import predict_intervals, predict_quantiles, fit_distribution

# Prediction intervals
intervals = predict_intervals(forecaster, train, fh=30, coverage=0.95)

# Quantile forecasts
quantiles = predict_quantiles(forecaster, train, fh=30, alpha=[0.1, 0.5, 0.9])

# Distribution fitting
dist_results = fit_distribution(data)
print(f"Best fit: {dist_results['best_fit']}")
```

## 8. Visualization (15 functions)

```python
from multi_time.visualization import (
    plot_series, plot_forecast, plot_residuals,
    plot_diagnostics, plot_decomposition, plot_acf_pacf,
    plot_rolling_statistics, plot_distribution, plot_lag_scatter,
    plot_boxplot_by_period, plot_correlation_heatmap, plot_stationarity_summary,
    plot_model_comparison, plot_error_distribution, plot_cumulative_error,
)

# Series overlay
plot_series(train, test, labels=["Train", "Test"], save_path="output/series.png")

# Forecast with intervals
plot_forecast(train, predictions, y_test=test, intervals=intervals,
              save_path="output/forecast.png")

# Rolling statistics with ±2σ bands
plot_rolling_statistics(data, window=12, save_path="output/rolling.png")

# Distribution: histogram + KDE + normal overlay + violin
plot_distribution(data, save_path="output/distribution.png")

# Lag scatter (autocorrelation check)
plot_lag_scatter(data, lags=[1, 7, 12], save_path="output/lags.png")

# Seasonal boxplots grouped by month/day/hour
plot_boxplot_by_period(data, period="month", save_path="output/seasonal_box.png")

# Correlation heatmap (multivariate)
plot_correlation_heatmap(multi_df, method="spearman", save_path="output/corr.png")

# Stationarity visual check (4 panels)
plot_stationarity_summary(data, window=12, save_path="output/stationarity.png")

# Model comparison (overlay + error bars)
preds_dict = {"naive": naive_preds, "theta": theta_preds}
metrics_dict = {"naive": {"mae": 2.1}, "theta": {"mae": 1.3}}
plot_model_comparison(test, preds_dict, metrics=metrics_dict, save_path="output/comparison.png")

# Error distributions per model
plot_error_distribution(test, preds_dict, save_path="output/errors.png")

# Cumulative error over time
plot_cumulative_error(test, preds_dict, save_path="output/cumulative.png")
```

## 9. Full Pipeline

```python
from multi_time import MultiTimePipeline, MultiTimeConfig

config = MultiTimeConfig(
    models=["naive", "exp_smoothing", "theta"],
    forecast_horizon=30,
    metrics=["mae", "mse", "mape"],
    transform_steps=["impute", "detrend"],
    output_dir="output",
)

pipeline = MultiTimePipeline(config)
result = pipeline.run(train, test)

print(f"Validation: {result.validation['is_valid']}")
print(f"Models: {list(result.evaluation_results.keys())}")
result.save("output/full_analysis.json")
```

## 10. CLI Scripts

```bash
# Generate synthetic data
uv run python scripts/run_synthetic.py --type configurable --n 365 --trend 0.2

# Validate data
uv run python scripts/run_validation.py -i data.csv -o validation.json

# Descriptive analysis
uv run python scripts/run_descriptive.py -i data.csv -o stats.json

# Forecast with specific model
uv run python scripts/run_forecast.py -i data.csv -m theta -H 30 --test-size 30

# Full pipeline
uv run python scripts/run_pipeline.py -i data.csv -o output/

# End-to-end (generate → validate → stats → train → visualize)
uv run python scripts/run_end_to_end.py --type daily --n 365 --models naive theta --ensemble

# Run ALL scripts at once
uv run python scripts/run_all.py --output-dir output
```
