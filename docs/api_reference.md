# 🎯 API Reference

> [!NOTE]
> High-level API overview mapping the facade logic of `multi-time`. Before integrating parameters like `fh` or `y`, reference the exact definitions in [Notation and Glossary](technical/notation.md).

## `multi_time.config`

### `MultiTimeConfig`

Dataclass holding all pipeline configuration parameters.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `frequency` | `str` | `"auto"` | Expected frequency (`D`, `H`, `M`, `auto`) |
| `imputation_strategy` | `str` | `"drift"` | Missing-value strategy |
| `forecast_horizon` | `int` | `12` | Steps ahead to forecast |
| `confidence_level` | `float` | `0.95` | Interval coverage (0–1) |
| `models` | `list[str]` | `["naive","exp_smoothing","auto_arima"]` | Forecaster names |
| `metrics` | `list[str]` | `["mae","mse","mape"]` | Evaluation metrics |
| `nlags_acf` | `int` | `40` | ACF/PACF lags |
| `rolling_window` | `int` | `12` | Rolling stats window |
| `significance_level` | `float` | `0.05` | Statistical test alpha |
| `transform_steps` | `list` | `[]` | Ordered transform pipeline steps |
| `seasonal_period` | `int\|None` | `None` | Seasonal period (auto if None) |
| `output_dir` | `str` | `"output"` | Default output directory |

### `MultiTimeConfig.to_dict() → dict`

Export configuration safely, handling enums and dates.

### `load_config(source) → MultiTimeConfig`

Load from YAML path or dict. Validates automatically.

### `setup_logging(level, log_file) → None`

Configure structured logging with console and optional file output.

### `get_logger(name) → Logger`

Get a namespaced logger instance.

---

## `multi_time.data`

### Loaders

- `load_csv_series(path, column, date_column, freq) → Series` — Load a single time series from CSV with date parsing
- `load_csv_dataframe(path, date_column, columns) → DataFrame` — Load a multivariate time series from CSV

### Core Generators (`generators_core.py`)

- `generate_daily_series(n, start, trend, noise_std, seed) → Series` — Daily with trend + noise
- `generate_hourly_series(n, start, daily_amplitude, baseline, noise_std, seed) → Series` — Hourly with daily seasonality
- `generate_weekly_series(n, start, trend, annual_amplitude, noise_std, seed) → Series` — Weekly with annual seasonality
- `generate_monthly_series(n, start, seasonal_amplitude, trend, noise_std, seed) → Series` — Monthly with annual seasonality

### Specialty Generators (`generators_specialty.py`)

- `generate_patchy_series(n, start, gap_ranges, seed) → Series` — Daily with deliberate NaN gaps
- `generate_irregular_series(n, year, seed) → Series` — Irregularly spaced series
- `generate_random_walk(n, start, freq, drift, volatility, initial_value, seed) → Series` — Non-stationary random walk
- `generate_multi_seasonal_series(n, start, freq, periods, amplitudes, trend, noise_std, seed) → Series` — Multiple seasonal components
- `generate_multivariate_series(n, start, freq, n_series, correlation, seed) → DataFrame` — Correlated multivariate series
- `generate_configurable_series(n, start, freq, baseline, trend, seasonal_period, seasonal_amplitude, noise_std, outlier_fraction, outlier_magnitude, gap_fraction, seed) → Series` — Fully configurable synthetic series

### Registry

- `GENERATOR_REGISTRY: dict[str, callable]` — Maps 10 generator names to factory functions
- `list_generators() → list[str]` — List all available generator names

---

## `multi_time.validate`

### Validation (`validation.py`)

- `validate_series(data) → ValidationResult` — Check types, NaN %, index monotonicity, duplicates

### Frequency Detection (`frequency.py`)

- `detect_frequency(data) → dict` — Infer frequency with fallback heuristics. Returns `inferred_freq`, `is_regular`, delta stats

### Patchiness Analysis (`patchiness.py`)

- `assess_patchiness(data, freq) → PatchinessResult` — Gap analysis: count, sizes, locations, patchiness score (0–1)

### Harmonization (`harmonize.py`)

- `harmonize_frequencies(series_list, target_freq, method) → list[Series]` — Resample multiple series to common frequency

---

## `multi_time.stats`

### Descriptive (`descriptive.py`)

- `compute_descriptive_stats(data) → dict` — Mean, std, var, skew, kurtosis, quantiles, IQR, range, CV
- `compute_acf_pacf(data, nlags, alpha) → dict` — ACF/PACF with significant lag detection
- `compute_rolling_stats(data, window) → DataFrame` — Rolling mean, std, min, max, median
- `compute_seasonal_decomposition(data, period, model) → dict` — Trend, seasonal, residual via STL
- `summarize_series(data, nlags, rolling_window) → dict` — Full summary combining all descriptive

### Statistical Tests

- `test_stationarity(data, alpha) → dict[str, StatTestResult]` — ADF + KPSS (`stationarity.py`)
- `test_normality(data, alpha) → dict[str, StatTestResult]` — Shapiro-Wilk + Jarque-Bera (`normality.py`)
- `test_seasonality(data, period, alpha) → StatTestResult` — Seasonality strength (threshold: 0.64) (`seasonality.py`)
- `test_heteroscedasticity(data, nlags, alpha) → StatTestResult` — Engle's ARCH test (`seasonality.py`)
- `test_granger_causality(data_x, data_y, maxlag, alpha) → StatTestResult` — Granger causality (`causality.py`)

### Result Dataclass (`result.py`)

- `StatTestResult` — `test_name`, `statistic`, `p_value`, `is_significant`, `alpha`, `interpretation`, `details`

---

## `multi_time.transform`

### Factory Functions

- `create_imputer(strategy)` → sktime `Imputer`
- `create_detrender()` → `Detrender`
- `create_deseasonalizer(sp, model)` → `Deseasonalizer`
- `create_box_cox()` → `BoxCoxTransformer`
- `create_differencer(lags)` → `Differencer`
- `create_lag_transformer(lags)` → `Lag`

### Pipeline

- `build_transform_pipeline(steps) → TransformerPipeline` — Chain transforms from list of names or `(name, params)` tuples
- `apply_transform(transformer, data, fit) → Series|DataFrame` — Fit and/or transform data
- `TRANSFORMER_REGISTRY: dict[str, callable]` — Maps 6 transformer names to factory functions

---

## `multi_time.modeling`

### Registry & Factory (`registry.py`)

- `create_forecaster(name, **params) → BaseForecaster` — Factory: naive, exp_smoothing, theta, poly_trend, auto_arima, sarimax
- `FORECASTER_REGISTRY: dict[str, callable]` — Maps 6 forecaster names to factory functions

### Ensemble (`ensemble.py`)

- `create_ensemble(specs, aggfunc) → EnsembleForecaster` — Ensemble from list of model specs

### Forecasting (`forecasters.py`)

- `run_forecast(forecaster, y_train, fh, X) → Series` — Fit + predict

### Evaluation & Tuning (`evaluation.py`)

- `evaluate_forecaster(forecaster, y, cv_strategy, initial_window, step_length, fh) → DataFrame` — Temporal cross-validation
- `tune_forecaster(forecaster, param_grid, y, cv_strategy, initial_window, fh) → ForecastingGridSearchCV` — Grid search

### Probabilistic (`probabilistic.py`)

- `predict_intervals(forecaster, y, fh, coverage) → DataFrame`
- `predict_quantiles(forecaster, y, fh, alpha) → DataFrame`
- `predict_variance(forecaster, y, fh) → DataFrame`
- `create_probabilistic_forecaster(name, **params) → BaseForecaster`
- `fit_distribution(data, distributions) → dict` — Fit parametric distributions with AIC/BIC ranking

---

## `multi_time.evaluate`

- `compute_metric(y_true, y_pred, metric_name, y_train) → float`
- `evaluate_forecast(y_true, y_pred, metrics_list, y_train) → dict`
- `compute_rmse(y_true, y_pred) → float`
- `list_available_metrics() → list[str]`
- `METRIC_REGISTRY: dict[str, type]` — Maps metric names to sktime classes

Available: `mae`, `mse`, `mape`, `mdape`, (optional) `mase`, `rmsse`.

---

## `multi_time.visualization`

### Series (`series.py`)

- `plot_series(*series, title, labels, figsize, save_path) → Figure` — Overlay multiple series
- `plot_multi_series_panel(series_dict, title, figsize, show_overlap, save_path) → Figure` — Stacked panels with shared x-axis
- `plot_series_correlation(series_dict, title, method, figsize, save_path) → Figure` — Cross-correlation matrix

### Forecast (`forecast.py`)

- `plot_forecast(y_train, y_pred, y_test, intervals, save_path) → Figure` — Forecast vs actuals with optional intervals
- `plot_residuals(residuals, save_path) → Figure` — Residuals over time + histogram

### Diagnostics (`diagnostics.py`)

- `plot_acf_pacf(acf_values, pacf_values, nlags, save_path) → Figure` — Side-by-side ACF/PACF
- `plot_decomposition(decomposition, save_path) → Figure` — 4-panel decomposition (observed/trend/seasonal/residual)
- `plot_diagnostics(data, save_path) → Figure` — 4-panel: series, histogram, Q-Q, rolling stats

### Statistics & Data Quality (`statistics.py`)

- `plot_rolling_statistics(data, window, save_path) → Figure` — Rolling mean ± 2σ bands + volatility
- `plot_distribution(data, bins, show_stats, save_path) → Figure` — Histogram + KDE + normal overlay + violin
- `plot_lag_scatter(data, lags, save_path) → Figure` — Scatter y(t) vs y(t-lag) with correlation
- `plot_boxplot_by_period(data, period, save_path) → Figure` — Seasonal boxplots (month/day/hour/quarter)
- `plot_correlation_heatmap(data, method, save_path) → Figure` — Annotated correlation matrix
- `plot_stationarity_summary(data, window, save_path) → Figure` — Visual stationarity check (4 panels)
- `plot_validation_summary(data, save_path) → Figure` — 4-panel data quality dashboard
- `plot_missing_data(data, save_path) → Figure` — Gap analysis (NaN segments)

### Comparison (`comparison.py`)

- `plot_model_comparison(y_test, predictions, metrics, save_path) → Figure` — Multi-model overlay + error bars
- `plot_error_distribution(y_test, predictions, save_path) → Figure` — Per-model error histograms
- `plot_cumulative_error(y_test, predictions, save_path) → Figure` — Cumulative |error| over time

---

## `multi_time.pipeline`

### `MultiTimePipeline(config)`

End-to-end pipeline: validate → describe → test → transform → forecast → evaluate.

### `pipeline.run(y_train, y_test) → PipelineResult`

Full pipeline execution with JSON export.

### `PipelineResult`

Aggregated dataclass with `.to_dict()` and `.save(path)`.
