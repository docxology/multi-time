# multi_time

Root Python package for multi-frequency time series analysis. v0.3.0.

## Public API — 55+ Exported Symbols

### Configuration

```python
from multi_time import MultiTimeConfig, load_config, setup_logging, get_logger

config = MultiTimeConfig(models=["theta", "naive"], forecast_horizon=30)
config = load_config("config.yaml")  # YAML override
setup_logging(level="DEBUG", log_file="run.log")
```

### Data I/O + 10 Generators

```python
from multi_time import load_csv_series, GENERATOR_REGISTRY, list_generators

data = load_csv_series("data.csv", column="price")  # pd.Series with DatetimeIndex

# Generators — all return pd.Series or pd.DataFrame
from multi_time import (
    generate_daily_series,        # Daily with trend + noise
    generate_hourly_series,       # Hourly with daily cycle
    generate_weekly_series,       # Weekly with annual seasonality
    generate_monthly_series,      # Monthly with seasonal amplitude
    generate_patchy_series,       # Random missing values
    generate_irregular_series,    # Non-uniform timestamps
    generate_random_walk,         # Brownian motion + drift
    generate_multi_seasonal_series,   # Multiple seasonal periods
    generate_multivariate_series,     # Correlated multi-column DataFrame
    generate_configurable_series,     # Full control: trend+season+noise+outliers+gaps
)

# Registry access
print(list_generators())  # ['daily', 'hourly', ..., 'configurable']
data = GENERATOR_REGISTRY["configurable"](n=365, trend=0.2, seasonal_period=7)
```

### Validation

```python
from multi_time import validate_series, detect_frequency, assess_patchiness

result: ValidationResult = validate_series(data)
# result.is_valid, result.n_missing, result.missing_pct, result.is_monotonic, ...

freq_info: dict = detect_frequency(data)
# {'inferred_freq': 'D', 'is_regular': True, 'median_delta': Timedelta('1 days')}

patchiness: PatchinessResult = assess_patchiness(data)
# patchiness.n_gaps, patchiness.mean_gap_size, patchiness.max_gap_size, ...
```

### Statistics

```python
from multi_time import (
    summarize_series,              # All-in-one: descriptive + ACF/PACF + rolling
    test_stationarity,             # ADF + KPSS → dict[str, StatTestResult]
    test_normality,                # Shapiro-Wilk + Jarque-Bera
    test_seasonality,              # Autocorrelation-based
    test_heteroscedasticity,       # ARCH/LM
    test_granger_causality,        # Bivariate Granger
    StatTestResult,                # Typed result dataclass
)

summary = summarize_series(data, nlags=40, rolling_window=12)
# {'descriptive': {...}, 'acf_pacf': {...}, 'rolling': {...}}

adf: StatTestResult = test_stationarity(data)["adf"]
# adf.statistic, adf.p_value, adf.is_significant, adf.interpretation
```

### Transformations

```python
from multi_time import build_transform_pipeline, apply_transform, TRANSFORMER_REGISTRY

pipeline = build_transform_pipeline(["impute", "detrend", "deseasonalize"])
clean_data = apply_transform(pipeline, data)

# Available: impute, detrend, deseasonalize, box_cox, difference, lag
```

### Forecasting

```python
from multi_time import (
    create_forecaster, create_ensemble, run_forecast,
    predict_intervals, predict_quantiles, FORECASTER_REGISTRY,
)

f = create_forecaster("theta")
preds = run_forecast(f, y_train, fh=30)      # pd.Series
intervals = predict_intervals(f, y_train, fh=30, coverage=0.95)  # pd.DataFrame

ens = create_ensemble(["naive", "theta", "exp_smoothing"])
# Available: naive, theta, exp_smoothing, ses, auto_ets, croston,
#            polynomial_trend, arima, auto_arima, bats, tbats
```

### Evaluation

```python
from multi_time import evaluate_forecast, METRIC_REGISTRY

metrics = evaluate_forecast(y_test, preds, metrics_list=["mae", "rmse", "mape"])
# {'mae': 1.23, 'rmse': 1.56, 'mape': 0.034}
```

### Pipeline

```python
from multi_time import MultiTimePipeline, MultiTimeConfig

pipeline = MultiTimePipeline(config=MultiTimeConfig(models=["theta"]))
result: PipelineResult = pipeline.run(y_train, y_test)
# result.validation, result.descriptive_stats, result.evaluation_results, ...
```

## Subpackages

| Package | Files | Exports | Purpose |
| --- | --- | --- | --- |
| `config/` | 2 | 4 | Configuration + structured logging |
| `data/` | 4 | 14 | CSV loaders + 10 generators + registry |
| `validate/` | 5 | 6 | Validation, frequency, patchiness, harmonization |
| `stats/` | 7 | 11 | Descriptive + 5 statistical tests |
| `transform/` | 1 | 9 | 6 transformer factories + pipeline builder |
| `modeling/` | 5 | 11 | Forecasting + probabilistic + tuning |
| `evaluate/` | 1 | 5 | Metric registry + evaluation |
| `visualization/` | 7 | 17 | 17 plot functions in 5 sub-modules |
| `pipeline/` | 1 | 2 | End-to-end orchestrator |
