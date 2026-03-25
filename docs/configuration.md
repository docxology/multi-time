# 🎯 Configuration Guide

> [!NOTE]
> Detailed schema for `MultiTimeConfig` and available estimator mappings. For standard terminology, refer to [Notation and Glossary](technical/notation.md).

```python
from multi_time.config import MultiTimeConfig, load_config

# From a YAML file
config = load_config("config.yaml")

# From a dict
config = load_config({
    "forecast_horizon": 24,
    "models": ["naive", "auto_arima"],
    "frequency": "D",
})

# Defaults
config = MultiTimeConfig()
```

## Full Config Schema

```yaml
# config.yaml
frequency: "auto"              # auto, D, H, M, W, Q, Y
imputation_strategy: "drift"   # drift, mean, median, ffill, bfill, nearest
forecast_horizon: 12           # Steps ahead (≥1)
confidence_level: 0.95         # Interval coverage (0–1)
significance_level: 0.05       # Statistical test alpha

# Models to run
models:
  - naive
  - exp_smoothing
  - auto_arima

# Model-specific keyword arguments (scikit-learn parameters)
model_kwargs:
  exp_smoothing:
    trend: add
    seasonal: add
    sp: 12
  auto_arima:
    sp: 12
    suppress_warnings: true

# Evaluation metrics
metrics:
  - mae
  - mse
  - mape

# Descriptive settings
nlags_acf: 40
rolling_window: 12
seasonal_period: 12            # null for auto-detect

# Transform pipeline (applied in order)
transform_steps:
  - impute
  - detrend

# Output
output_dir: "output"
log_level: "INFO"
log_file: "pipeline.log"
```

## Available Forecasters

| Name | `sktime` Class Path | Required Package |
| --- | --- | --- |
| `naive` | `sktime.forecasting.naive.NaiveForecaster` | sktime (core) |
| `exp_smoothing` | `sktime.forecasting.exp_smoothing.ExponentialSmoothing` | sktime (core) |
| `theta` | `sktime.forecasting.theta.ThetaForecaster` | sktime (core) |
| `poly_trend` | `sktime.forecasting.trend.PolynomialTrendForecaster` | sktime (core) |
| `auto_arima` | `sktime.forecasting.arima.AutoARIMA` | pmdarima |
| `sarimax` | `sktime.forecasting.sarimax.SARIMAX` | statsmodels |

## Available Transforms

| Name | `sktime` Class Path | Key Parameters |
| --- | --- | --- |
| `impute` | `sktime.transformations.series.impute.Imputer` | `strategy` |
| `detrend` | `sktime.transformations.series.detrend.Detrender` | — |
| `deseasonalize` | `sktime.transformations.series.detrend.Deseasonalizer` | `sp`, `model` |
| `box_cox` | `sktime.transformations.series.boxcox.BoxCoxTransformer` | — |
| `difference` | `sktime.transformations.series.difference.Differencer` | `lags` |
| `lag` | `sktime.transformations.series.lag.Lag` | `lags` |

## Available Metrics

| Name | `sktime` Class Path | Requires `y_train` |
| --- | --- | --- |
| `mae` | `sktime.performance_metrics.forecasting.MeanAbsoluteError` | No |
| `mse` | `sktime.performance_metrics.forecasting.MeanSquaredError` | No |
| `mape` | `sktime.performance_metrics.forecasting.MeanAbsolutePercentageError` | No |
| `mdape` | `sktime.performance_metrics.forecasting.MedianAbsolutePercentageError` | No |
| `mase` | `sktime.performance_metrics.forecasting.MeanAbsoluteScaledError` | Yes |
| `rmsse` | `sktime.performance_metrics.forecasting.MeanSquaredScaledError` | Yes |

## Dependency Groups

```bash
# Core (sktime, statsmodels, scipy, pandas, numpy, pyyaml)
uv pip install -e .

# Forecasting extras (pmdarima, tbats)
uv pip install -e ".[forecasting]"

# Visualization (matplotlib)
uv pip install -e ".[visualization]"

# Probabilistic (skpro)
uv pip install -e ".[probabilistic]"

# Deep learning (pytorch-forecasting, torch, lightning)
uv pip install -e ".[deep-learning]"

# Everything
uv pip install -e ".[all,dev]"
```
