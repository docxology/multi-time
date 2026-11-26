# 🎯 Configuration Guide

> [!NOTE]
> Detailed schema for `MultiTimeConfig` and available estimator mappings. For standard terminology, refer to [Notation and Glossary](technical/notation.md).

Use the Python constructor to configure a forecast pipeline.

```python
from multi_time.config import MultiTimeConfig

config = MultiTimeConfig(
    forecast_horizon=24,
    models=["naive", "theta"],
    metrics=["mae", "mse", "mape"],
    transform_steps=["impute"],
)
```

`load_config` accepts a YAML path or dictionary and validates scalar settings.
The current loader skips `models`, `metrics`, and `transform_steps` because
they use dataclass default factories. Set those fields through the Python
constructor. Pass estimator-specific parameters to `create_forecaster()`.

```yaml
# Example scalar settings for load_config("config.yaml")
forecast_horizon: 24
imputation_strategy: "drift"
nlags_acf: 24
rolling_window: 12
significance_level: 0.05
seasonal_period: 12
```

Individual scripts choose which settings to consume. Check the
[script's flags](guides/scripts_reference.md) before combining CLI and YAML
settings.

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

## Optional dependencies

```bash
uv sync --locked
uv sync --locked --extra forecasting
uv sync --locked --extra visualization
uv sync --locked --extra probabilistic
uv sync --locked --extra all --extra dev
```

The `deep-learning` extra installs additional dependencies. The current
forecaster registry has no deep learning entry, and `all` does not include
that extra. The plotting extra enables matplotlib-based visualization.
