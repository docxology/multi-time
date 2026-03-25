# 🎯 Module: modeling

> [!NOTE]
> Forecaster factory, ensemble builder, temporal evaluation, tuning, and probabilistic forecasting mapping directly to `sktime`. Consult [Notation and Glossary](../technical/notation.md) for parameter terminology (e.g. `fh`, `sp`).

## Files

| File | Purpose |
| --- | --- |
| `registry.py` | `FORECASTER_REGISTRY` (6 models), `create_forecaster()` factory |
| `ensemble.py` | `create_ensemble()` — build EnsembleForecaster from specs |
| `evaluation.py` | `evaluate_forecaster()` (temporal CV), `tune_forecaster()` (grid search) |
| `probabilistic.py` | `predict_intervals()`, `predict_quantiles()`, `predict_variance()`, `fit_distribution()` |
| `forecasters.py` | **Facade** — re-exports all + `run_forecast()` |

## Forecaster Registry

| Name | sktime Class | Required Package |
| --- | --- | --- |
| `naive` | `sktime.forecasting.naive.NaiveForecaster` | sktime (core) |
| `exp_smoothing` | `sktime.forecasting.exp_smoothing.ExponentialSmoothing` | sktime (core) |
| `theta` | `sktime.forecasting.theta.ThetaForecaster` | sktime (core) |
| `poly_trend` | `sktime.forecasting.trend.PolynomialTrendForecaster` | sktime (core) |
| `auto_arima` | `sktime.forecasting.arima.AutoARIMA` | pmdarima |
| `sarimax` | `sktime.forecasting.sarimax.SARIMAX` | statsmodels |

## Machine Learning Integration

`sktime` supports using standard `scikit-learn` regressors (e.g., `RandomForestRegressor`) as time series forecasters via reductions.

```python
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.sk_wrapper import make_reduction

# Convert any scikit-learn regressor to a forecaster
regressor = RandomForestRegressor(n_estimators=100)
ml_forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
```

## Key Interfaces

### `run_forecast(forecaster, y_train, fh, X) → Series`

Fit a forecaster and generate predictions in one call.

### `evaluate_forecaster(forecaster, y, cv_strategy, ...) → DataFrame`

Temporal cross-validation with expanding or sliding window splitters.

### `tune_forecaster(forecaster, param_grid, y, ...) → ForecastingGridSearchCV`

Hyperparameter tuning via grid search over expanding windows.

## Tests

`tests/modeling/test_forecasters.py` — Tests for factory, ensemble, forecast, and CV evaluation.

`tests/modeling/test_probabilistic.py` — Tests for prediction intervals and distribution fitting.
