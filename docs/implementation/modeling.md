# multi_time.modeling

**Forecasting + Ensemble + Tuning + Probabilistic**

## Exports

| Symbol | Type | Description |
| --- | --- | --- |
| `create_forecaster` | `(name: str) → BaseForecaster` | Factory from registry |
| `create_ensemble` | `(specs: list[str | tuple]) → EnsembleForecaster` | Multi-model ensemble |
| `run_forecast` | `(forecaster, y_train, fh) → pd.Series` | Fit + predict |
| `evaluate_forecaster` | `(forecaster, y, cv_strategy, fh) → pd.DataFrame` | Rolling-origin cross-validation |
| `tune_forecaster` | `(forecaster, param_grid, y) → BaseForecaster` | Grid search tuning |
| `predict_intervals` | `(forecaster, y_train, fh, coverage) → pd.DataFrame` | Prediction intervals |
| `predict_quantiles` | `(forecaster, y_train, fh, alpha) → pd.DataFrame` | Quantile predictions |
| `predict_variance` | `(forecaster, y_train, fh) → pd.Series` | Prediction variance |
| `create_probabilistic_forecaster` | `(name: str) → BaseForecaster` | Probabilistic variant |
| `fit_distribution` | `(residuals) → dict` | Fit residual distributions |
| `FORECASTER_REGISTRY` | `dict[str, Callable]` | Name → factory mapping |

## Forecaster Registry

```python
FORECASTER_REGISTRY = {
    "naive":         _create_naive,          # NaiveForecaster(strategy="last")
    "exp_smoothing": _create_exp_smoothing,  # ExponentialSmoothing(trend="add", seasonal="add")
    "theta":         _create_theta,          # ThetaForecaster()
    "poly_trend":    _create_poly_trend,     # PolynomialTrendForecaster(degree=1)
    "auto_arima":    _create_auto_arima,     # AutoARIMA (needs pmdarima)
    "sarimax":       _create_sarimax,        # SARIMAX
}
```

## Usage

```python
from multi_time.modeling import create_forecaster, run_forecast, predict_intervals

# Single model
f = create_forecaster("theta")
preds = run_forecast(f, y_train, fh=30)

# With intervals
f2 = create_forecaster("theta")
intervals = predict_intervals(f2, y_train, fh=30, coverage=0.95)
# DataFrame with columns: ('Coverage', 0.95, 'lower'), ('Coverage', 0.95, 'upper')

# Ensemble
from multi_time.modeling import create_ensemble
ens = create_ensemble(["naive", "theta", "exp_smoothing"])
ens_preds = run_forecast(ens, y_train, fh=30)

# Tuning
from multi_time.modeling import tune_forecaster
best = tune_forecaster(f, {"sp": [7, 12, 30]}, y_train)
```
