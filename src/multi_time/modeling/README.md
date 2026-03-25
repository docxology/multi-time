# multi_time.modeling

**Forecasting + Ensemble + Tuning + Probabilistic**

## Exports

| Symbol | Type | Description |
| --- | --- | --- |
| `create_forecaster` | `(name: str) → BaseForecaster` | Factory from registry |
| `create_ensemble` | `(names: list[str]) → EnsembleForecaster` | Multi-model ensemble |
| `run_forecast` | `(forecaster, y_train, fh) → pd.Series` | Fit + predict |
| `evaluate_forecaster` | `(forecaster, data, test_size, metrics) → dict` | Eval with train/test split |
| `tune_forecaster` | `(forecaster, data, param_grid) → BaseForecaster` | Grid search tuning |
| `predict_intervals` | `(forecaster, y_train, fh, coverage) → pd.DataFrame` | Prediction intervals |
| `predict_quantiles` | `(forecaster, y_train, fh, quantiles) → pd.DataFrame` | Quantile predictions |
| `predict_variance` | `(forecaster, y_train, fh) → pd.Series` | Prediction variance |
| `create_probabilistic_forecaster` | `(name: str) → BaseForecaster` | Probabilistic variant |
| `fit_distribution` | `(residuals) → dict` | Fit residual distributions |
| `FORECASTER_REGISTRY` | `dict[str, Callable]` | Name → factory mapping |

## Forecaster Registry

```python
FORECASTER_REGISTRY = {
    "naive":             lambda: NaiveForecaster(strategy="last"),
    "seasonal_naive":    lambda: NaiveForecaster(strategy="seasonal_last", sp=12),
    "theta":             lambda: ThetaForecaster(),
    "exp_smoothing":     lambda: ExponentialSmoothing(),
    "ses":               lambda: ExponentialSmoothing(trend=None, seasonal=None),
    "auto_ets":          lambda: AutoETS(),
    "croston":           lambda: Croston(),
    "polynomial_trend":  lambda: PolynomialTrendForecaster(degree=1),
    "arima":             lambda: ARIMA(order=(1, 1, 0)),
    "auto_arima":        lambda: AutoARIMA(suppress_warnings=True),
    "bats":              lambda: BATS(use_trend=True),
    "tbats":             lambda: TBATS(use_trend=True),
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
best = tune_forecaster(f, y_train, param_grid={"sp": [7, 12, 30]})
```
