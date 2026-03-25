# 🎯 Guide: Training Models (Simple → Complex)

> [!NOTE]
> How to train time series forecasters from simple baselines through complex models with `multi-time` and `sktime`. Ensure fluency with [Notation and Glossary](../technical/notation.md) before building advanced ML reductions.

## Step 1: Start with Baselines

Always establish a baseline before trying complex models:

```python
from multi_time.data import generate_daily_series
from multi_time.modeling import create_forecaster, run_forecast
from multi_time.evaluate import evaluate_forecast

data = generate_daily_series(n=365, trend=0.1)
train, test = data.iloc[:-30], data.iloc[-30:]

# Naive baseline (last value repeated)
naive = create_forecaster("naive", strategy="last")
preds = run_forecast(naive, train, fh=30)
baseline = evaluate_forecast(test, preds)
print(f"Baseline MAE: {baseline['mae']:.2f}")
```

## Step 2: Simple Statistical Models

```python
# Theta method (strong on competition benchmarks)
theta = create_forecaster("theta")
preds = run_forecast(theta, train, fh=30)

# Exponential Smoothing (with trend)
ets = create_forecaster("exp_smoothing", trend="add", sp=1, seasonal=None)
preds = run_forecast(ets, train, fh=30)

# Polynomial trend
poly = create_forecaster("poly_trend", degree=2)
preds = run_forecast(poly, train, fh=30)
```

## Step 3: ARIMA Family

```python
# SARIMAX (manual order selection)
sarimax = create_forecaster("sarimax")
preds = run_forecast(sarimax, train, fh=30)

# AutoARIMA (automatic p, d, q selection)
# Requires: uv pip install -e ".[forecasting]"
auto = create_forecaster("auto_arima", sp=7)  # Weekly seasonality
preds = run_forecast(auto, train, fh=30)
```

## Step 4: Machine Learning Reductions (scikit-learn)

You can wrap standard scikit-learn regressors into explicit forecasters via `sktime.forecasting.sk_wrapper.make_reduction`:

```python
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.sk_wrapper import make_reduction

# Create a reduction forecaster
regressor = RandomForestRegressor(n_estimators=100)
ml_forecaster = make_reduction(
    regressor,
    window_length=15,    # Number of past lags to use as features
    strategy="recursive" # Strategy for multi-step forecasting
)

preds = run_forecast(ml_forecaster, train, fh=30)
```

## Step 5: Compare Models

### Model Configuration Schemas

When building dynamic pipelines, you can specify configurations via `MultiTimeConfig`.

```json
{
  "models": ["naive", "theta", "exp_smoothing"],
  "model_kwargs": {
    "exp_smoothing": {"trend": "add", "seasonal": "add", "sp": 12},
    "auto_arima": {"sp": 12, "suppress_warnings": true}
  }
}
```

```python
models = ["naive", "theta", "exp_smoothing", "poly_trend"]
results = {}
for name in models:
    f = create_forecaster(name)
    p = run_forecast(f, train, fh=30)
    results[name] = evaluate_forecast(test, p)
    print(f"{name:20s}: MAE={results[name]['mae']:.2f}, MAPE={results[name]['mape']:.4f}")
```

## Step 6: Ensemble

Combine multiple `sktime` models for robustness using `EnsembleForecaster`:

```python
from multi_time.modeling import create_ensemble

ensemble = create_ensemble(["naive", "theta", "exp_smoothing"])
preds = run_forecast(ensemble, train, fh=30)
metrics = evaluate_forecast(test, preds)
print(f"Ensemble MAE: {metrics['mae']:.2f}")
```

## Step 7: Cross-Validation

Proper temporal evaluation with expanding window:

```python
from multi_time.modeling import evaluate_forecaster

f = create_forecaster("theta")
cv_results = evaluate_forecaster(f, data, initial_window=100, fh=1)
print(f"Mean CV MAE: {cv_results['test_MeanAbsoluteError'].mean():.2f}")
```

## Step 8: Probabilistic Forecasting

```python
from multi_time.modeling import predict_intervals, predict_quantiles, fit_distribution

# Prediction intervals
f = create_forecaster("theta")
intervals = predict_intervals(f, train, fh=30, coverage=[0.80, 0.95])

# Quantile forecasts
quantiles = predict_quantiles(f, train, fh=30, alpha=[0.1, 0.25, 0.5, 0.75, 0.9])

# Distribution fitting
dist = fit_distribution(data)
print(f"Best fit: {dist['best_fit']}")
```

## Step 9: Full Pipeline

```python
from multi_time import MultiTimePipeline, MultiTimeConfig

config = MultiTimeConfig(
    models=["naive", "theta", "exp_smoothing"],
    forecast_horizon=30,
    metrics=["mae", "mse", "mape"],
    transform_steps=["impute"],
    output_dir="output",
)
pipeline = MultiTimePipeline(config)
result = pipeline.run(train, test)
result.save("output/full_results.json")
```

## CLI

```bash
# Single model
uv run python scripts/run_forecast.py -i data.csv -m theta -H 30 --test-size 30

# Full pipeline
uv run python scripts/run_pipeline.py -i data.csv -c config.yaml -o output/
```
