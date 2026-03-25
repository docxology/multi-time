# multi_time.evaluate

**Forecast Evaluation Metrics**

## Exports

| Symbol | Type | Description |
| --- | --- | --- |
| `compute_metric` | `(y_true, y_pred, metric: str) → float` | Single metric |
| `evaluate_forecast` | `(y_true, y_pred, metrics_list) → dict[str, float]` | Multi-metric evaluation |
| `compute_rmse` | `(y_true, y_pred) → float` | Root Mean Squared Error |
| `list_available_metrics` | `() → list[str]` | List registered metric names |
| `METRIC_REGISTRY` | `dict[str, Callable]` | Name → sktime metric mapping |

## Metric Registry

```python
METRIC_REGISTRY = {
    "mae":   MeanAbsoluteError(),              # Mean Absolute Error
    "rmse":  MeanSquaredError(square_root=True), # Root Mean Squared Error
    "mse":   MeanSquaredError(),                # Mean Squared Error
    "mape":  MeanAbsolutePercentageError(),     # Mean Absolute Percentage Error
    "smape": MeanAbsolutePercentageError(symmetric=True),  # Symmetric MAPE
    "mase":  MeanAbsoluteScaledError(),         # Mean Absolute Scaled Error
}
```

## Output Schema

```python
evaluate_forecast(y_true, y_pred, metrics_list=["mae", "rmse", "mape"]) → {
    "mae": 1.234,
    "rmse": 1.567,
    "mape": 0.034,
}
```

## Usage

```python
from multi_time.evaluate import evaluate_forecast, compute_metric

# Multi-metric
metrics = evaluate_forecast(y_test, predictions, metrics_list=["mae", "rmse", "mape"])
print(f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

# Single metric
mae = compute_metric(y_test, predictions, "mae")
```
