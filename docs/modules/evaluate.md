# 🎯 Module: evaluate

> [!NOTE]
> The `multi_time.evaluate` subpackage provides a registry-based interface for computing `sktime.performance_metrics` evaluations. Review [Notation and Glossary](../technical/notation.md) to understand dependencies like `y_train`.

## Module Map

| Module | Functions | Description |
| --- | --- | --- |
| `metrics.py` | `compute_metric`, `evaluate_forecast`, `compute_rmse`, `list_available_metrics` | Metric computation |

## Available Metrics

| Name | `sktime` Metric Class | Formula | Requires `y_train` |
| --- | --- | --- | --- |
| `mae` | `MeanAbsoluteError` | Σ\|y - ŷ\| / n | No |
| `mse` | `MeanSquaredError` | Σ(y - ŷ)² / n | No |
| `mape` | `MeanAbsolutePercentageError` | Σ\|y - ŷ\|/\|y\| / n | No |
| `mdape` | `MedianAbsolutePercentageError` | median(\|y - ŷ\|/\|y\|) | No |
| `mase` | `MeanAbsoluteScaledError` | MAE / in-sample MAE | Yes |
| `rmsse` | `MeanSquaredScaledError` | √(MSE / in-sample MSE) | Yes |

*All classes are imported from `sktime.performance_metrics.forecasting`.*

## Usage

```python
from multi_time.evaluate import evaluate_forecast, compute_rmse

# Multiple metrics
metrics = evaluate_forecast(y_true, y_pred, metrics_list=["mae", "mse", "mape"])

# Single metric
rmse = compute_rmse(y_true, y_pred)

# All available
print(list_available_metrics())
```

## Notes

- **Scaled metrics** (`mase`, `rmsse`) require the training series via `y_train` parameter
- Failed metrics return `NaN` rather than raising (logged at WARNING level)
- The registry (`METRIC_REGISTRY`) maps string names to sktime metric classes

## Related Tests

- `tests/evaluate/test_metrics.py` — 10 tests
