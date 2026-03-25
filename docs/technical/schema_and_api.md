# 🎯 Schema and API Design

> [!NOTE]
> Data contracts, input/output schemas, and extension points for `multi-time`. Verify parameters in [Notation and Glossary](notation.md).

## Input Data Contracts

### Series Input

All functions accepting time series data expect:

```text
pandas.Series with:
  - DatetimeIndex (monotonic preferred)
  - Numeric dtype (float64 or int64)
  - Name attribute (optional but recommended)
```

### DataFrame Input

Multivariate functions accept:

```text
pandas.DataFrame with:
  - DatetimeIndex
  - One or more numeric columns
```

### CSV Format

```csv
date,value,volume
2023-01-01,100.5,1200
2023-01-02,102.3,1350
...
```

Required: first column parseable as datetime. At least one numeric column.

## Output Schemas

### ValidationResult

```json
{
  "is_valid": true,
  "n_observations": 100,
  "n_missing": 0,
  "missing_pct": 0.0,
  "is_monotonic": true,
  "has_duplicates": false,
  "dtype": "float64",
  "errors": [],
  "warnings": []
}
```

### PatchinessResult

```json
{
  "n_gaps": 3,
  "longest_gap": 8,
  "total_missing_periods": 15,
  "patchiness_score": 0.15,
  "gap_locations": [[10, 15], [40, 42], [70, 78]]
}
```

### StatTestResult

```json
{
  "test_name": "Augmented Dickey-Fuller",
  "statistic": -4.23,
  "p_value": 0.001,
  "is_significant": true,
  "interpretation": "Stationary (ADF rejects unit root at α=0.05)",
  "details": {"used_lag": 2, "nobs": 98, "critical_values": {"1%": -3.5, "5%": -2.9, "10%": -2.6}}
}
```

### PipelineResult

```json
{
  "validation": { ... },
  "frequency": { ... },
  "patchiness": { ... },
  "descriptive_stats": { ... },
  "statistical_tests": { ... },
  "forecast_results": { ... },
  "evaluation_results": {
    "naive": {"mae": 2.1, "mse": 6.5, "mape": 0.04},
    "theta": {"mae": 1.8, "mse": 5.2, "mape": 0.03}
  },
  "pipeline_log": ["[VALIDATE] Starting...", "..."]
}
```

## Extension Points

### Adding a Forecaster

```python
# In modeling/forecasters.py
FORECASTER_REGISTRY["my_model"] = lambda **kw: MyCustomForecaster(**kw)
```

### Adding a Transformer

```python
# In transform/transformers.py
TRANSFORMER_REGISTRY["my_transform"] = lambda **kw: MyTransformer(**kw)
```

### Adding a Metric

```python
# In evaluate/metrics.py
METRIC_REGISTRY["my_metric"] = MyCustomMetric
```

### Adding a Plot Function

Add to `visualization/plots.py` and re-export in `visualization/__init__.py`.

## Configuration Override Precedence

1. **CLI argument** (highest priority)
2. **YAML config file** (via `--config`)
3. **MultiTimeConfig defaults** (lowest priority)
