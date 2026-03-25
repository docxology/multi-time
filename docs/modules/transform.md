# 🎯 Module: transform

> [!NOTE]
> The `multi_time.transform` subpackage wraps `sktime` transformer classes with factory functions and a configurable pipeline builder. For definitions of parameters like `lags` and `sp`, see [Notation and Glossary](../technical/notation.md).

## Module Map

| Module | Functions | Description |
| --- | --- | --- |
| `transformers.py` | 6 factories + `build_transform_pipeline` + `apply_transform` | Transform creation + pipeline |

## Transformer Registry

| Name | Factory | `sktime` Class Path | Key Parameters |
| :--- | :--- | :--- | :--- |
| `impute` | `create_imputer` | `sktime.transformations.series.impute.Imputer` | `strategy` (drift/mean/median/ffill) |
| `detrend` | `create_detrender` | `sktime.transformations.series.detrend.Detrender` | — |
| `deseasonalize` | `create_deseasonalizer` | `sktime.transformations.series.detrend.Deseasonalizer` | `sp`, `model` (add/mult) |
| `box_cox` | `create_box_cox` | `sktime.transformations.series.boxcox.BoxCoxTransformer` | — |
| `difference` | `create_differencer` | `sktime.transformations.series.difference.Differencer` | `lags` |
| `lag` | `create_lag_transformer` | `sktime.transformations.series.lag.Lag` | `lags` |

*Note: For multivariate models, `sktime.transformations.compose.ColumnConcatenator` can be used to flatten wide data frames.*

## Pipeline Builder

```python
from multi_time.transform import build_transform_pipeline, apply_transform

# From names
pipeline = build_transform_pipeline(["impute", "detrend"])

# With parameters
pipeline = build_transform_pipeline([
    ("impute", {"strategy": "mean"}),
    ("deseasonalize", {"sp": 12}),
])

result = apply_transform(pipeline, data)
```

## How It Works

1. `build_transform_pipeline` looks up each name in `TRANSFORMER_REGISTRY`
2. Calls the corresponding factory function with provided params
3. Chains all transformers into an sktime `TransformerPipeline`
4. `apply_transform` calls `.fit_transform()` or `.transform()`

## Related Tests

- `tests/transform/test_transformers.py` — 11 tests
