# multi_time.transform

**sktime Transformer Wrappers + Pipeline Builder**

## Exports

| Symbol | Type | Description |
| --- | --- | --- |
| `create_imputer` | `(**kwargs) → Imputer` | Missing value imputer (method: mean/ffill/bfill) |
| `create_detrender` | `(degree?) → Detrender` | Polynomial trend removal |
| `create_deseasonalizer` | `(sp?, model?) → Deseasonalizer` | Seasonal component removal |
| `create_box_cox` | `() → BoxCoxTransformer` | Box-Cox power transform |
| `create_differencer` | `(lags?) → Differencer` | d-th order differencing |
| `create_lag_transformer` | `(lags?) → WindowSummarizer` | Lag feature extraction |
| `build_transform_pipeline` | `(list[str]) → TransformPipeline` | Compose chain from names |
| `apply_transform` | `(pipeline, data) → pd.Series` | Apply pipeline to data |
| `TRANSFORMER_REGISTRY` | `dict[str, Callable]` | Name → factory mapping |

## Registry

```python
TRANSFORMER_REGISTRY = {
    "impute":        create_imputer,
    "detrend":       create_detrender,
    "deseasonalize": create_deseasonalizer,
    "box_cox":       create_box_cox,
    "difference":    create_differencer,
    "lag":           create_lag_transformer,
}
```

## Pipeline Builder

```python
from multi_time.transform import build_transform_pipeline, apply_transform

# Chain: impute → detrend → deseasonalize
pipeline = build_transform_pipeline(["impute", "detrend", "deseasonalize"])
clean = apply_transform(pipeline, data)

# With custom params
from multi_time.transform import create_imputer, create_differencer
from sktime.transformations.compose import TransformerPipeline
pipeline = TransformerPipeline([create_imputer(method="ffill"), create_differencer(lags=[1])])
```

## Underlying sktime Classes

| Factory | sktime Class |
| --- | --- |
| `create_imputer` | `sktime.transformations.series.impute.Imputer` |
| `create_detrender` | `sktime.transformations.series.detrend.Detrender` |
| `create_deseasonalizer` | `sktime.transformations.series.detrend.Deseasonalizer` |
| `create_box_cox` | `sktime.transformations.series.boxcox.BoxCoxTransformer` |
| `create_differencer` | `sktime.transformations.series.difference.Differencer` |
| `create_lag_transformer` | `sktime.transformations.series.summarize.WindowSummarizer` |
