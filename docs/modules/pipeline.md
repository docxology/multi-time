# 🎯 Module: pipeline

> [!NOTE]
> The `multi_time.pipeline` subpackage provides a configurable 6-stage end-to-end orchestration pipeline for time series analysis. Review [Notation and Glossary](../technical/notation.md) to understand state propagation mapping.

## Module Map

| Module | Classes | Description |
| --- | --- | --- |
| `__init__.py` | `MultiTimePipeline`, `PipelineResult` | Pipeline orchestrator + results |

## Pipeline Stages

```text
1. VALIDATE  → validate_series, detect_frequency, assess_patchiness
2. DESCRIBE  → summarize_series (descriptive stats, ACF/PACF, rolling)
3. TEST      → test_stationarity, test_normality, test_seasonality
4. TRANSFORM → imputation, detrending, custom transform pipeline
5. FORECAST  → fit + predict for each model in config.models
6. EVALUATE  → compute metrics for each model vs. test data
```

Each stage is independently callable and logs to `PipelineResult.pipeline_log`.

## Usage

```python
from multi_time import MultiTimePipeline, MultiTimeConfig

config = MultiTimeConfig(
    models=["naive", "theta"],
    forecast_horizon=30,
    metrics=["mae", "mape"],
    output_dir="output",
)
pipeline = MultiTimePipeline(config)
result = pipeline.run(y_train, y_test)

# Access results
result.validation["is_valid"]
result.descriptive_stats["descriptive"]["mean"]
result.statistical_tests["stationarity"]["adf"]["interpretation"]
result.evaluation_results["naive"]["mae"]

# Export
result.save("output/pipeline_results.json")
```

## PipelineResult Fields

| Field | Type | Content |
| --- | --- | --- |
| `validation` | `dict` | ValidationResult data |
| `frequency` | `dict` | Frequency detection results |
| `patchiness` | `dict` | Patchiness analysis |
| `descriptive_stats` | `dict` | Full descriptive summary |
| `statistical_tests` | `dict` | Test results by category |
| `forecast_results` | `dict` | Model → predictions |
| `evaluation_results` | `dict` | Model → metrics |
| `pipeline_log` | `list[str]` | Timestamped log entries |

## Related Tests

- `tests/pipeline/test_pipeline.py` — 10 tests
