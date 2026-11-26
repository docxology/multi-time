# multi_time.pipeline

**End-to-End Configurable Pipeline Orchestrator**

## Exports

| Symbol | Type | Description |
| --- | --- | --- |
| `MultiTimePipeline` | `class` | 6-stage configurable pipeline |
| `PipelineResult` | `dataclass` | Typed pipeline output |

## Pipeline Stages

```
validate → describe → test → transform → forecast → evaluate
```

1. **Validate** — `validate_series()` + `detect_frequency()` + `assess_patchiness()`
2. **Describe** — `summarize_series()` (descriptive stats + ACF/PACF)
3. **Test** — `test_stationarity()` + `test_normality()`
4. **Transform** — `build_transform_pipeline()` + `apply_transform()` (if configured)
5. **Forecast** — `run_forecast()` for each model in `config.models`
6. **Evaluate** — `evaluate_forecast()` against test set

The in-memory result includes `forecast_results` as a mapping from model
name to indexed predictions represented by dictionaries. See the
[serialized result schema](../technical/schema_and_api.md#pipelineresult).

## Usage

```python
from multi_time.pipeline import MultiTimePipeline, PipelineResult
from multi_time.config import MultiTimeConfig

config = MultiTimeConfig(
    models=["naive", "theta", "exp_smoothing"],
    forecast_horizon=30,
    metrics=["mae", "mse", "mape"],
    transform_steps=["impute"],
)

pipeline = MultiTimePipeline(config=config)
result: PipelineResult = pipeline.run(y_train, y_test)

print(result.pipeline_log)
print(result.evaluation_results)
```

## Configuration Driven

All pipeline behavior is controlled by `MultiTimeConfig`:

```python
config.models              # Which forecasters to run
config.forecast_horizon    # How far ahead to predict
config.metrics             # Which evaluation metrics
config.transform_steps     # Which pre-processing steps
config.significance_level  # Statistical test significance level
```
