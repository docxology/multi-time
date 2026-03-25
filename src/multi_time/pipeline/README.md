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

## Schema: PipelineResult

```python
@dataclass
class PipelineResult:
    validation: dict                         # validate_series().to_dict()
    descriptive_stats: dict                  # summarize_series() output
    statistical_tests: dict                  # {test_name: StatTestResult.to_dict()}
    predictions: dict[str, pd.Series]        # {model_name: predicted_series}
    evaluation_results: dict[str, dict]      # {model_name: {metric: value}}
    pipeline_log: list[str]                  # Stage completion messages

    def to_dict(self) -> dict[str, Any]: ...
```

## Usage

```python
from multi_time.pipeline import MultiTimePipeline
from multi_time.config import MultiTimeConfig

config = MultiTimeConfig(
    models=["naive", "theta", "exp_smoothing"],
    forecast_horizon=30,
    metrics=["mae", "rmse", "mape"],
    transformations=["impute"],
)

pipeline = MultiTimePipeline(config=config)
result: PipelineResult = pipeline.run(y_train, y_test)

print(result.pipeline_log)           # ['validate: OK', 'describe: OK', ...]
print(result.evaluation_results)     # {'theta': {'mae': 1.2, ...}}
```

## Configuration Driven

All pipeline behavior is controlled by `MultiTimeConfig`:

```python
config.models              # Which forecasters to run
config.forecast_horizon    # How far ahead to predict
config.metrics             # Which evaluation metrics
config.transformations     # Which pre-processing steps
config.alpha               # Statistical test significance level
```
