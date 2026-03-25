# multi_time.config

**Configuration + Logging**

## Exports

| Symbol | Type | Description |
| --- | --- | --- |
| `MultiTimeConfig` | `dataclass` | Central configuration for all pipeline behavior |
| `load_config` | `(path: str) → MultiTimeConfig` | Load YAML/dict config file |
| `setup_logging` | `(level: str, log_file: str) → None` | Configure structured logging |
| `get_logger` | `(name: str) → Logger` | Module-level logger factory |

## Schema: MultiTimeConfig

```python
@dataclass
class MultiTimeConfig:
    # Forecasting
    models: list[str] = field(default_factory=lambda: ["naive", "theta"])
    forecast_horizon: int = 12
    test_fraction: float = 0.2
    
    # Evaluation
    metrics: list[str] = field(default_factory=lambda: ["mae", "rmse", "mape"])
    
    # Transformations
    transformations: list[str] = field(default_factory=list)
    
    # Output
    output_dir: str = "output"
    
    # Statistical tests
    alpha: float = 0.05
```

## YAML Config Example

```yaml
models: [naive, theta, exp_smoothing]
forecast_horizon: 30
test_fraction: 0.15
metrics: [mae, rmse, mape, smape]
transformations: [impute, detrend]
output_dir: output/analysis
alpha: 0.05
```

## Usage

```python
from multi_time.config import MultiTimeConfig, load_config, setup_logging

# Programmatic
config = MultiTimeConfig(models=["theta"], forecast_horizon=30)

# YAML
config = load_config("config.yaml")

# Logging
setup_logging(level="DEBUG", log_file="pipeline.log")
```
