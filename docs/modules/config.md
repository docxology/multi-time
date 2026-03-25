# 🎯 Module: config

> [!NOTE]
> The `multi_time.config` subpackage provides centralized configuration and structured logging for the entire toolkit. For variable naming standards, refer to [Notation and Glossary](../technical/notation.md).

## Module Map

| Module | Functions/Classes | Description |
| --- | --- | --- |
| `settings.py` | `MultiTimeConfig`, `load_config` | Dataclass config + YAML/dict loading |
| `logging.py` | `setup_logging`, `get_logger` | Structured console + file logging |

## MultiTimeConfig Fields

| Field | Type | Default | Validation |
| --- | --- | --- | --- |
| `frequency` | `str` | `"auto"` | — |
| `imputation_strategy` | `str` | `"drift"` | Must be: drift, mean, median, ffill, bfill, nearest |
| `forecast_horizon` | `int` | `12` | Must be ≥ 1 |
| `confidence_level` | `float` | `0.95` | Must be in (0, 1) |
| `significance_level` | `float` | `0.05` | Must be in (0, 1) |
| `models` | `list[str]` | `["naive","exp_smoothing","auto_arima"]` | — |
| `metrics` | `list[str]` | `["mae","mse","mape"]` | — |
| `nlags_acf` | `int` | `40` | Must be ≥ 1 |
| `rolling_window` | `int` | `12` | Must be ≥ 2 |
| `output_dir` | `str` | `"output"` | — |
| `transform_steps` | `list` | `[]` | — |
| `seasonal_period` | `int\|None` | `None` | Auto-detected if None |
| `log_level` | `str` | `"INFO"` | — |
| `log_file` | `str\|None` | `None` | — |

## Usage

```python
from multi_time.config import MultiTimeConfig, load_config

# From YAML
config = load_config("config.yaml")

# From dict
config = load_config({"forecast_horizon": 24, "models": ["theta"]})

# Defaults
config = MultiTimeConfig()
errors = config.validate()  # Returns [] if valid
```

## Related Tests

- `tests/config/test_config.py` — 11 tests covering defaults, validation, YAML loading, error cases
