# multi_time.config

**Configuration + Logging**

## Exports

| Symbol | Type | Description |
| --- | --- | --- |
| `MultiTimeConfig` | `dataclass` | Central configuration for all pipeline behavior |
| `load_config` | `(source: str | Path | dict) → MultiTimeConfig` | Load YAML/dict config file |
| `setup_logging` | `(level: str, log_file: str) → None` | Configure structured logging |
| `get_logger` | `(name: str) → Logger` | Module-level logger factory |

See the [configuration module reference](../modules/config.md) for the dataclass fields and defaults.

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

See the [configuration guide](../configuration.md) for loader behavior and list-valued settings.
