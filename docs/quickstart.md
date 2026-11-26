# Quickstart

Install Git, uv, and Python 3.11 or newer. The locked setup below uses
Python 3.14 and the development extra, which includes pytest and matplotlib.

```bash
uv sync --locked --python 3.14 --extra dev
uv run --no-sync --python 3.14 pytest -q
```

## Run a chronological holdout

```python
from multi_time import MultiTimeConfig, MultiTimePipeline
from multi_time.data import generate_daily_series

series = generate_daily_series(n=90, seed=42)
train, test = series.iloc[:-12], series.iloc[-12:]
config = MultiTimeConfig(models=["naive"], forecast_horizon=len(test))
result = MultiTimePipeline(config).run(train, test)
print(result.evaluation_results["naive"])
```

The Python result carries validation, frequency and gap diagnostics, test
results, forecasts, and metrics. `result.save(path)` writes the summary
and metrics to JSON. See the [API reference](api_reference.md) for the
forecast results kept in memory.

The [scripts reference](guides/scripts_reference.md) covers the CLI flags
and where each script writes its report. For a complete synthetic example
that writes a CSV, JSON report, and forecast plot, follow the
[manuscript render instructions](rendering.md).
