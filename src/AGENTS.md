# Agents — src/

## Overview

Source root for the `multi-time` package. All business logic lives in `src/multi_time/`. The core logic explicitly wraps and integrates state-of-the-art methodology from `sktime` (e.g. `AutoARIMA`, `EnsembleForecaster`, Machine Learning Reductions via `make_reduction`, and exhaustive evaluation metrics).

## Conventions

- All source code lives under `src/multi_time/`
- Each subpackage has an `__init__.py` exporting public API (facade pattern)
- All functions are fully typed with Python 3.10+ annotations (`from __future__ import annotations`)
- Every module uses `logging.getLogger(__name__)` for structured logging
- Optional dependencies (matplotlib, pmdarima, skpro, statsmodels) handled dynamically and lazily where appropriate
- Dataclass result objects for typed schemas: `ValidationResult`, `PatchinessResult`, `StatTestResult`, `PipelineResult`
- Registry pattern for extensible components: `GENERATOR_REGISTRY`, `FORECASTER_REGISTRY` (sktime models + `make_reduction` wrappers), `TRANSFORMER_REGISTRY`, `METRIC_REGISTRY` (`sktime.performance_metrics.forecasting`)

## Key Patterns

### Registry Pattern

```python
FORECASTER_REGISTRY: dict[str, Callable] = {
    "naive": lambda: NaiveForecaster(strategy="last"),
    "theta": lambda: ThetaForecaster(),
    ...
}
# Usage: create_forecaster("theta")
```

### Facade Pattern

```python
# stats/tests.py re-exports from 4 sub-modules:
from multi_time.stats.stationarity import test_stationarity
from multi_time.stats.normality import test_normality
# Users import from: multi_time.stats
```

### Typed Dataclass Results

```python
@dataclass
class StatTestResult:
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    interpretation: str
    details: dict[str, Any]
```
