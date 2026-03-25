# Agents — multi_time/

## Overview

Root package for the multi-time toolkit. 9 subpackages, 40+ Python files, 55+ public symbols.

## Conventions

- Each subpackage is self-contained with `__init__.py` re-exporting public API (facade pattern)
- Registry pattern for extensible generators, forecasters, transformers, metrics
- All functions typed with `from __future__ import annotations`
- `logging.getLogger(__name__)` in every module
- Optional deps guarded: `try: import matplotlib except ImportError: ...`
- `MultiTimeConfig` dataclass drives all configurable behavior

## Architecture

```
__init__.py          50+ re-exports from all 9 subpackages
config/              MultiTimeConfig + YAML loader + logging
data/                GENERATOR_REGISTRY (10 entries) + CSV I/O
validate/            ValidationResult + PatchinessResult dataclasses
stats/               StatTestResult dataclass + 5 test categories
transform/           TRANSFORMER_REGISTRY (6 entries) + pipeline builder
modeling/            FORECASTER_REGISTRY (11+ entries) + probabilistic
evaluate/            METRIC_REGISTRY (6+ entries) + evaluate_forecast
visualization/       17 plot functions + Agg backend + COLORS palette
pipeline/            MultiTimePipeline + PipelineResult
```

## Design Patterns

### 1. Registry Pattern

```python
GENERATOR_REGISTRY: dict[str, Callable[..., pd.Series | pd.DataFrame]] = {
    "daily": generate_daily_series,
    "configurable": generate_configurable_series,
    ...
}
```

All registries follow the same shape: `dict[str, Callable]` or `dict[str, Callable[..., T]]`.

### 2. Facade Pattern

Each subpackage has a facade `.py` that re-exports from internal modules:

```
stats/tests.py ← stationarity.py + normality.py + seasonality.py + causality.py
stats/__init__.py ← tests.py + descriptive.py + result.py
```

### 3. Typed Dataclass Results

```python
@dataclass
class ValidationResult:
    is_valid: bool
    n_observations: int
    n_missing: int
    missing_pct: float
    is_monotonic: bool
    has_duplicates: bool
    dtype: str
    index_type: str

    def to_dict(self) -> dict[str, Any]: ...

@dataclass
class StatTestResult:
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float = 0.05
    interpretation: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]: ...
```

### 4. Factory Functions

All creation uses factory functions (not direct class instantiation):

```python
f = create_forecaster("theta")     # not ThetaForecaster()
p = build_transform_pipeline(["impute", "detrend"])
```
