# multi_time.data

**Data Loading + 10 Generators**

## Exports

| Symbol | Type | Description |
| --- | --- | --- |
| `load_csv_series` | `(path, column?) → pd.Series` | Load CSV to Series with DatetimeIndex |
| `load_csv_dataframe` | `(path) → pd.DataFrame` | Load CSV to DataFrame with DatetimeIndex |
| `generate_daily_series` | `(**kwargs) → pd.Series` | Daily with trend + noise |
| `generate_hourly_series` | `(**kwargs) → pd.Series` | Hourly with daily cycle |
| `generate_weekly_series` | `(**kwargs) → pd.Series` | Weekly with annual seasonality |
| `generate_monthly_series` | `(**kwargs) → pd.Series` | Monthly with seasonal amplitude |
| `generate_patchy_series` | `(**kwargs) → pd.Series` | Random missing values |
| `generate_irregular_series` | `(**kwargs) → pd.Series` | Non-uniform timestamps |
| `generate_random_walk` | `(**kwargs) → pd.Series` | Brownian motion + drift |
| `generate_multi_seasonal_series` | `(**kwargs) → pd.Series` | Multiple seasonal periods |
| `generate_multivariate_series` | `(**kwargs) → pd.DataFrame` | Correlated multi-column |
| `generate_configurable_series` | `(**kwargs) → pd.Series` | Full control over all params |
| `GENERATOR_REGISTRY` | `dict[str, Callable]` | Name → generator function mapping |
| `list_generators` | `() → list[str]` | List available generator names |

## Generator Registry

```python
GENERATOR_REGISTRY = {
    "daily":           generate_daily_series,
    "hourly":          generate_hourly_series,
    "weekly":          generate_weekly_series,
    "monthly":         generate_monthly_series,
    "patchy":          generate_patchy_series,
    "irregular":       generate_irregular_series,
    "random_walk":     generate_random_walk,
    "multi_seasonal":  generate_multi_seasonal_series,
    "multivariate":    generate_multivariate_series,
    "configurable":    generate_configurable_series,
}
```

## Generator Signatures

### `generate_configurable_series` (most flexible)

```python
def generate_configurable_series(
    n: int = 365,
    start: str = "2023-01-01",
    freq: str = "D",
    baseline: float = 100.0,
    trend: float = 0.1,
    seasonal_period: int | None = None,
    seasonal_amplitude: float = 0.0,
    noise_std: float = 1.0,
    outlier_fraction: float = 0.0,
    gap_fraction: float = 0.0,
    seed: int = 42,
) -> pd.Series:
```

### `generate_multivariate_series`

```python
def generate_multivariate_series(
    n: int = 200,
    n_series: int = 3,
    start: str = "2023-01-01",
    freq: str = "D",
    correlation: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
```

## Usage

```python
from multi_time.data import GENERATOR_REGISTRY, load_csv_series

# Load existing data
data = load_csv_series("data.csv", column="price")

# Generate synthetic
data = GENERATOR_REGISTRY["configurable"](
    n=365, trend=0.3, seasonal_period=7,
    seasonal_amplitude=5, noise_std=2, gap_fraction=0.05,
)
```
