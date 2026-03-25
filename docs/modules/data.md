# 🎯 Module: data

> [!NOTE]
> CSV loading and 10 synthetic data generators strictly implementing the unified registry pattern. See [Notation and Glossary](../technical/notation.md) for data simulation variables.

## Files

| File | Purpose |
| --- | --- |
| `loaders.py` | `load_csv_series()`, `load_csv_dataframe()` — CSV I/O with date parsing |
| `generators_core.py` | 4 regular-frequency generators: daily, hourly, weekly, monthly |
| `generators_specialty.py` | 6 specialty generators: patchy, irregular, random_walk, multi_seasonal, multivariate, configurable |
| `generators.py` | **Facade** — re-exports all generators + `GENERATOR_REGISTRY` + `list_generators()` |

## Generator Registry

| Name | Function | Description |
| --- | --- | --- |
| `daily` | `generate_daily_series` | Daily with trend + noise |
| `hourly` | `generate_hourly_series` | Hourly with daily seasonality |
| `weekly` | `generate_weekly_series` | Weekly with annual seasonality |
| `monthly` | `generate_monthly_series` | Monthly with annual seasonality |
| `patchy` | `generate_patchy_series` | Daily with deliberate NaN gaps |
| `irregular` | `generate_irregular_series` | Irregularly spaced observations |
| `random_walk` | `generate_random_walk` | Non-stationary cumulative walk |
| `multi_seasonal` | `generate_multi_seasonal_series` | Multiple overlapping seasonal cycles |
| `multivariate` | `generate_multivariate_series` | Correlated multi-column DataFrame |
| `configurable` | `generate_configurable_series` | Fully parameterized (trend+season+noise+outliers+gaps) |

## Tests

`tests/data/test_data.py` — 30 tests covering all generators, registry, and loaders.
