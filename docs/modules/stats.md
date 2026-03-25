# 🎯 Module: stats

> [!NOTE]
> Descriptive statistics and 5 statistical tests split across focused sub-modules. Review [Notation and Glossary](../technical/notation.md) for test properties like `alpha` and $y_t$.

## Files

| File | Purpose |
| --- | --- |
| `descriptive.py` | `compute_descriptive_stats()`, `compute_acf_pacf()`, `compute_rolling_stats()`, `compute_seasonal_decomposition()`, `summarize_series()` |
| `result.py` | `StatTestResult` dataclass (shared by all test modules) |
| `stationarity.py` | `test_stationarity()` — ADF + KPSS tests |
| `normality.py` | `test_normality()` — Shapiro-Wilk + Jarque-Bera tests |
| `seasonality.py` | `test_seasonality()` (decomposition-based) + `test_heteroscedasticity()` (Engle's ARCH) |
| `causality.py` | `test_granger_causality()` — Granger causality with F-test across lags |
| `tests.py` | **Facade** — re-exports all test functions + `StatTestResult` |

## Key Interfaces

### `StatTestResult` dataclass

All tests return this structure: `test_name`, `statistic`, `p_value`, `is_significant`, `alpha`, `interpretation`, `details`.

### `summarize_series(data, nlags, rolling_window) → dict`

Comprehensive summary combining descriptive stats, ACF/PACF, rolling statistics, and seasonality info into a single dict.

## Tests

`tests/stats/test_statistical_tests.py` — Tests for all statistical test functions including stationarity, normality, seasonality, heteroscedasticity, and Granger causality.

`tests/stats/test_descriptive.py` — Tests for descriptive statistics functions.
