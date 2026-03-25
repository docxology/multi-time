# multi_time.stats

**Descriptive + Inferential Statistics**

## Exports

| Symbol | Type | Description |
| --- | --- | --- |
| `compute_descriptive_stats` | `(pd.Series) → dict` | mean, std, skew, kurtosis, percentiles |
| `compute_acf_pacf` | `(pd.Series, nlags) → dict` | ACF + PACF values |
| `compute_rolling_stats` | `(pd.Series, window) → dict` | Rolling mean, std, min, max |
| `compute_seasonal_decomposition` | `(pd.Series, period) → dict` | STL decomposition |
| `summarize_series` | `(pd.Series, nlags, rolling_window) → dict` | All-in-one summary |
| `test_stationarity` | `(pd.Series) → dict[str, StatTestResult]` | ADF + KPSS |
| `test_normality` | `(pd.Series) → dict[str, StatTestResult]` | Shapiro-Wilk + Jarque-Bera |
| `test_seasonality` | `(pd.Series, period) → StatTestResult` | Autocorrelation-based |
| `test_heteroscedasticity` | `(pd.Series) → StatTestResult` | ARCH/LM test |
| `test_granger_causality` | `(pd.Series, pd.Series, maxlag) → StatTestResult` | Bivariate Granger |
| `StatTestResult` | `dataclass` | Typed test result |

## Schema: StatTestResult

```python
@dataclass
class StatTestResult:
    test_name: str           # e.g. "ADF", "Shapiro-Wilk"
    statistic: float         # Test statistic value
    p_value: float           # p-value
    is_significant: bool     # p_value < alpha
    alpha: float = 0.05      # Significance level
    interpretation: str = "" # Human-readable conclusion
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]: ...
```

## Output Schemas

### `summarize_series` Output

```python
{
    "descriptive": {
        "mean": float, "std": float, "min": float, "max": float,
        "q25": float, "q50": float, "q75": float,
        "skewness": float, "kurtosis": float, "n": int,
    },
    "acf_pacf": {
        "acf_values": list[float],   # length nlags+1
        "pacf_values": list[float],  # length nlags+1
        "nlags": int,
    },
    "rolling": {
        "mean": list[float],
        "std": list[float],
    },
}
```

### `test_stationarity` Output

```python
{
    "adf": StatTestResult(
        test_name="ADF",
        statistic=-3.45,
        p_value=0.009,
        is_significant=True,
        interpretation="Stationary (reject H0: unit root)",
        details={"critical_values": {"1%": -3.44, "5%": -2.87, "10%": -2.57}, "n_lags_used": 14},
    ),
    "kpss": StatTestResult(
        test_name="KPSS",
        statistic=0.12,
        p_value=0.10,
        is_significant=False,
        interpretation="Stationary (fail to reject H0: stationarity)",
    ),
}
```

## Statistical Tests Reference

| Test | H₀ | Significant = | Module |
| --- | --- | --- | --- |
| ADF | Unit root present | Stationary | `stationarity.py` |
| KPSS | Series is stationary | Non-stationary | `stationarity.py` |
| Shapiro-Wilk | Normal distribution | Non-normal | `normality.py` |
| Jarque-Bera | Normal distribution | Non-normal | `normality.py` |
| ARCH/LM | No heteroscedasticity | Heteroscedastic | `seasonality.py` |
| Granger | X does not cause Y | X Granger-causes Y | `causality.py` |

## Usage

```python
from multi_time.stats import summarize_series, test_stationarity, test_normality

summary = summarize_series(data, nlags=40, rolling_window=12)
print(f"Mean: {summary['descriptive']['mean']:.2f}")

stationarity = test_stationarity(data)
print(f"ADF: {stationarity['adf'].interpretation}")
print(f"KPSS: {stationarity['kpss'].interpretation}")

normality = test_normality(data)
print(f"Shapiro p={normality['shapiro'].p_value:.4f}")
```
