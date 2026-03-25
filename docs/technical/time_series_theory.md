# 🎯 Time Series Theory Reference

> [!NOTE]
> Analytical foundations used by `multi-time`. Before engaging deeply with these tests, please review the centralized [Notation and Glossary](notation.md) to familiarize yourself with the structural meaning of variables such as $y_t$, $X$, and $h$.

## Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) do not change over time. Stationarity is a prerequisite for many classical models.

### Tests

| Test | H₀ | Reject Means | Method |
| --- | --- | --- | --- |
| **ADF** (Augmented Dickey-Fuller) | Unit root exists (non-stationary) | Series is stationary | Regression with lagged differences |
| **KPSS** | Series is stationary | Series is non-stationary | LM test around deterministic trend |

**Best practice**: Use ADF + KPSS together. If ADF rejects and KPSS does not → stationary. If ADF fails and KPSS rejects → non-stationary. Conflicting → trend-stationary or further analysis needed.

## Decomposition

Time series = Trend + Seasonal + Residual

- **Additive**: Y(t) = T(t) + S(t) + R(t) — when seasonal amplitude is constant
- **Multiplicative**: Y(t) = T(t) × S(t) × R(t) — when seasonal amplitude grows with level

`multi-time` uses STL (Seasonal and Trend decomposition using Loess) via statsmodels.

## ARIMA / SARIMA

**ARIMA(p, d, q)**:

- **p** = autoregressive order (ACF for MA, PACF for AR)
- **d** = differencing order (from ADF test)
- **q** = moving average order

**SARIMA(p,d,q)(P,D,Q,s)** adds seasonal terms with period `s`.

`auto_arima` (via pmdarima) performs automatic model selection using AIC/BIC.

## Exponential Smoothing (ETS)

**Error-Trend-Seasonal** framework:

- Simple (no trend, no seasonal)
- Holt (additive trend)
- Holt-Winters (trend + seasonal)
- Damped trend variants

Mapped via `create_forecaster("exp_smoothing", sp=12, trend="add", seasonal="add")`.

## Theta Method

Decomposes into two θ-lines:

1. θ=0: linear regression (captures trend)
2. θ=2: emphasizes recent observations via SES

Forecast = weighted combination. Simple, robust, strong on M3/M4 competitions.

## ACF and PACF

- **ACF** (Autocorrelation Function): correlation between Y(t) and Y(t-k) for all k
- **PACF** (Partial ACF): direct correlation at lag k, removing intermediate correlations

### Interpretation Rules

- **AR(p)**: PACF cuts off after lag p, ACF decays
- **MA(q)**: ACF cuts off after lag q, PACF decays
- **ARMA**: both decay gradually

## Normality Tests

| Test | Best For | Limitation |
| --- | --- | --- |
| **Shapiro-Wilk** | Small samples (n < 5000) | Low power for n > 5000 |
| **Jarque-Bera** | Large samples | Based only on skewness + kurtosis |

## Heteroscedasticity (ARCH Effects)

**Engle's ARCH test** checks if residual variance follows an autoregressive pattern. Significant ARCH effects indicate volatility clustering → consider GARCH models.

## Granger Causality

Tests whether past values of X help predict Y beyond what Y's own past provides. Does **not** imply true causation — only predictive precedence.

## Distribution Fitting

`fit_distribution` fits parametric distributions (normal, t, exponential, gamma, lognormal) to data and ranks by:

- **AIC** (Akaike Information Criterion) — penalizes complexity
- **BIC** (Bayesian Information Criterion) — stronger complexity penalty
- **KS statistic** (Kolmogorov-Smirnov) — max CDF deviation
