# 🎯 sktime Ecosystem Integration

> [!NOTE]
> `multi-time` is built directly on the [sktime](https://github.com/sktime/sktime) unified interface for time series machine learning. This document specifies the exact mapping of `multi-time` facade methods to their low-level `sktime` counterparts, including advanced topics like tag querying and probabilistic signatures.

## sktime Base Architectures

| Base Class | Module Mapping | Purpose |
| :--- | :--- | :--- |
| `BaseForecaster` | `multi_time.modeling.forecasters` | Unified `fit()`, `predict()`, and `update()` interface for all statistical and ML models. |
| `BaseTransformer` | `multi_time.transform.transformers` | Unified `fit()`, `transform()`, and `inverse_transform()` pipeline interface. |
| `BaseSplitter` | `multi_time.modeling.evaluation` | Temporal cross-validation avoiding data leakage. |

These inherited paradigms ensure that any model integrated into `multi-time` adheres to a strict scikit-learn-like API, allowing seamless pipelining.

## State-of-the-Art Forecasters

The following core forecasters are wrapped by our `FORECASTER_REGISTRY`:

| multi-time Alias | Full `sktime` Path | Category | Strengths |
| :--- | :--- | :--- | :--- |
| `naive` | `sktime.forecasting.naive.NaiveForecaster` | Baseline | Essential for benchmarking; handles simple last-value or seasonal repetition. |
| `auto_arima` | `sktime.forecasting.arima.AutoARIMA` | Statistical | Automatic optimal order `(p, d, q)` selection. Robust for univariate series with seasonality. |
| `exp_smoothing` | `sktime.forecasting.exp_smoothing.ExponentialSmoothing` | Statistical | Handles trends and varying seasonality (ETS) in short-term univariate forecasts. |
| `theta` | `sktime.forecasting.theta.ThetaForecaster` | Decomposition | Decomposes time series to capture medium-term dynamics. Strong benchmark history. |
| `poly_trend` | `sktime.forecasting.trend.PolynomialTrendForecaster` | Trend | Deterministic continuous curve fitting. |
| `sarimax` | `sktime.forecasting.sarimax.SARIMAX` | Statistical | Statsmodels-based ARIMA with exogenous variable support. |

### Note on Machine Learning Reductions

While `multi-time` currently focuses on statistical models, `sktime` supports wrapping arbitrary `scikit-learn` regressors (e.g., `RandomForestRegressor`, `XGBoost`) into time series forecasters using `sktime.forecasting.sk_wrapper.make_reduction`. This uses a rolling window to map $y_t = f(y_{t-1}, ..., y_{t-k})$.

## Pipeline Transformers

Data preprocessing in `multi-time` heavily utilizes `sktime`'s series-to-series transformations. These are stateful (fitted on train, applied to test).

| multi-time Alias | Full `sktime` Path | Purpose | Behavior |
| :--- | :--- | :--- | :--- |
| `impute` | `sktime.transformations.series.impute.Imputer` | Gap Filling | Supports forward-fill, mean, median, nearest, or drift interpolations. |
| `detrend` | `sktime.transformations.series.detrend.Detrender` | Stationarity | Fits a polynomial/linear trend and subtracts it. |
| `deseasonalize` | `sktime.transformations.series.detrend.Deseasonalizer` | Stationarity | Removes fixed periodic seasonal components. |
| `difference` | `sktime.transformations.series.difference.Differencer` | Stationarity | Computes $y_t - y_{t-k}$ difference to stationarize mean. |
| `box_cox` | `sktime.transformations.series.boxcox.BoxCoxTransformer` | Normalization | Variance stabilization for heteroscedastic data. |
| `lag` | `sktime.transformations.series.lag.Lag` | Feature Eng | Shifts series backward to create explicit lag features. |

*Advanced usage extension*: `sktime.transformations.compose.ColumnConcatenator` can be utilized when advancing multivariate models.

## Evaluation Metrics Registry

All evaluation routing is conducted via `sktime.performance_metrics.forecasting`. These match scikit-learn API signatures.

| Metric | sktime Class | Characteristics |
| :--- | :--- | :--- |
| **MAE** | `MeanAbsoluteError` | Interpetable standard absolute error. Linear penalty. |
| **MAPE** | `MeanAbsolutePercentageError` | Scale-independent. Avoid if zeros exist in targets. |
| **sMAPE** | `MeanAbsolutePercentageError(symmetric=True)` | Symmetric MAPE. Bound between 0 and 200%. |
| **MSE** | `MeanSquaredError` | Penalizes large outlier errors quadratically. |
| **RMSE** | `MeanSquaredError(square_root=True)` | Root MSE. Retains original unit interpretability. |
| **MASE** | `MeanAbsoluteScaledError` | Scaled against naive in-sample persistence. Good for multi-series. |

## Forecasting Compositors

### Ensembles

`multi-time` utilizes `sktime.forecasting.compose.EnsembleForecaster`.

- **Purpose**: Combines multiple independent forecasters (e.g. `theta` + `auto_arima`).
- **Aggregation**: Outputs are aggregated using `mean`, `median`, or inverse-variance weighting. This provides immense robustness against individual model failure.

### Advanced Pipelining

`sktime` provides strict separation between transforming the target variable ($y$) and transforming the exogenous features ($X$).

1. **`TransformedTargetForecaster`**:
   Applies transformers strictly to the target $y$ (e.g., differencing or Box-Cox normalization) and then automatically applies `inverse_transform` to the generated forecasts.

2. **`ForecastingPipeline`**:
   Applies transformers entirely to the exogenous covariates $X$, while $y$ passes through untouched.

## Advanced Querying & Probabilistic Output

### Tag Capabilities (`get_tags()`)

Every estimator in the `sktime` ecosystem possesses a dictionary of tags queryable via `get_tags()`. `multi-time` relies on dynamic querying of these tags to build dynamic pipelines. Examples of critical tags include:

- `handles-missing-data`: Determines whether data must be routed through the `Imputer` first.
- `requires-y-train`: Used by metrics and models to denote input requirements.
- `predicts-probability`: Boolean indicating if the model natively supports quantile and interval prediction bounds.

### Probabilistic Forecasting Signatures

When querying bound estimations, implementations return standardized pandas structures, completely decoupled from intrinsic length:

- `predict_interval(fh, coverage=0.95)`: Returns a DataFrame featuring hierarchical columns for `lower` and `upper` bound bounds per forecast step.
- `predict_quantiles(fh, alpha=[0.1, 0.9])`: Yields a DataFrame with distinct percentile limits dynamically evaluated across the horizon.

## Cross-Validation

`sktime.split.ExpandingWindowSplitter` — expanding window CV for temporal data (no data leakage).

## Related Projects

| Project | Use in multi-time |
| --- | --- |
| [skpro](https://github.com/sktime/skpro) | Probabilistic prediction (optional) |
| [skbase](https://github.com/sktime/skbase) | Base classes (bundled with sktime) |
| [pytorch-forecasting](https://github.com/sktime/pytorch-forecasting) | Deep learning models (optional) |

## Version Compatibility

- **sktime ≥ 0.26** (tested with 0.40.1)
- **statsmodels ≥ 0.14** (ADF, KPSS, ARCH, Granger)
- **scipy ≥ 1.11** (distribution fitting, Shapiro-Wilk)
- **pandas ≥ 2.0** (DatetimeIndex, frequency inference)
