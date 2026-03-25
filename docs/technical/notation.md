# 🎯 Glossary and Mathematical Notation

> [!NOTE]
> This document centralizes all mathematical symbols, variable conventions, and technical glossary terms used systematically across the `multi-time` engine and its theoretical `sktime` foundations. Please adhere to these definitions when inspecting source or integrating new modules.

## Mathematical Notation

| Symbol | Code Variable | Concept Definition |
| :--- | :--- | :--- |
| $y$ | `y`, `y_train` | The target endogenous time series variable being observed or forecasted. |
| $y_t$ | `y.iloc[-1]` | The current observation of the series at time step $t$. |
| $y_{t-k}$ | `y.shift(k)` | An observation of the series lagged by exactly $k$ time steps. |
| $X$ | `X`, `X_train` | Exogenous variables (covariates) used to aid the forecasting of $y$. |
| $\hat{y}$ | `y_pred` | The point forecast (predicted mean/median) produced by the model. |
| $h$ | `fh` | The forecasting horizon; the number of steps into the future to predict. |
| $s$ | `sp` | Seasonal period (e.g., 12 for monthly data with annual seasonality). |
| $\alpha$ | `alpha` | The significance level for statistical tests (defaults to 0.05). |
| $1-\alpha$ | `coverage` | The confidence coverage for prediction intervals (e.g., 0.95 or 0.80). |

## Core Ecosystem Glossary

- **Endogenous (`y`)**: The internal target variable whose future values we seek to predict based purely on its own past (and optionally external data).
- **Exogenous (`X`)**: External driver variables representing independent features (e.g., holidays, marketing spend) that causally influence $y$.
- **Stationarity**: A fundamental property requiring the mean, variance, and autocorrelation structure of $y$ to remain constant over time, enabling reliable classical forecasting (e.g., via ARIMA).
- **Heteroscedasticity**: The condition where the variance of the error terms $R(t)$ changes or clusters over time (volatile periods vs calm periods), often requiring GARCH modeling or `BoxCox` transforms.
- **Data Leakage**: The catastrophic error of allowing future information (from the test set) to influence the training or transformation of the training set. Strictly prevented via `ExpandingWindowSplitter`.
- **Reduction (`make_reduction`)**: The architectural pattern of transforming a tabular Machine Learning model (e.g., Random Forest) into a stateful time series forecaster using a sliding historical window.

## Architectural `sktime` Terminology

- **Forecaster**: Any estimator inheriting from `BaseForecaster` providing `fit()`, `predict()`, and `update()`.
- **Transformer**: Any estimator inheriting from `BaseTransformer` providing `fit()`, `transform()`, and `inverse_transform()`.
- **TransformedTargetForecaster**: A sequential pipeline applying transformers exclusively to the target variable $y$ before modeling, reversing the transform automatically during prediction.
- **ForecastingPipeline**: A standard pipeline applying transformers to the exogenous covariates $X$ prior to forecasting.

> [!TIP]
> When reviewing specific algorithm behavior, always check the `get_tags()` method on the estimator to instantly verify capabilities such as `handles-missing-data` or `predicts-probability`.
