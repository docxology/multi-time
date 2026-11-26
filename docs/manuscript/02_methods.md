# Methods

The end-to-end script generates a seeded daily series, validates its index,
and computes descriptive statistics and diagnostic tests. It then splits
the complete series in time order and fits the default `naive` forecaster
using only the earlier observations. This last-value forecast repeats the
final training observation for all steps in the holdout, a standard
forecasting baseline [@hyndman2021fpp].

The error at each held-out timestamp is the observation minus the forecast.
Mean absolute error averages the absolute errors, while mean squared error
averages their squares. For these positive actuals, mean absolute percentage error averages the
absolute errors divided by their corresponding actuals. The JSON report records
the latter as a fraction.

The demonstration series is complete, so the script does not impute it.
When the script receives generated gaps, its optional imputation step is
before the split. For an applied study with missing values, fit preprocessing
within each training window so that it cannot use held-out data.
