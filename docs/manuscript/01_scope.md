# Scope and design

Multi-Time operates on pandas objects with time indexes. The data and
validation modules distinguish separate tasks: identifying a cadence,
finding gaps, and aligning multiple series to a requested frequency.
The statistical module uses SciPy and statsmodels for diagnostics.
Transformers and forecasters use sktime's estimator interface [@loning2019sktime],
with local registries providing names and defaults for supported operations.

The Python pipeline groups validation, description, statistical tests,
transformation, forecasting, and evaluation. The scripts show different
ways to compose those operations. The example here is an execution example
on generated data. Applied performance depends on the data, model choices,
and evaluation protocol.
