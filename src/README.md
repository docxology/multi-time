# src/

Source root for the `multi-time` package (v0.3.0). Contains the `multi_time` Python package. This package is built explicitly as a robust wrapper around the `sktime` ecosystem, incorporating state-of-the-art implementations (e.g., `AutoARIMA`, `EnsembleForecaster`, ML Reductions via `make_reduction`, and exhaustive metric registries).

## Structure

```
src/
└── multi_time/              # Main package (9 subpackages, 40+ Python files)
    ├── __init__.py          # Root re-exports: 50+ public symbols
    ├── config/              # MultiTimeConfig + structured logging
    │   ├── settings.py      # Dataclass config + YAML loader
    │   └── logging.py       # get_logger(), setup_logging()
    ├── data/                # CSV I/O + 10 generators
    │   ├── loaders.py       # load_csv_series(), load_csv_dataframe()
    │   ├── generators_core.py      # 4 regular: daily, hourly, weekly, monthly
    │   ├── generators_specialty.py # 6 specialty: patchy, irregular, random_walk, ...
    │   └── generators.py    # Facade + GENERATOR_REGISTRY
    ├── validate/            # Validation + frequency + patchiness
    │   ├── validation.py    # validate_series() → ValidationResult
    │   ├── frequency.py     # detect_frequency()
    │   ├── patchiness.py    # assess_patchiness() → PatchinessResult
    │   ├── harmonize.py     # harmonize_frequencies()
    │   └── validators.py    # Facade
    ├── stats/               # Descriptive + 5 tests
    │   ├── descriptive.py   # compute_*, summarize_series()
    │   ├── result.py        # StatTestResult dataclass
    │   ├── stationarity.py  # ADF + KPSS
    │   ├── normality.py     # Shapiro-Wilk + Jarque-Bera
    │   ├── seasonality.py   # Decomposition + ARCH
    │   ├── causality.py     # Granger causality
    │   └── tests.py         # Facade
    ├── transform/           # sktime wrappers (Detrender, Imputer, Lag, BoxCox)
    │   └── transformers.py  # 6 factories + pipeline builder + TRANSFORMER_REGISTRY
    ├── modeling/             # Forecasting (AutoARIMA, Theta, Ensembles, ML Reductions)
    │   ├── registry.py      # FORECASTER_REGISTRY + create_forecaster()
    │   ├── ensemble.py      # create_ensemble()
    │   ├── evaluation.py    # evaluate_forecaster() + tune_forecaster()
    │   ├── probabilistic.py # predict_intervals/quantiles/variance + fit_distribution
    │   └── forecasters.py   # Facade + run_forecast()
    ├── evaluate/            # sktime.performance_metrics (MAE, MAPE, sMAPE, RMSE)
    │   └── metrics.py       # METRIC_REGISTRY + evaluate_forecast()
    ├── visualization/       # 19 plot functions in 5 sub-modules
    │   ├── core.py          # Agg backend, COLORS, save_or_show()
    │   ├── series.py        # plot_series()
    │   ├── forecast.py      # plot_forecast(), plot_residuals()
    │   ├── diagnostics.py   # plot_acf_pacf(), plot_decomposition(), plot_diagnostics()
    │   ├── statistics.py    # 8 functions: rolling, distribution, lag, boxplot, heatmap, ...
    │   ├── comparison.py    # 3 functions: model_comparison, error_distribution, cumulative_error
    │   └── plots.py         # Facade
    └── pipeline/            # End-to-end
        └── pipeline.py      # MultiTimePipeline + PipelineResult
```

## Design Patterns

- **Facade** — each subpackage has a facade module re-exporting core functions
- **Registry** — extensible dicts for generators, forecasters, transformers, metrics
- **Dataclass** — typed result objects (`ValidationResult`, `PatchinessResult`, `StatTestResult`, `PipelineResult`, `MultiTimeConfig`)
- **Type hints** — all functions typed with `from __future__ import annotations`
- **Logging** — `logging.getLogger(__name__)` in every module
