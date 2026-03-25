"""
multi-time: Comprehensive multi-frequency time series analysis toolkit.

Built on the sktime ecosystem. Provides validation, descriptive statistics,
statistical testing, transformation, forecasting, probabilistic modeling,
evaluation, and visualization — all configurable and modular.

Subpackages:
    config      — Configuration system and structured logging
    data        — Data loading, I/O, and 10 sample data generators
    validate    — Series validation, frequency detection, patchiness analysis
    stats       — Descriptive stats + statistical tests (ADF, KPSS, etc.)
    transform   — sktime transformer wrappers and pipeline builder
    modeling    — Forecasting (factory, ensemble, tuning) + probabilistic
    evaluate    — Forecast evaluation metrics
    visualization — 19 matplotlib-based plotting and diagnostics functions
    pipeline    — End-to-end configurable pipeline orchestrator

Example:
    >>> import multi_time as mt
    >>> data = mt.generate_daily_series(n=365)
    >>> val = mt.validate_series(data)
    >>> stats = mt.summarize_series(data)
    >>> forecaster = mt.create_forecaster("theta")
    >>> predictions = mt.run_forecast(forecaster, data, fh=30)
"""

__version__ = "0.3.0"

# ── Configuration ──────────────────────────────────────────────────
from multi_time.config import MultiTimeConfig, load_config, get_logger, setup_logging

# ── Data ───────────────────────────────────────────────────────────
from multi_time.data import (
    load_csv_series,
    load_csv_dataframe,
    generate_daily_series,
    generate_hourly_series,
    generate_weekly_series,
    generate_monthly_series,
    generate_patchy_series,
    generate_irregular_series,
    generate_random_walk,
    generate_multi_seasonal_series,
    generate_multivariate_series,
    generate_configurable_series,
    GENERATOR_REGISTRY,
    list_generators,
)

# ── Validation ─────────────────────────────────────────────────────
from multi_time.validate import (
    validate_series,
    detect_frequency,
    assess_patchiness,
    harmonize_frequencies,
    ValidationResult,
    PatchinessResult,
)

# ── Statistics ─────────────────────────────────────────────────────
from multi_time.stats import (
    compute_descriptive_stats,
    compute_acf_pacf,
    compute_rolling_stats,
    compute_seasonal_decomposition,
    summarize_series,
    test_stationarity,
    test_normality,
    test_seasonality,
    test_heteroscedasticity,
    test_granger_causality,
    StatTestResult,
)

# ── Transformations ────────────────────────────────────────────────
from multi_time.transform import (
    create_imputer,
    create_detrender,
    create_deseasonalizer,
    create_box_cox,
    create_differencer,
    create_lag_transformer,
    build_transform_pipeline,
    apply_transform,
    TRANSFORMER_REGISTRY,
)

# ── Modeling ───────────────────────────────────────────────────────
from multi_time.modeling import (
    create_forecaster,
    create_ensemble,
    run_forecast,
    evaluate_forecaster,
    tune_forecaster,
    FORECASTER_REGISTRY,
    predict_intervals,
    predict_quantiles,
    predict_variance,
    create_probabilistic_forecaster,
    fit_distribution,
)

# ── Evaluation ─────────────────────────────────────────────────────
from multi_time.evaluate import (
    compute_metric,
    evaluate_forecast,
    compute_rmse,
    list_available_metrics,
    METRIC_REGISTRY,
)

# ── Pipeline ───────────────────────────────────────────────────────
from multi_time.pipeline import MultiTimePipeline, PipelineResult

__all__ = [
    "__version__",
    # Config
    "MultiTimeConfig",
    "load_config",
    "get_logger",
    "setup_logging",
    # Data
    "load_csv_series",
    "load_csv_dataframe",
    "generate_daily_series",
    "generate_hourly_series",
    "generate_weekly_series",
    "generate_monthly_series",
    "generate_patchy_series",
    "generate_irregular_series",
    "generate_random_walk",
    "generate_multi_seasonal_series",
    "generate_multivariate_series",
    "generate_configurable_series",
    "GENERATOR_REGISTRY",
    "list_generators",
    # Validation
    "validate_series",
    "detect_frequency",
    "assess_patchiness",
    "harmonize_frequencies",
    "ValidationResult",
    "PatchinessResult",
    # Statistics
    "compute_descriptive_stats",
    "compute_acf_pacf",
    "compute_rolling_stats",
    "compute_seasonal_decomposition",
    "summarize_series",
    "test_stationarity",
    "test_normality",
    "test_seasonality",
    "test_heteroscedasticity",
    "test_granger_causality",
    "StatTestResult",
    # Transformations
    "create_imputer",
    "create_detrender",
    "create_deseasonalizer",
    "create_box_cox",
    "create_differencer",
    "create_lag_transformer",
    "build_transform_pipeline",
    "apply_transform",
    "TRANSFORMER_REGISTRY",
    # Modeling
    "create_forecaster",
    "create_ensemble",
    "run_forecast",
    "evaluate_forecaster",
    "tune_forecaster",
    "FORECASTER_REGISTRY",
    "predict_intervals",
    "predict_quantiles",
    "predict_variance",
    "create_probabilistic_forecaster",
    "fit_distribution",
    # Evaluation
    "compute_metric",
    "evaluate_forecast",
    "compute_rmse",
    "list_available_metrics",
    "METRIC_REGISTRY",
    # Pipeline
    "MultiTimePipeline",
    "PipelineResult",
]
