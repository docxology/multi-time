"""
multi_time.pipeline — Configurable end-to-end analysis pipeline.

Orchestrates validation → description → statistical testing →
transformation → forecasting → evaluation in a single configurable flow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_time.config import MultiTimeConfig
from multi_time.validate import validate_series, detect_frequency, assess_patchiness
from multi_time.stats import compute_descriptive_stats, summarize_series
from multi_time.stats import test_stationarity, test_normality, test_seasonality
from multi_time.transform import build_transform_pipeline, create_imputer, apply_transform
from multi_time.modeling import create_forecaster, run_forecast
from multi_time.evaluate import evaluate_forecast

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Aggregated results from a pipeline run.

    Attributes:
        validation: Validation result dict.
        frequency: Frequency detection result dict.
        patchiness: Patchiness analysis result dict.
        descriptive_stats: Descriptive statistics dict.
        statistical_tests: Dict of test name → StatTestResult dict.
        forecast_results: Dict of model name → predictions.
        evaluation_results: Dict of model name → metrics dict.
        pipeline_log: List of log messages from each stage.
    """

    validation: dict[str, Any] = field(default_factory=dict)
    frequency: dict[str, Any] = field(default_factory=dict)
    patchiness: dict[str, Any] = field(default_factory=dict)
    descriptive_stats: dict[str, Any] = field(default_factory=dict)
    statistical_tests: dict[str, Any] = field(default_factory=dict)
    forecast_results: dict[str, Any] = field(default_factory=dict)
    evaluation_results: dict[str, Any] = field(default_factory=dict)
    pipeline_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert all results to a JSON-serializable dict."""
        return {
            "validation": self.validation,
            "frequency": self.frequency,
            "patchiness": self.patchiness,
            "descriptive_stats": self.descriptive_stats,
            "statistical_tests": self.statistical_tests,
            "evaluation_results": self.evaluation_results,
            "pipeline_log": self.pipeline_log,
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Pipeline results saved to %s", out)


class MultiTimePipeline:
    """Configurable end-to-end time series analysis pipeline.

    Stages (all optional, controlled by config):
    1. validate — data quality checks
    2. describe — descriptive statistics
    3. test — statistical tests
    4. transform — preprocessing pipeline
    5. forecast — model fitting and prediction
    6. evaluate — forecast evaluation metrics

    Args:
        config: MultiTimeConfig controlling pipeline behavior.

    Example:
        >>> from multi_time.pipeline import MultiTimePipeline
        >>> from multi_time.config import MultiTimeConfig
        >>> config = MultiTimeConfig(forecast_horizon=12, models=["naive", "exp_smoothing"])
        >>> pipeline = MultiTimePipeline(config)
        >>> result = pipeline.run(y_train, y_test)
    """

    def __init__(self, config: MultiTimeConfig | None = None) -> None:
        self.config = config or MultiTimeConfig()
        self.result = PipelineResult()
        logger.info("MultiTimePipeline initialized with config: %s", self.config)

    def _log(self, stage: str, message: str) -> None:
        """Add a message to the pipeline log."""
        entry = f"[{stage}] {message}"
        self.result.pipeline_log.append(entry)
        logger.info(entry)

    def validate(self, data: pd.Series) -> dict[str, Any]:
        """Stage 1: Validate the time series."""
        self._log("VALIDATE", "Starting data validation")

        val_result = validate_series(data)
        self.result.validation = val_result.to_dict()

        freq_result = detect_frequency(data)
        self.result.frequency = freq_result

        patch_result = assess_patchiness(data)
        self.result.patchiness = patch_result.to_dict()

        self._log(
            "VALIDATE",
            f"Valid={val_result.is_valid}, freq={freq_result.get('inferred_freq')}, "
            f"gaps={patch_result.n_gaps}",
        )
        return self.result.validation

    def describe(self, data: pd.Series) -> dict[str, Any]:
        """Stage 2: Compute descriptive statistics."""
        self._log("DESCRIBE", "Computing descriptive statistics")

        summary = summarize_series(
            data,
            nlags=self.config.nlags_acf,
            rolling_window=self.config.rolling_window,
        )
        self.result.descriptive_stats = summary

        self._log(
            "DESCRIBE",
            f"Computed stats: mean={summary['descriptive'].get('mean', 'N/A'):.4f}, "
            f"std={summary['descriptive'].get('std', 'N/A'):.4f}",
        )
        return summary

    def test(self, data: pd.Series) -> dict[str, Any]:
        """Stage 3: Run statistical tests."""
        self._log("TEST", "Running statistical tests")

        alpha = self.config.significance_level
        tests: dict[str, Any] = {}

        stationarity = test_stationarity(data, alpha=alpha)
        tests["stationarity"] = {k: v.to_dict() for k, v in stationarity.items()}

        normality = test_normality(data, alpha=alpha)
        tests["normality"] = {k: v.to_dict() for k, v in normality.items()}

        seasonality = test_seasonality(data, period=self.config.seasonal_period, alpha=alpha)
        tests["seasonality"] = seasonality.to_dict()

        self.result.statistical_tests = tests
        self._log("TEST", f"Tests complete: {len(tests)} categories")
        return tests

    def transform(self, data: pd.Series) -> pd.Series:
        """Stage 4: Apply transformations."""
        self._log("TRANSFORM", "Applying transformations")

        transformed = data.copy()

        if data.isna().any():
            imputer = create_imputer(strategy=self.config.imputation_strategy)
            transformed = apply_transform(imputer, transformed)
            self._log("TRANSFORM", f"Imputed {data.isna().sum()} missing values")

        if self.config.transform_steps:
            pipeline = build_transform_pipeline(self.config.transform_steps)
            transformed = apply_transform(pipeline, transformed)
            self._log("TRANSFORM", f"Applied {len(self.config.transform_steps)} transform steps")

        return transformed

    def forecast(
        self,
        y_train: pd.Series,
        y_test: pd.Series | None = None,
    ) -> dict[str, pd.Series]:
        """Stage 5: Run forecasting models."""
        self._log("FORECAST", f"Running {len(self.config.models)} models")

        fh = self.config.forecast_horizon
        predictions: dict[str, pd.Series] = {}

        for model_name in self.config.models:
            try:
                forecaster = create_forecaster(model_name)
                preds = run_forecast(forecaster, y_train, fh=fh)
                predictions[model_name] = preds
                self._log("FORECAST", f"{model_name}: {len(preds)} predictions generated")
            except Exception as e:
                self._log("FORECAST", f"{model_name}: FAILED — {e}")
                logger.error("Forecaster '%s' failed: %s", model_name, e, exc_info=True)

        self.result.forecast_results = {k: v.to_dict() for k, v in predictions.items()}
        return predictions

    def evaluate(
        self,
        y_true: pd.Series,
        predictions: dict[str, pd.Series],
        y_train: pd.Series | None = None,
    ) -> dict[str, dict[str, float]]:
        """Stage 6: Evaluate forecast accuracy."""
        self._log("EVALUATE", "Computing evaluation metrics")

        evaluations: dict[str, dict[str, float]] = {}

        for model_name, y_pred in predictions.items():
            try:
                common_idx = y_true.index.intersection(y_pred.index)
                if len(common_idx) == 0:
                    self._log("EVALUATE", f"{model_name}: no overlapping indices")
                    continue

                metrics = evaluate_forecast(
                    y_true.loc[common_idx],
                    y_pred.loc[common_idx],
                    metrics_list=self.config.metrics,
                    y_train=y_train,
                )
                evaluations[model_name] = metrics
                self._log(
                    "EVALUATE",
                    f"{model_name}: {', '.join(f'{k}={v:.4f}' for k, v in metrics.items())}",
                )
            except Exception as e:
                self._log("EVALUATE", f"{model_name}: FAILED — {e}")
                logger.error("Evaluation of '%s' failed: %s", model_name, e, exc_info=True)

        self.result.evaluation_results = evaluations
        return evaluations

    def run(
        self,
        y_train: pd.Series,
        y_test: pd.Series | None = None,
    ) -> PipelineResult:
        """Run the full pipeline end-to-end.

        Args:
            y_train: Training time series.
            y_test: Optional test series for evaluation.

        Returns:
            PipelineResult with all stage outputs.
        """
        self._log("PIPELINE", "Starting full pipeline run")
        self.result = PipelineResult()

        self.validate(y_train)
        self.describe(y_train)
        self.test(y_train)
        y_transformed = self.transform(y_train)
        predictions = self.forecast(y_transformed, y_test)

        if y_test is not None and predictions:
            self.evaluate(y_test, predictions, y_train=y_train)

        if self.config.output_dir:
            output_path = Path(self.config.output_dir) / "pipeline_results.json"
            self.result.save(output_path)

        self._log("PIPELINE", "Pipeline run complete")
        return self.result
