"""Tests for multi_time.pipeline subpackage."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multi_time.config import MultiTimeConfig
from multi_time.pipeline import MultiTimePipeline, PipelineResult


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_default_empty(self):
        result = PipelineResult()
        assert result.validation == {}
        assert result.pipeline_log == []

    def test_to_dict(self):
        result = PipelineResult()
        result.validation = {"is_valid": True}
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["validation"]["is_valid"]

    def test_save(self):
        result = PipelineResult()
        result.validation = {"is_valid": True}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            result.save(path)
            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["validation"]["is_valid"]


class TestMultiTimePipeline:
    """Tests for MultiTimePipeline."""

    def test_validate_stage(self, daily_series):
        pipeline = MultiTimePipeline()
        result = pipeline.validate(daily_series)
        assert result["is_valid"]
        assert pipeline.result.validation["n_observations"] == 100

    def test_describe_stage(self, daily_series):
        pipeline = MultiTimePipeline()
        summary = pipeline.describe(daily_series)
        assert "descriptive" in summary
        assert summary["descriptive"]["count"] == 100

    def test_test_stage(self, daily_series):
        pipeline = MultiTimePipeline()
        tests = pipeline.test(daily_series)
        assert "stationarity" in tests
        assert "normality" in tests
        assert "seasonality" in tests

    def test_transform_imputes_missing(self, patchy_series):
        pipeline = MultiTimePipeline()
        transformed = pipeline.transform(patchy_series)
        assert transformed.isna().sum() == 0

    def test_forecast_stage(self, daily_series):
        config = MultiTimeConfig(models=["naive"], forecast_horizon=5)
        pipeline = MultiTimePipeline(config=config)
        predictions = pipeline.forecast(daily_series)
        assert "naive" in predictions
        assert len(predictions["naive"]) == 5

    def test_full_pipeline_without_test(self, daily_series):
        config = MultiTimeConfig(
            models=["naive"],
            forecast_horizon=5,
            output_dir="",
        )
        pipeline = MultiTimePipeline(config=config)
        result = pipeline.run(daily_series)
        assert isinstance(result, PipelineResult)
        assert result.validation["is_valid"]
        assert result.descriptive_stats["descriptive"]["count"] == 100
        assert "naive" in result.forecast_results
        assert len(result.pipeline_log) > 0

    def test_full_pipeline_with_test_data(self, daily_series):
        train = daily_series.iloc[:80]
        test = daily_series.iloc[80:]
        config = MultiTimeConfig(
            models=["naive"],
            forecast_horizon=len(test),
            metrics=["mae"],
            output_dir="",
        )
        pipeline = MultiTimePipeline(config=config)
        result = pipeline.run(train, test)
        assert isinstance(result, PipelineResult)

    def test_pipeline_log_populated(self, daily_series):
        config = MultiTimeConfig(models=["naive"], forecast_horizon=3, output_dir="")
        pipeline = MultiTimePipeline(config=config)
        result = pipeline.run(daily_series)
        assert len(result.pipeline_log) >= 5
