"""Tests for multi_time.transform subpackage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


class TestTransformerFactories:
    """Tests for individual transformer creation functions."""

    def test_create_imputer(self, patchy_series):
        imputer = create_imputer(strategy="mean")
        result = apply_transform(imputer, patchy_series)
        assert result.isna().sum() == 0
        assert len(result) == len(patchy_series)

    def test_create_imputer_ffill(self, patchy_series):
        imputer = create_imputer(strategy="ffill")
        result = apply_transform(imputer, patchy_series)
        assert result.isna().sum() == 0

    def test_create_detrender(self, daily_series):
        detrender = create_detrender()
        result = apply_transform(detrender, daily_series)
        assert len(result) == len(daily_series)
        assert abs(result.mean()) < abs(daily_series.mean())

    def test_create_deseasonalizer(self, monthly_series):
        deseas = create_deseasonalizer(sp=12)
        result = apply_transform(deseas, monthly_series)
        assert len(result) == len(monthly_series)

    def test_create_box_cox(self, daily_series):
        positive = daily_series + abs(daily_series.min()) + 1
        bc = create_box_cox()
        result = apply_transform(bc, positive)
        assert len(result) == len(positive)

    def test_create_differencer(self, daily_series):
        diff = create_differencer(lags=1)
        result = apply_transform(diff, daily_series)
        assert len(result) >= len(daily_series) - 1

    def test_create_lag_transformer(self, daily_series):
        lag = create_lag_transformer(lags=1)
        result = apply_transform(lag, daily_series)
        assert result is not None


class TestBuildPipeline:
    """Tests for build_transform_pipeline."""

    def test_simple_pipeline(self, patchy_series):
        pipeline = build_transform_pipeline(["impute"])
        result = apply_transform(pipeline, patchy_series)
        assert result.isna().sum() == 0

    def test_pipeline_with_params(self, patchy_series):
        pipeline = build_transform_pipeline([
            ("impute", {"strategy": "mean"}),
        ])
        result = apply_transform(pipeline, patchy_series)
        assert result.isna().sum() == 0

    def test_unknown_transformer_raises(self):
        with pytest.raises(ValueError, match="Unknown transformer"):
            build_transform_pipeline(["nonexistent"])

    def test_registry_has_expected_keys(self):
        expected = {"impute", "detrend", "deseasonalize", "box_cox", "difference", "lag"}
        assert expected == set(TRANSFORMER_REGISTRY.keys())
