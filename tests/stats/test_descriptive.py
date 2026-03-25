"""Tests for multi_time.stats subpackage — descriptive module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_time.stats import (
    compute_descriptive_stats,
    compute_acf_pacf,
    compute_rolling_stats,
    compute_seasonal_decomposition,
    summarize_series,
)


class TestDescriptiveStats:
    """Tests for compute_descriptive_stats."""

    def test_basic_stats(self, daily_series):
        stats = compute_descriptive_stats(daily_series)
        assert stats["count"] == 100
        assert "mean" in stats
        assert "std" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_quantiles(self, daily_series):
        stats = compute_descriptive_stats(daily_series)
        assert stats["q25"] <= stats["q50"] <= stats["q75"]
        assert stats["iqr"] == pytest.approx(stats["q75"] - stats["q25"])

    def test_range(self, daily_series):
        stats = compute_descriptive_stats(daily_series)
        assert stats["range"] == pytest.approx(stats["max"] - stats["min"])

    def test_cv_nonzero_mean(self, daily_series):
        stats = compute_descriptive_stats(daily_series)
        assert stats["cv"] > 0
        assert stats["cv"] == pytest.approx(stats["std"] / abs(stats["mean"]))

    def test_handles_nan(self, patchy_series):
        stats = compute_descriptive_stats(patchy_series)
        assert stats["count"] == 85
        assert "mean" in stats

    def test_all_nan_returns_error(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        stats = compute_descriptive_stats(s)
        assert stats["count"] == 0


class TestACFPACF:
    """Tests for compute_acf_pacf."""

    def test_acf_values(self, daily_series):
        result = compute_acf_pacf(daily_series, nlags=20)
        assert len(result["acf_values"]) == 21
        assert result["acf_values"][0] == pytest.approx(1.0)
        assert result["nlags"] == 20

    def test_pacf_values(self, daily_series):
        result = compute_acf_pacf(daily_series, nlags=20)
        assert len(result["pacf_values"]) == 21

    def test_significant_lags_lists(self, monthly_series):
        result = compute_acf_pacf(monthly_series, nlags=20)
        assert isinstance(result["significant_acf_lags"], list)
        assert isinstance(result["significant_pacf_lags"], list)

    def test_short_series_adjusts_nlags(self, short_series):
        result = compute_acf_pacf(short_series, nlags=40)
        assert result["nlags"] < 40


class TestRollingStats:
    """Tests for compute_rolling_stats."""

    def test_output_columns(self, daily_series):
        result = compute_rolling_stats(daily_series, window=7)
        assert "rolling_mean" in result.columns
        assert "rolling_std" in result.columns
        assert "rolling_min" in result.columns
        assert "rolling_max" in result.columns
        assert "rolling_median" in result.columns

    def test_output_length(self, daily_series):
        result = compute_rolling_stats(daily_series, window=7)
        assert len(result) == len(daily_series)


class TestSeasonalDecomposition:
    """Tests for compute_seasonal_decomposition."""

    def test_decomposition_components(self, monthly_series):
        result = compute_seasonal_decomposition(monthly_series, period=12)
        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result
        assert "observed" in result

    def test_decomposition_length(self, monthly_series):
        result = compute_seasonal_decomposition(monthly_series, period=12)
        assert len(result["seasonal"]) == len(monthly_series)


class TestSummarizeSeries:
    """Tests for summarize_series."""

    def test_summary_structure(self, daily_series):
        summary = summarize_series(daily_series)
        assert "descriptive" in summary
        assert "acf_pacf" in summary
        assert "rolling" in summary

    def test_summary_descriptive(self, daily_series):
        summary = summarize_series(daily_series)
        assert summary["descriptive"]["count"] == 100
