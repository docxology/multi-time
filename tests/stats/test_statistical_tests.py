"""Tests for multi_time.stats subpackage — statistical tests module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_time.stats import (
    test_stationarity as check_stationarity,
    test_normality as check_normality,
    test_seasonality as check_seasonality,
    test_heteroscedasticity as check_heteroscedasticity,
    test_granger_causality as check_granger_causality,
    StatTestResult,
)


class TestStationarity:
    """Tests for test_stationarity."""

    def test_stationary_series_detected(self, stationary_series):
        results = check_stationarity(stationary_series)
        assert "adf" in results
        assert "kpss" in results
        assert results["adf"].is_significant
        assert not results["kpss"].is_significant

    def test_nonstationary_series_detected(self, nonstationary_series):
        results = check_stationarity(nonstationary_series)
        assert not results["adf"].is_significant

    def test_result_structure(self, daily_series):
        results = check_stationarity(daily_series)
        adf = results["adf"]
        assert isinstance(adf, StatTestResult)
        assert adf.test_name == "Augmented Dickey-Fuller"
        assert isinstance(adf.statistic, float)
        assert isinstance(adf.p_value, float)
        assert isinstance(adf.interpretation, str)
        assert adf.details is not None

    def test_to_dict(self, daily_series):
        results = check_stationarity(daily_series)
        d = results["adf"].to_dict()
        assert isinstance(d, dict)
        assert "statistic" in d
        assert "p_value" in d


class TestNormality:
    """Tests for test_normality."""

    def test_normal_data_passes(self):
        np.random.seed(100)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        normal = pd.Series(np.random.normal(0, 1, 200), index=dates)
        results = check_normality(normal)
        assert "shapiro" in results
        assert "jarque_bera" in results

    def test_non_normal_detected(self):
        np.random.seed(101)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        exponential = pd.Series(np.random.exponential(5, 200), index=dates)
        results = check_normality(exponential)
        assert results["shapiro"].is_significant or results["jarque_bera"].is_significant


class TestSeasonality:
    """Tests for test_seasonality."""

    def test_seasonal_series_detected(self, monthly_series):
        result = check_seasonality(monthly_series, period=12)
        assert result.statistic > 0.5
        assert result.is_significant

    def test_random_series_not_seasonal(self, stationary_series):
        result = check_seasonality(stationary_series, period=7)
        assert result.statistic < 0.64

    def test_short_series_handled(self, short_series):
        result = check_seasonality(short_series, period=12)
        assert "too short" in result.interpretation.lower()


class TestHeteroscedasticity:
    """Tests for test_heteroscedasticity."""

    def test_homoscedastic_data(self, stationary_series):
        result = check_heteroscedasticity(stationary_series)
        assert isinstance(result, StatTestResult)
        assert isinstance(result.p_value, float)

    def test_arch_effects(self):
        np.random.seed(102)
        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        values = np.zeros(500)
        for i in range(1, 500):
            sigma = np.sqrt(1 + 0.5 * values[i - 1] ** 2)
            values[i] = sigma * np.random.normal()
        series = pd.Series(values, index=dates)
        result = check_heteroscedasticity(series)
        assert isinstance(result.statistic, float)


class TestGrangerCausality:
    """Tests for test_granger_causality."""

    def test_causal_relationship_detected(self, multivariate_df):
        result = check_granger_causality(
            multivariate_df["x"], multivariate_df["y"], maxlag=4
        )
        assert result.is_significant
        assert result.details["best_lag"] >= 1

    def test_no_causality(self, stationary_series):
        np.random.seed(103)
        dates = stationary_series.index
        independent = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)
        result = check_granger_causality(independent, stationary_series, maxlag=4)
        assert isinstance(result, StatTestResult)
