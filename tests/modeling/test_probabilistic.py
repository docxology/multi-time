"""Tests for multi_time.modeling subpackage — probabilistic module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_time.modeling import (
    predict_intervals,
    predict_quantiles,
    create_probabilistic_forecaster,
    fit_distribution,
    create_forecaster,
)


class TestPredictIntervals:
    """Tests for predict_intervals."""

    def test_basic_intervals(self, daily_series):
        f = create_forecaster("theta")
        intervals = predict_intervals(f, daily_series, fh=5, coverage=0.95)
        assert isinstance(intervals, pd.DataFrame)
        assert len(intervals) == 5

    def test_multiple_coverages(self, daily_series):
        f = create_forecaster("theta")
        intervals = predict_intervals(f, daily_series, fh=5, coverage=[0.80, 0.95])
        assert isinstance(intervals, pd.DataFrame)


class TestPredictQuantiles:
    """Tests for predict_quantiles."""

    def test_basic_quantiles(self, daily_series):
        f = create_forecaster("theta")
        quantiles = predict_quantiles(
            f, daily_series, fh=5, alpha=[0.1, 0.5, 0.9]
        )
        assert isinstance(quantiles, pd.DataFrame)
        assert len(quantiles) == 5


class TestFitDistribution:
    """Tests for fit_distribution."""

    def test_normal_data(self):
        np.random.seed(200)
        data = pd.Series(np.random.normal(100, 10, 500))
        result = fit_distribution(data)
        assert "norm" in result
        assert "best_fit" in result
        assert result["norm"]["ks_stat"] >= 0

    def test_exponential_data(self):
        np.random.seed(201)
        data = pd.Series(np.random.exponential(5, 500))
        result = fit_distribution(data)
        assert "expon" in result
        assert "aic" in result["expon"]

    def test_custom_distributions(self):
        np.random.seed(202)
        data = pd.Series(np.random.normal(0, 1, 200))
        result = fit_distribution(data, distributions=["norm", "t"])
        assert "norm" in result
        assert "t" in result


class TestCreateProbabilisticForecaster:
    """Tests for create_probabilistic_forecaster."""

    def test_creates_forecaster(self):
        f = create_probabilistic_forecaster("exp_smoothing", sp=1, seasonal=None)
        assert f is not None


class TestPredictVariance:
    """Tests for predict_variance."""

    def test_basic_variance(self, daily_series):
        from multi_time.modeling import predict_variance
        f = create_forecaster("theta")
        variance = predict_variance(f, daily_series, fh=5)
        assert isinstance(variance, (pd.Series, pd.DataFrame))
        assert len(variance) == 5

    def test_variance_non_negative(self, daily_series):
        from multi_time.modeling import predict_variance
        f = create_forecaster("theta")
        variance = predict_variance(f, daily_series, fh=3)
        # Variance values should be non-negative
        if isinstance(variance, pd.Series):
            assert (variance >= 0).all()


class TestTuneForecaster:
    """Tests for tune_forecaster."""

    def test_basic_tuning(self, daily_series):
        from multi_time.modeling import tune_forecaster
        f = create_forecaster("exp_smoothing", sp=1, seasonal=None, trend="add")
        # tune_forecaster signature: (forecaster, param_grid, y, ...)
        result = tune_forecaster(
            f,
            param_grid={"trend": ["add", "mul"]},
            y=daily_series,
            initial_window=50,
            fh=1,
        )
        assert result is not None

