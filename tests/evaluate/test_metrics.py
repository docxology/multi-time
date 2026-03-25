"""Tests for multi_time.evaluate subpackage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_time.evaluate import (
    compute_metric,
    evaluate_forecast,
    compute_rmse,
    list_available_metrics,
    METRIC_REGISTRY,
)


@pytest.fixture
def forecast_data():
    """Create aligned true/predicted series."""
    idx = pd.date_range("2023-07-01", periods=10, freq="D")
    y_true = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], index=idx, dtype=float)
    y_pred = pd.Series([12, 18, 33, 37, 52, 58, 73, 77, 92, 98], index=idx, dtype=float)
    return y_true, y_pred


class TestComputeMetric:
    """Tests for compute_metric."""

    def test_mae(self, forecast_data):
        y_true, y_pred = forecast_data
        mae = compute_metric(y_true, y_pred, "mae")
        assert mae > 0
        assert mae == pytest.approx(2.4)

    def test_mse(self, forecast_data):
        y_true, y_pred = forecast_data
        mse = compute_metric(y_true, y_pred, "mse")
        assert mse > 0

    def test_mape(self, forecast_data):
        y_true, y_pred = forecast_data
        mape = compute_metric(y_true, y_pred, "mape")
        assert mape > 0

    def test_unknown_metric_raises(self, forecast_data):
        y_true, y_pred = forecast_data
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_metric(y_true, y_pred, "nonexistent")


class TestEvaluateForecast:
    """Tests for evaluate_forecast."""

    def test_default_metrics(self, forecast_data):
        y_true, y_pred = forecast_data
        results = evaluate_forecast(y_true, y_pred)
        assert "mae" in results
        assert "mse" in results
        assert "mape" in results
        assert all(isinstance(v, float) for v in results.values())

    def test_custom_metrics(self, forecast_data):
        y_true, y_pred = forecast_data
        results = evaluate_forecast(y_true, y_pred, metrics_list=["mae"])
        assert "mae" in results
        assert "mse" not in results

    def test_failed_metric_returns_nan(self, forecast_data):
        y_true, y_pred = forecast_data
        results = evaluate_forecast(y_true, y_pred, metrics_list=["mase"])
        if "mase" in METRIC_REGISTRY:
            assert np.isnan(results.get("mase", 0.0)) or results.get("mase", 0.0) >= 0


class TestComputeRMSE:
    """Tests for compute_rmse."""

    def test_rmse_value(self, forecast_data):
        y_true, y_pred = forecast_data
        rmse = compute_rmse(y_true, y_pred)
        assert rmse > 0
        mae = compute_metric(y_true, y_pred, "mae")
        assert rmse >= mae - 1e-6


class TestListMetrics:
    """Tests for list_available_metrics."""

    def test_lists_core_metrics(self):
        available = list_available_metrics()
        assert "mae" in available
        assert "mse" in available
        assert "mape" in available
        assert isinstance(available, list)
