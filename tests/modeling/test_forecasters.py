"""Tests for multi_time.modeling subpackage — forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_time.modeling import (
    create_forecaster,
    create_ensemble,
    run_forecast,
    evaluate_forecaster,
    FORECASTER_REGISTRY,
)


class TestCreateForecaster:
    """Tests for create_forecaster factory."""

    def test_create_naive(self):
        f = create_forecaster("naive")
        assert f is not None

    def test_create_exp_smoothing(self):
        f = create_forecaster("exp_smoothing", sp=12)
        assert f is not None

    def test_create_theta(self):
        f = create_forecaster("theta")
        assert f is not None

    def test_create_poly_trend(self):
        f = create_forecaster("poly_trend", degree=2)
        assert f is not None

    def test_create_sarimax(self):
        f = create_forecaster("sarimax")
        assert f is not None

    def test_unknown_forecaster_raises(self):
        with pytest.raises(ValueError, match="Unknown forecaster"):
            create_forecaster("nonexistent")

    def test_registry_has_expected_keys(self):
        expected = {"naive", "exp_smoothing", "theta", "poly_trend", "auto_arima", "sarimax"}
        assert expected == set(FORECASTER_REGISTRY.keys())


class TestRunForecast:
    """Tests for run_forecast."""

    def test_naive_forecast(self, daily_series):
        f = create_forecaster("naive")
        preds = run_forecast(f, daily_series, fh=5)
        assert len(preds) == 5
        assert not preds.isna().any()

    def test_exp_smoothing_forecast(self, daily_series):
        f = create_forecaster("exp_smoothing", sp=1, seasonal=None, trend="add")
        preds = run_forecast(f, daily_series, fh=6)
        assert len(preds) == 6

    def test_theta_forecast(self, daily_series):
        f = create_forecaster("theta")
        preds = run_forecast(f, daily_series, fh=10)
        assert len(preds) == 10

    def test_poly_trend_forecast(self, daily_series):
        f = create_forecaster("poly_trend")
        preds = run_forecast(f, daily_series, fh=5)
        assert len(preds) == 5

    def test_forecast_with_list_fh(self, daily_series):
        f = create_forecaster("naive")
        preds = run_forecast(f, daily_series, fh=[1, 2, 3])
        assert len(preds) == 3


class TestCreateEnsemble:
    """Tests for create_ensemble."""

    def test_basic_ensemble(self, daily_series):
        ensemble = create_ensemble(["naive", "theta"])
        preds = run_forecast(ensemble, daily_series, fh=5)
        assert len(preds) == 5

    def test_ensemble_with_params(self, daily_series):
        ensemble = create_ensemble([
            "naive",
            ("poly_trend", {"degree": 1}),
        ])
        preds = run_forecast(ensemble, daily_series, fh=3)
        assert len(preds) == 3


class TestEvaluateForecaster:
    """Tests for evaluate_forecaster."""

    def test_evaluation_returns_dataframe(self, daily_series):
        f = create_forecaster("naive")
        result = evaluate_forecaster(f, daily_series, initial_window=50, fh=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
