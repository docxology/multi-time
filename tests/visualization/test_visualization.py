"""Tests for multi_time.visualization subpackage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# matplotlib may not be installed; skip if missing
pytest.importorskip("matplotlib")

from multi_time.visualization import (
    plot_series,
    plot_forecast,
    plot_acf_pacf,
    plot_decomposition,
    plot_diagnostics,
    plot_residuals,
    plot_rolling_statistics,
    plot_distribution,
    plot_lag_scatter,
    plot_boxplot_by_period,
    plot_correlation_heatmap,
    plot_stationarity_summary,
    plot_validation_summary,
    plot_missing_data,
    plot_model_comparison,
    plot_error_distribution,
    plot_cumulative_error,
)


class TestPlotSeries:
    """Tests for plot_series."""

    def test_single_series(self, daily_series):
        fig = plot_series(daily_series, title="Test")
        assert fig is not None

    def test_multiple_series(self, daily_series, monthly_series):
        fig = plot_series(daily_series, monthly_series, labels=["Daily", "Monthly"])
        assert fig is not None

    def test_save_to_file(self, daily_series):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "series.png"
            fig = plot_series(daily_series, save_path=path)
            assert path.exists()


class TestPlotForecast:
    """Tests for plot_forecast."""

    def test_basic_forecast_plot(self, daily_series):
        train = daily_series.iloc[:80]
        pred_idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=20, freq="D")
        y_pred = pd.Series(np.random.normal(50, 2, 20), index=pred_idx)
        fig = plot_forecast(train, y_pred, title="Test Forecast")
        assert fig is not None

    def test_forecast_with_actuals(self, daily_series):
        train = daily_series.iloc[:80]
        test = daily_series.iloc[80:]
        pred = test + np.random.normal(0, 1, len(test))
        fig = plot_forecast(train, pred, y_test=test)
        assert fig is not None


class TestPlotACFPACF:
    """Tests for plot_acf_pacf."""

    def test_basic_acf_pacf(self):
        acf = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        pacf = [1.0, 0.7, 0.1, 0.05, -0.02, -0.01]
        fig = plot_acf_pacf(acf, pacf, nlags=5)
        assert fig is not None


class TestPlotDecomposition:
    """Tests for plot_decomposition."""

    def test_basic_decomposition(self, monthly_series):
        decomp = {
            "observed": monthly_series,
            "trend": monthly_series.rolling(12).mean(),
            "seasonal": pd.Series(np.sin(np.arange(len(monthly_series))), index=monthly_series.index),
            "residual": pd.Series(np.random.normal(0, 1, len(monthly_series)), index=monthly_series.index),
        }
        fig = plot_decomposition(decomp)
        assert fig is not None


class TestPlotDiagnostics:
    """Tests for plot_diagnostics."""

    def test_basic_diagnostics(self, daily_series):
        fig = plot_diagnostics(daily_series, title="Test Diagnostics")
        assert fig is not None


class TestPlotResiduals:
    """Tests for plot_residuals."""

    def test_basic_residuals(self, daily_series):
        residuals = pd.Series(np.random.normal(0, 1, len(daily_series)), index=daily_series.index)
        fig = plot_residuals(residuals)
        assert fig is not None

    def test_save_residuals(self, daily_series):
        residuals = pd.Series(np.random.normal(0, 1, len(daily_series)), index=daily_series.index)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "residuals.png"
            fig = plot_residuals(residuals, save_path=path)
            assert path.exists()


# ── New statistical visualization tests ──────────────────────────────────────


class TestPlotRollingStatistics:
    """Tests for plot_rolling_statistics."""

    def test_basic(self, daily_series):
        fig = plot_rolling_statistics(daily_series, window=7)
        assert fig is not None

    def test_save(self, daily_series):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rolling.png"
            fig = plot_rolling_statistics(daily_series, save_path=path)
            assert path.exists()


class TestPlotDistribution:
    """Tests for plot_distribution."""

    def test_basic(self, daily_series):
        fig = plot_distribution(daily_series, title="Dist Test")
        assert fig is not None

    def test_without_stats(self, daily_series):
        fig = plot_distribution(daily_series, show_stats=False)
        assert fig is not None


class TestPlotLagScatter:
    """Tests for plot_lag_scatter."""

    def test_default_lags(self, daily_series):
        fig = plot_lag_scatter(daily_series)
        assert fig is not None

    def test_custom_lags(self, daily_series):
        fig = plot_lag_scatter(daily_series, lags=[1, 3])
        assert fig is not None


class TestPlotBoxplotByPeriod:
    """Tests for plot_boxplot_by_period."""

    def test_month_boxplot(self, monthly_series):
        fig = plot_boxplot_by_period(monthly_series, period="month")
        assert fig is not None

    def test_dayofweek_boxplot(self, daily_series):
        fig = plot_boxplot_by_period(daily_series, period="dayofweek")
        assert fig is not None


class TestPlotCorrelationHeatmap:
    """Tests for plot_correlation_heatmap."""

    def test_basic(self):
        df = pd.DataFrame(np.random.randn(100, 4), columns=["A", "B", "C", "D"])
        fig = plot_correlation_heatmap(df)
        assert fig is not None


class TestPlotStationaritySummary:
    """Tests for plot_stationarity_summary."""

    def test_basic(self, daily_series):
        fig = plot_stationarity_summary(daily_series, window=7)
        assert fig is not None


class TestPlotModelComparison:
    """Tests for plot_model_comparison."""

    def test_basic(self, daily_series):
        y_test = daily_series.iloc[-20:]
        preds = {
            "model_a": y_test + np.random.normal(0, 1, 20),
            "model_b": y_test + np.random.normal(0, 2, 20),
        }
        metrics = {"model_a": {"mae": 1.0}, "model_b": {"mae": 2.0}}
        fig = plot_model_comparison(y_test, preds, metrics=metrics)
        assert fig is not None


class TestPlotErrorDistribution:
    """Tests for plot_error_distribution."""

    def test_basic(self, daily_series):
        y_test = daily_series.iloc[-20:]
        preds = {"naive": y_test + np.random.normal(0, 1, 20)}
        fig = plot_error_distribution(y_test, preds)
        assert fig is not None


class TestPlotCumulativeError:
    """Tests for plot_cumulative_error."""

    def test_basic(self, daily_series):
        y_test = daily_series.iloc[-20:]
        preds = {
            "m1": y_test + np.random.normal(0, 1, 20),
            "m2": y_test + np.random.normal(0, 0.5, 20),
        }
        fig = plot_cumulative_error(y_test, preds)
        assert fig is not None


class TestPlotValidationSummary:
    """Tests for plot_validation_summary."""

    def test_basic(self, daily_series):
        fig = plot_validation_summary(daily_series)
        assert fig is not None

    def test_with_validation_result(self, daily_series):
        val_dict = {
            "is_valid": True,
            "is_monotonic": True,
            "has_duplicates": False,
        }
        fig = plot_validation_summary(daily_series, validation_result=val_dict)
        assert fig is not None

    def test_with_freq_result(self, daily_series):
        freq_dict = {"inferred_freq": "D", "is_regular": True}
        fig = plot_validation_summary(daily_series, freq_result=freq_dict)
        assert fig is not None

    def test_save_to_file(self, daily_series):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "validation_summary.png"
            fig = plot_validation_summary(daily_series, save_path=path)
            assert path.exists()


class TestPlotMissingData:
    """Tests for plot_missing_data."""

    def test_with_gaps(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.Series(np.random.normal(50, 5, 100), index=idx, name="value")
        # Insert gaps
        data.iloc[20:25] = np.nan
        data.iloc[60:65] = np.nan
        fig = plot_missing_data(data)
        assert fig is not None

    def test_without_gaps(self, daily_series):
        fig = plot_missing_data(daily_series)
        assert fig is not None

    def test_save_to_file(self):
        idx = pd.date_range("2023-01-01", periods=50, freq="D")
        data = pd.Series(np.random.normal(10, 2, 50), index=idx, name="value")
        data.iloc[10:15] = np.nan
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing_data.png"
            fig = plot_missing_data(data, save_path=path)
            assert path.exists()

