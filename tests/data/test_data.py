"""Tests for multi_time.data subpackage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    list_generators,
    GENERATOR_REGISTRY,
)


class TestCoreGenerators:
    """Tests for core generator functions."""

    def test_daily_series_defaults(self):
        s = generate_daily_series()
        assert len(s) == 365
        assert s.index.freq == "D"
        assert s.name == "daily"
        assert not s.isna().any()

    def test_daily_series_custom(self):
        s = generate_daily_series(n=50, start="2024-01-01", trend=0.5, seed=99)
        assert len(s) == 50
        assert s.index[0] == pd.Timestamp("2024-01-01")

    def test_hourly_series(self):
        s = generate_hourly_series(n=48)
        assert len(s) == 48
        assert s.name == "hourly"
        assert s.index.freq == "h"

    def test_weekly_series(self):
        s = generate_weekly_series(n=52)
        assert len(s) == 52
        assert s.name == "weekly"

    def test_monthly_series(self):
        s = generate_monthly_series(n=24, seasonal_amplitude=5.0)
        assert len(s) == 24
        assert s.name == "monthly"

    def test_reproducible_seeds(self):
        s1 = generate_daily_series(n=10, seed=42)
        s2 = generate_daily_series(n=10, seed=42)
        pd.testing.assert_series_equal(s1, s2)


class TestSpecialtyGenerators:
    """Tests for specialty generator functions."""

    def test_patchy_series(self):
        s = generate_patchy_series(n=100)
        assert len(s) == 100
        assert s.isna().sum() == 15

    def test_patchy_custom_gaps(self):
        s = generate_patchy_series(n=50, gap_ranges=[(5, 10)])
        assert s.isna().sum() == 5

    def test_irregular_series(self):
        s = generate_irregular_series(n=40, year=2024)
        assert len(s) == 40
        assert s.index[0].year == 2024
        assert pd.infer_freq(s.index) is None

    def test_random_walk(self):
        s = generate_random_walk(n=100, drift=0.1, volatility=2.0)
        assert len(s) == 100
        assert s.name == "random_walk"
        # Non-stationary: variance should grow
        first_half_var = s.iloc[:50].var()
        assert isinstance(first_half_var, float)

    def test_random_walk_different_freq(self):
        s = generate_random_walk(n=24, freq="MS")
        assert s.index.freq == "MS"

    def test_multi_seasonal(self):
        s = generate_multi_seasonal_series(n=168)
        assert len(s) == 168
        assert s.name == "multi_seasonal"

    def test_multi_seasonal_custom_periods(self):
        s = generate_multi_seasonal_series(
            n=365, freq="D", periods=[7, 365], amplitudes=[3.0, 10.0]
        )
        assert len(s) == 365

    def test_multivariate(self):
        df = generate_multivariate_series(n=100, n_series=4)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 4)
        assert list(df.columns) == ["series_1", "series_2", "series_3", "series_4"]

    def test_multivariate_correlation(self):
        df = generate_multivariate_series(n=500, n_series=2, correlation=0.9, seed=42)
        corr = df.corr().iloc[0, 1]
        assert corr > 0.7  # High correlation

    def test_configurable_basic(self):
        s = generate_configurable_series(n=100)
        assert len(s) == 100
        assert s.name == "configurable"

    def test_configurable_with_seasonality(self):
        s = generate_configurable_series(
            n=365, seasonal_period=7, seasonal_amplitude=5.0
        )
        assert len(s) == 365
        assert not s.isna().any()

    def test_configurable_with_outliers(self):
        s = generate_configurable_series(n=200, outlier_fraction=0.1, seed=42)
        assert len(s) == 200

    def test_configurable_with_gaps(self):
        s = generate_configurable_series(n=100, gap_fraction=0.15)
        assert s.isna().sum() == 15

    def test_configurable_all_features(self):
        s = generate_configurable_series(
            n=200, trend=0.5, seasonal_period=12, seasonal_amplitude=3.0,
            noise_std=2.0, outlier_fraction=0.05, gap_fraction=0.1,
        )
        assert len(s) == 200
        assert s.isna().sum() == 20  # 10% of 200


class TestRegistry:
    """Tests for generator registry."""

    def test_registry_keys(self):
        expected = {
            "daily", "hourly", "weekly", "monthly", "patchy", "irregular",
            "random_walk", "multi_seasonal", "multivariate", "configurable",
        }
        assert expected == set(GENERATOR_REGISTRY.keys())

    def test_list_generators(self):
        names = list_generators()
        assert "daily" in names
        assert "configurable" in names
        assert names == sorted(names)

    def test_registry_callable(self):
        for name, fn in GENERATOR_REGISTRY.items():
            assert callable(fn), f"{name} is not callable"


class TestLoaders:
    """Tests for CSV loading functions."""

    @pytest.fixture
    def csv_file(self):
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        df = pd.DataFrame({"value": range(20), "other": range(20, 40)}, index=dates)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name)
            return f.name

    def test_load_csv_series_default(self, csv_file):
        s = load_csv_series(csv_file)
        assert len(s) == 20
        assert isinstance(s, pd.Series)

    def test_load_csv_series_column(self, csv_file):
        s = load_csv_series(csv_file, column="other")
        assert s.iloc[0] == 20

    def test_load_csv_series_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_csv_series("/nonexistent.csv")

    def test_load_csv_series_bad_column(self, csv_file):
        with pytest.raises(ValueError, match="not found"):
            load_csv_series(csv_file, column="no_such_col")

    def test_load_csv_dataframe(self, csv_file):
        df = load_csv_dataframe(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20

    def test_load_csv_dataframe_columns_filter(self, csv_file):
        df = load_csv_dataframe(csv_file, columns=["value"])
        assert list(df.columns) == ["value"]
