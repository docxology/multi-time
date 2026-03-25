"""Tests for multi_time.validate subpackage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_time.validate import (
    validate_series,
    detect_frequency,
    assess_patchiness,
    harmonize_frequencies,
)


class TestValidateSeries:
    """Tests for validate_series."""

    def test_valid_daily_series(self, daily_series):
        result = validate_series(daily_series)
        assert result.is_valid
        assert result.n_observations == 100
        assert result.n_missing == 0
        assert result.is_monotonic

    def test_patchy_series_reports_missing(self, patchy_series):
        result = validate_series(patchy_series)
        assert result.is_valid
        assert result.n_missing == 15
        assert result.missing_pct == pytest.approx(15.0)

    def test_empty_series_invalid(self):
        s = pd.Series([], dtype=float)
        result = validate_series(s)
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_non_numeric_invalid(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        s = pd.Series(["a"] * 10, index=dates)
        result = validate_series(s)
        assert not result.is_valid

    def test_dataframe_single_column(self, daily_series):
        df = daily_series.to_frame()
        result = validate_series(df)
        assert result.is_valid

    def test_dataframe_multi_column_invalid(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = validate_series(df)
        assert not result.is_valid

    def test_non_monotonic_warning(self):
        dates = pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"])
        s = pd.Series([1.0, 2.0, 3.0], index=dates)
        result = validate_series(s)
        assert not result.is_monotonic
        assert any("monotonic" in w.lower() for w in result.warnings)

    def test_duplicate_index_warning(self):
        dates = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"])
        s = pd.Series([1.0, 2.0, 3.0], index=dates)
        result = validate_series(s)
        assert result.has_duplicates

    def test_to_dict(self, daily_series):
        result = validate_series(daily_series)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "is_valid" in d
        assert "n_observations" in d


class TestDetectFrequency:
    """Tests for detect_frequency."""

    def test_daily_frequency(self, daily_series):
        result = detect_frequency(daily_series)
        assert result["inferred_freq"] == "D"
        assert result["is_regular"]

    def test_hourly_frequency(self, hourly_series):
        result = detect_frequency(hourly_series)
        assert result["inferred_freq"] is not None
        assert result["is_regular"]

    def test_monthly_frequency(self, monthly_series):
        result = detect_frequency(monthly_series)
        assert result["inferred_freq"] in ("MS", "M")

    def test_irregular_frequency(self, irregular_series):
        result = detect_frequency(irregular_series)
        assert result["inferred_freq"] is not None
        assert result["median_delta"] is not None


class TestAssessPatchiness:
    """Tests for assess_patchiness."""

    def test_patchy_series_finds_gaps(self, patchy_series):
        result = assess_patchiness(patchy_series)
        assert result.n_gaps == 3
        assert result.longest_gap == 8
        assert result.total_missing_periods == 15
        assert result.patchiness_score > 0

    def test_clean_series_no_gaps(self, daily_series):
        result = assess_patchiness(daily_series)
        assert result.n_gaps == 0
        assert result.patchiness_score == 0.0

    def test_to_dict(self, patchy_series):
        result = assess_patchiness(patchy_series)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "n_gaps" in d


class TestHarmonizeFrequencies:
    """Tests for harmonize_frequencies."""

    def test_harmonize_to_daily(self, daily_series, hourly_series):
        result = harmonize_frequencies([daily_series, hourly_series], target_freq="D")
        assert len(result) == 2
        for s in result:
            freq = pd.infer_freq(s.index)
            assert freq == "D"

    def test_harmonize_preserves_count(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        s = pd.Series(range(10), index=dates, dtype=float)
        result = harmonize_frequencies([s], target_freq="D")
        assert len(result[0]) == 10
