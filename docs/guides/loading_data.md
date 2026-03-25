# 🎯 Guide: Loading Time Series Data

> [!NOTE]
> How to read various kinds of time series data into `multi-time`. Before proceeding, please review the centralized [Notation and Glossary](../technical/notation.md) to familiarize yourself with standard time series terminology.

## From CSV Files

### Simple CSV (date + value)

```csv
date,temperature
2023-01-01,10.5
2023-01-02,11.2
```

```python
from multi_time.data import load_csv_series
s = load_csv_series("temperature.csv")
```

### Multi-column CSV

```csv
date,temp,humidity,wind
2023-01-01,10.5,65,12
```

```python
from multi_time.data import load_csv_series, load_csv_dataframe

# Single column
temp = load_csv_series("weather.csv", column="temp")

# Full DataFrame
df = load_csv_dataframe("weather.csv", columns=["temp", "humidity"])
```

### Different Date Formats

The loader uses pandas `parse_dates=True`, which handles most common formats:

```python
# ISO 8601: 2023-01-01, 2023-01-01T12:00:00
# US format: 01/01/2023
# European: 01-Jan-2023
s = load_csv_series("data.csv")  # Auto-detected
```

### Non-Standard Date Column

```python
# Date column at position 2 (0-indexed)
s = load_csv_series("data.csv", date_column=2)

# Date column by name
s = load_csv_series("data.csv", date_column="timestamp")
```

### Forcing Frequency

```python
# Force daily frequency (fills gaps with NaN)
s = load_csv_series("sparse_data.csv", freq="D")
```

## From Generators (Synthetic Data)

For testing, prototyping, and learning:

```python
from multi_time.data import (
    generate_daily_series,
    generate_monthly_series,
    generate_patchy_series,
    generate_irregular_series,
)

# Daily with trend + noise
daily = generate_daily_series(n=365, trend=0.1, noise_std=2.0, seed=42)

# Monthly with annual seasonality
monthly = generate_monthly_series(n=60, seasonal_amplitude=10.0)

# Patchy (with NaN gaps)
patchy = generate_patchy_series(n=100, gap_ranges=[(10, 20), (50, 55)])

# Irregular spacing
irregular = generate_irregular_series(n=50, year=2024)
```

## After Loading: Validate

Always validate after loading:

```python
from multi_time.validate import validate_series, detect_frequency, assess_patchiness

result = validate_series(data)
print(f"Valid: {result.is_valid}")
print(f"N obs: {result.n_observations}, Missing: {result.missing_pct:.1f}%")

freq = detect_frequency(data)
print(f"Frequency: {freq['inferred_freq']}, Regular: {freq['is_regular']}")

gaps = assess_patchiness(data)
print(f"Gaps: {gaps.n_gaps}, Score: {gaps.patchiness_score:.3f}")
```

## Multi-Frequency Harmonization

When working with series at different frequencies:

```python
from multi_time.validate import harmonize_frequencies

# Resample hourly + daily → common daily
aligned = harmonize_frequencies([hourly, daily], target_freq="D")
```

## CLI Script

```bash
uv run python scripts/run_validation.py -i data.csv --column temperature
```
