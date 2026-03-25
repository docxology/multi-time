# Guide: CLI# 🎯 Scripts Reference

> [!NOTE]
> Reference for the 8 thin CLI orchestrator scripts within `multi-time`. For underlying terminology, refer strictly to [Notation and Glossary](../technical/notation.md).

## Master Script

### `run_all.py`

Runs all orchestrators sequentially, each to its own `output/` subfolder.

```bash
uv run python scripts/run_all.py                    # Run all
uv run python scripts/run_all.py --skip forecast     # Skip forecast
uv run python scripts/run_all.py --only synthetic multi_series  # Selected only
uv run python scripts/run_all.py --dry-run           # Print commands without running
uv run python scripts/run_all.py -o output --log-level DEBUG
```

**Output**: Per-script subfolders + `output/run_all_report.json` master report.

---

## Generation

### `run_synthetic.py`

```bash
# List available generators
uv run python scripts/run_synthetic.py --list

# Daily series
uv run python scripts/run_synthetic.py --type daily --n 365 -o output/daily.csv

# Configurable complex series
uv run python scripts/run_synthetic.py --type configurable --n 500 \
    --trend 0.3 --seasonal-period 12 --seasonal-amplitude 8 \
    --noise-std 2.0 --gap-fraction 0.05 --outlier-fraction 0.02

# Random walk
uv run python scripts/run_synthetic.py --type random_walk --n 200 --drift 0.5
```

| Flag | Default | Description |
| --- | --- | --- |
| `--type` | `daily` | Generator name (see `--list`) |
| `--n` | `365` | Number of observations |
| `--start` | `2023-01-01` | Start date |
| `--seed` | `42` | Random seed |
| `--output` | stdout | CSV output path |
| `--trend` | `0.1` | Trend slope |
| `--seasonal-period` | auto | Seasonal period |
| `--seasonal-amplitude` | `0.0` | Seasonal amplitude |
| `--noise-std` | `2.0` | Noise standard deviation |
| `--gap-fraction` | `0.0` | Fraction of missing values |
| `--outlier-fraction` | `0.0` | Fraction of outliers |

---

## Analysis

### `run_validation.py`

```bash
uv run python scripts/run_validation.py -i data.csv -o validation.json
uv run python scripts/run_validation.py -i data.csv --column temperature
```

### `run_descriptive.py`

```bash
uv run python scripts/run_descriptive.py -i data.csv -o stats.json
uv run python scripts/run_descriptive.py -i data.csv --nlags 30 --window 12
```

---

## Forecasting

### `run_forecast.py`

```bash
uv run python scripts/run_forecast.py -i data.csv -m theta -H 30 --test-size 30
uv run python scripts/run_forecast.py -i data.csv -m auto_arima -H 24 -o forecast.json
```

Auto-imputes NaN and infers `DatetimeIndex.freq` before fitting.

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `naive` | Forecaster name |
| `--horizon` | `12` | Forecast steps |
| `--test-size` | none | Hold-out for evaluation |

---

## Pipeline

### `run_pipeline.py`

```bash
uv run python scripts/run_pipeline.py -i data.csv -o output/
uv run python scripts/run_pipeline.py -i data.csv -c config.yaml --output-dir output/
```

### `run_end_to_end.py`

Full pipeline: generate → validate → stats → train → evaluate → visualize.

```bash
uv run python scripts/run_end_to_end.py --type daily --n 365 \
    --models naive theta exp_smoothing --ensemble \
    --horizon 30 -o output/e2e

uv run python scripts/run_end_to_end.py --type configurable --n 500 \
    --trend 0.3 --seasonal-period 12 --seasonal-amplitude 8 \
    --models naive theta --no-plots -o output/e2e
```

| Flag | Default | Description |
| --- | --- | --- |
| `--type` | `daily` | Generator name |
| `--models` | `naive theta` | Space-separated model names |
| `--ensemble` | off | Also run ensemble of models |
| `--no-plots` | off | Skip visualization |
| `--test-fraction` | `0.15` | Test set fraction |

---

## Multi-Series Coordination

### `run_multi_series.py`

Generates multiple series with different frequencies and overlaps, harmonizes them, computes cross-correlations, and forecasts each independently.

```bash
uv run python scripts/run_multi_series.py -o output/multi
uv run python scripts/run_multi_series.py --n-series 4 --harmonize-freq D --models naive theta
```

| Flag | Default | Description |
| --- | --- | --- |
| `--n-series` | `4` | Number of series to generate |
| `--harmonize-freq` | `D` | Frequency to harmonize to (D, W, MS, etc.) |
| `--models` | `naive theta` | Space-separated model names |
| `--horizon` | `14` | Forecast horizon |
| `--test-fraction` | `0.15` | Test set fraction |
