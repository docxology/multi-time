# Scripts

Thin orchestrator scripts that demonstrate all multi-time and sktime functionalities.
Each script contains **zero business logic** — all work is delegated to the `multi_time` library.

## Quick Start

```bash
# Run everything (generates ~103 output files)
uv run python scripts/run_all.py

# Run individual scripts
uv run python scripts/run_synthetic.py --type configurable --n 365 --trend 0.3
uv run python scripts/run_validation.py -i output/synthetic/synthetic_configurable.csv
uv run python scripts/run_descriptive.py -i data.csv --output-dir output/descriptive
uv run python scripts/run_analysis.py -i data.csv --seasonal-period 12 -o output/analysis
uv run python scripts/run_forecast.py -i data.csv --models naive theta --ensemble --horizon 30
uv run python scripts/run_pipeline.py -i data.csv --test-size 30 -o output/pipeline
uv run python scripts/run_end_to_end.py --type configurable --n 500 --models naive theta exp_smoothing
uv run python scripts/run_multi_series.py --n-series 4 --harmonize-freq D --models naive theta
```

## Script Overview

| Script | Purpose | Key Features | Viz Plots |
| --- | --- | --- | --- |
| `run_all.py` | Master orchestrator | Dependency DAG, progress, JSON report | — |
| `run_synthetic.py` | Data generation | 10 generators, multivariate, configurable | 3 |
| `run_validation.py` | Data quality | validate + frequency + patchiness | 3 |
| `run_descriptive.py` | Descriptive stats | ACF/PACF, rolling, decomposition | 11 |
| `run_analysis.py` | Full analysis | 6 stat tests, all viz functions | 14 |
| `run_forecast.py` | Forecasting | Multi-model, ensemble, temporal CV | 7 |
| `run_pipeline.py` | Full pipeline | MultiTimePipeline 6-stage | 4 |
| `run_end_to_end.py` | End-to-end demo | Generate→validate→test→train→viz | 17 |
| `run_multi_series.py` | Multi-series demo | Overlap, correlation, per-series forecast | 24 |

## run_all.py Execution Order

```text
synthetic → validation → descriptive → analysis → forecast → pipeline → end_to_end → multi_series
```

All output goes to `output/<script_name>/` with JSON reports and numbered PNG plots.

## Key CLI Patterns

All scripts support:

- `--output-dir` / `-o` — Output directory
- `--log-level` — DEBUG / INFO / WARNING
- `--no-plots` — Skip visualization (faster execution)
- `--config` / `-c` — YAML configuration file

## Visualization Coverage

The scripts collectively demonstrate all **19 visualization functions**:

1. `plot_series` — Raw time series overview
2. `plot_validation_summary` — 4-panel data quality dashboard
3. `plot_missing_data` — Gap analysis (NaN segments)
4. `plot_diagnostics` — 4-panel diagnostics (histogram, ACF, QQ, series)
5. `plot_acf_pacf` — Autocorrelation and partial autocorrelation
6. `plot_rolling_statistics` — Rolling mean/std with ±2σ bands
7. `plot_distribution` — Histogram + KDE + violin
8. `plot_stationarity_summary` — 4-panel stationarity visual check
9. `plot_lag_scatter` — Scatter at multiple lag values
10. `plot_boxplot_by_period` — Seasonal boxplots (month, day-of-week)
11. `plot_correlation_heatmap` — Lag-based auto-correlation matrix
12. `plot_decomposition` — STL seasonal decomposition (trend + seasonal + residual)
13. `plot_forecast` — Training data + forecast ± prediction intervals
14. `plot_residuals` — Residual analysis (distribution + ACF)
15. `plot_model_comparison` — Multi-model overlay with metrics
16. `plot_error_distribution` — Error histograms per model
17. `plot_cumulative_error` — Cumulative error curves
18. `plot_multi_series_panel` — Stacked panels with shared x-axis and overlap highlighting
19. `plot_series_correlation` — Pairwise cross-correlation matrix between series

## Statistical Tests Used

- **ADF** (Augmented Dickey-Fuller) + **KPSS** — Stationarity
- **Shapiro-Wilk** + **Jarque-Bera** — Normality
- **Seasonal decomposition test** — Seasonality
- **ARCH** (Engle's) — Heteroscedasticity
