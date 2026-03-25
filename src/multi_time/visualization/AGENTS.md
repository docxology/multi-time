# Agents — Visualization

## Overview

Time series visualization subpackage with **17 plot functions** across 5 sub-modules + 1 facade.

## Files

| File | Functions | Purpose |
| --- | --- | --- |
| `core.py` | `check_matplotlib()`, `save_or_show()` | Agg backend + shared utilities |
| `series.py` | `plot_series()` | Multi-series overlay |
| `forecast.py` | `plot_forecast()`, `plot_residuals()` | Forecast + residual analysis |
| `diagnostics.py` | 3 functions | ACF/PACF, decomposition, diagnostics |
| `statistics.py` | 8 functions | Rolling, distribution, lag, boxplot, heatmap, stationarity, validation, missing |
| `comparison.py` | 3 functions | Model comparison, error distribution, cumulative error |
| `plots.py` | — | Facade re-exporting all 17 functions |

## Conventions

- All functions return `matplotlib.figure.Figure`
- All accept `save_path: str | Path | None` for PNG output
- `save_or_show()` handles saving + `plt.close(fig)` to prevent memory leaks
- Consistent `COLORS` palette across all plots
- Optional dependency: guarded with `try: import matplotlib`
- All functions fully typed with `from __future__ import annotations`
