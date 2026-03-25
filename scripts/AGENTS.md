# Agents — Scripts

## Overview

Thin orchestrator scripts for multi-time. Contains zero business logic — all
work is delegated to the `multi_time` library subpackages.

## Conventions

- **Zero Business Logic**: Scripts only parse CLI args and call library functions
- **Configurable**: All params via CLI flags or `--config config.yaml`
- **Logged**: Every stage logged via `get_logger(__name__)`
- **Output**: Each script outputs to `output/<name>/` subfolder
- **Viz-Enabled**: All scripts import visualization functions with `HAS_VIZ` fallback

## File Inventory (9 files)

- `run_all.py` — Master orchestrator with dependency DAG + progress reporting
- `run_synthetic.py` — Synthetic data generation (10 generators + registry)
- `run_validation.py` — Data quality assessment (validate + freq + patchiness)
- `run_descriptive.py` — Descriptive statistics + 11 visualization plots
- `run_analysis.py` — Full analysis (6 stat tests + up to 14 viz plots)
- `run_forecast.py` — Multi-model forecasting (ensemble, temporal CV, 7 viz)
- `run_pipeline.py` — Full MultiTimePipeline (6-stage) + 4 viz + JSON output
- `run_end_to_end.py` — Complete demo (generate → all 17 viz, temporal CV)
- `run_multi_series.py` — Multi-series coordination (generation, harmonization, cross-correlation, individual forecasting, 24 viz plots)

## Key Patterns

- `HAS_VIZ` guard for optional matplotlib imports
- `GENERATOR_REGISTRY` / `FORECASTER_REGISTRY` for dynamic discovery
- `argparse` for all CLI, `argparse.RawDescriptionHelpFormatter` for epilog
- `json.dump(..., default=str)` for serializing report dicts with dates/NaN
