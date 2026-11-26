# Agents

## Overview
This is the root of the `multi-time` package — a comprehensive multi-frequency time series analysis toolkit built on the sktime ecosystem.

## Conventions
- **Python ≥3.11** with type hints throughout
- **uv** for all environment management (`uv sync --locked`, `uv run`)
- **pytest** for testing, mirror `src/` subpackage structure in `tests/`
- **Thin Orchestrator** pattern: scripts in `scripts/` contain zero business logic
- **Registry Pattern**: forecasters, transformers, metrics use extensible registries
- **Configurable**: all behavior driven by `MultiTimeConfig` dataclass

## Key Commands
```bash
uv sync --locked --python 3.14 --extra dev
uv run pytest tests/ -v
uv run python scripts/run_pipeline.py -i data.csv -o output/
```

See the [documentation index](docs/README.md) and [manuscript render guide](docs/rendering.md).

## Structure
- `src/multi_time/` — 9 subpackages (config, data, validate, stats, transform, modeling, evaluate, visualization, pipeline)
- `tests/` — pytest suite mirroring src/ subpackages
- `scripts/` — 9 thin CLI orchestrators
- `docs/` — architecture, API reference, configuration, guides, module docs
- `output/` — default output directory (gitignored)
