# 🎯 Agents — Documentation

> [!NOTE]
> Documentation hub for `multi-time` — a modular time series analysis toolkit. See [Notation and Glossary](technical/notation.md) for standard structural terminology.

## Structure

- Root docs: `api_reference.md`, `architecture.md`, `configuration.md`, `examples.md`
- `modules/` — Per-subpackage docs (9 files mapping to src/ subpackages)
- `technical/` — Deep dives (sktime integration, time series theory, schema/API contracts)
- `guides/` — How-to guides (data loading, visualization, modeling, CLI scripts)

## Conventions

- Every module doc maps directly to a `src/multi_time/<name>/` subpackage
- File tables list all sub-module files including facades
- Test references point to the corresponding `tests/<name>/` directory
- CLI examples use `uv run python scripts/` prefix
