# 🎯 Module: validate

> [!NOTE]
> Series validation, frequency detection, patchiness analysis, and frequency harmonization logic. Before deep-diving into validation rules, see [Notation and Glossary](../technical/notation.md).

## Files

| File | Purpose |
| --- | --- |
| `validation.py` | `ValidationResult` dataclass, `validate_series()` — type, NaN, index checks |
| `frequency.py` | `detect_frequency()` — `pd.infer_freq` + heuristic fallback |
| `patchiness.py` | `PatchinessResult` dataclass, `assess_patchiness()` — gap analysis |
| `harmonize.py` | `harmonize_frequencies()` — resample multiple series to common frequency |
| `validators.py` | **Facade** — re-exports all from the 4 sub-modules |

## Key Dataclasses

### `ValidationResult`

`is_valid`, `n_observations`, `n_missing`, `missing_pct`, `dtype`, `index_type`, `is_monotonic`, `has_duplicates`, `warnings`, `errors`

### `PatchinessResult`

`n_gaps`, `total_missing_periods`, `gap_sizes`, `longest_gap`, `mean_gap_size`, `gap_locations`, `patchiness_score`

Both support `.to_dict()` for JSON export.

## Tests

`tests/validate/test_validation.py` — Tests for validation, frequency detection, patchiness, and harmonization.
