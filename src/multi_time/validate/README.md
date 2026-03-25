# multi_time.validate

**Validation + Frequency Detection + Patchiness Analysis**

## Exports

| Symbol | Type | Description |
| --- | --- | --- |
| `validate_series` | `(pd.Series) → ValidationResult` | Data quality check |
| `detect_frequency` | `(pd.Series) → dict` | Infer frequency + regularity |
| `assess_patchiness` | `(pd.Series) → PatchinessResult` | Gap analysis |
| `harmonize_frequencies` | `(list[pd.Series], target_freq) → list[pd.Series]` | Resample to common freq |
| `ValidationResult` | `dataclass` | Typed validation output |
| `PatchinessResult` | `dataclass` | Typed patchiness output |

## Schema: ValidationResult

```python
@dataclass
class ValidationResult:
    is_valid: bool               # False if major issues found
    n_observations: int
    n_missing: int
    missing_pct: float           # 0.0 to 100.0
    is_monotonic: bool           # Index strictly increasing
    has_duplicates: bool         # Duplicate timestamps
    dtype: str                   # e.g. 'float64'
    index_type: str              # e.g. 'DatetimeIndex'
    issues: list[str]            # Human-readable issue descriptions

    def to_dict(self) -> dict[str, Any]: ...
```

## Schema: PatchinessResult

```python
@dataclass
class PatchinessResult:
    n_gaps: int                  # Number of contiguous gap regions
    mean_gap_size: float         # Average gap length
    max_gap_size: int            # Largest gap
    gap_positions: list[tuple[int, int]]  # (start_idx, end_idx) pairs
    total_missing: int
    coverage: float              # 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]: ...
```

## Frequency Detection Output

```python
detect_frequency(data) → {
    "inferred_freq": "D",            # pandas offset alias (D, H, W, MS, etc.)
    "is_regular": True,              # All deltas consistent
    "median_delta": Timedelta("1 days"),
    "n_unique_deltas": 1,
}
```

## Usage

```python
from multi_time.validate import validate_series, detect_frequency, assess_patchiness

data = pd.Series(...)
val = validate_series(data)
if not val.is_valid:
    print(f"Issues: {val.issues}")

freq = detect_frequency(data)
print(f"Frequency: {freq['inferred_freq']}, Regular: {freq['is_regular']}")

patchiness = assess_patchiness(data)
print(f"Gaps: {patchiness.n_gaps}, Coverage: {patchiness.coverage:.1%}")
```
