# tests/

pytest test suite for the `multi-time` package. Test structure mirrors `src/multi_time/` subpackages.

## Structure

```text
tests/
├── conftest.py          # Shared fixtures (9 data fixtures)
├── config/              # Tests for multi_time.config
├── data/                # Tests for multi_time.data
├── validate/            # Tests for multi_time.validate
├── stats/               # Tests for multi_time.stats
├── transform/           # Tests for multi_time.transform
├── modeling/            # Tests for multi_time.modeling
├── evaluate/            # Tests for multi_time.evaluate
├── visualization/       # Tests for multi_time.visualization
└── pipeline/            # Tests for multi_time.pipeline
```

## Running Tests

```bash
# Full suite
uv run pytest tests/ -v

# Single subpackage
uv run pytest tests/stats/ -v

# With coverage
uv run pytest tests/ --cov=multi_time --cov-report=term-missing
```

## Fixtures (conftest.py)

| Fixture | Description |
| --- | --- |
| `daily_series` | 100 daily points with trend + noise |
| `hourly_series` | 168 hourly points (1 week) with daily seasonality |
| `monthly_series` | 60 monthly points (5 years) with annual seasonality |
| `patchy_series` | 100 daily points with 3 NaN gaps (15 missing) |
| `irregular_series` | 50 irregularly spaced points |
| `stationary_series` | 200 white noise points |
| `nonstationary_series` | 200 random walk points |
| `short_series` | 10 daily points |
| `multivariate_df` | 200-row DataFrame with causal x→y |
