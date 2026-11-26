# Reproduction

From the project checkout, install the locked development dependencies
and run the same synthetic example.

```bash
uv sync --locked --python 3.14 --extra dev
MPLBACKEND=Agg uv run --no-sync --python 3.14 python scripts/run_end_to_end.py \
  --type daily --n 90 --seed 42 --models naive --horizon 12 \
  --test-fraction 0.2 -o output/demo
```

The script writes `output/demo/synthetic_data.csv` and
`output/demo/e2e_report.json`, along with diagnostic plots and the forecast
plot. The render procedure copies the report and forecast plot to their
validation locations under `output/`. The lockfile and the command specify
the environment and experiment inputs needed to reproduce the values.
