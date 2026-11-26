# Regenerate the software manuscript

The canonical [manuscript](manuscript/README.md) describes one reproducible
synthetic example. Generate its report and plot before rendering.

## Generate the example evidence

From a clean project checkout named `multi-time`:

```bash
uv sync --locked --python 3.14 --extra dev
MPLBACKEND=Agg uv run --no-sync --python 3.14 python scripts/run_end_to_end.py \
  --type daily --n 90 --seed 42 --models naive --horizon 12 \
  --test-fraction 0.2 -o output/demo

mkdir -p output/figures output/reports
cp output/demo/13_forecast.png output/figures/holdout.png
cp output/demo/e2e_report.json output/reports/software-demo.json
cp docs/manuscript/figure_registry.json output/figures/figure_registry.json
```

The synthetic CSV and diagnostic plots stay under `output/demo/`. The
report copied to `output/reports/` provides the source values quoted in the
manuscript. The figure registry binds the manuscript label to the forecast
plot. The disposable `output/` tree is ignored by Git.

## Render and validate

Use Git, uv, Pandoc 3.14, pandoc-crossref 0.3.26 built against Pandoc 3.14,
and XeLaTeX with BibTeX. From the same checkout, register the project with
the pinned renderer and run its two stages.

```bash
project_root="$(pwd -P)"
git clone https://github.com/docxology/template.git ../template-renderer
cd ../template-renderer
git checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
uv sync --locked --no-dev
mkdir -p projects/working
ln -s "$project_root" projects/working/multi-time

uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/multi-time
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/multi-time
```

Look in the project checkout for `output/pdf/multi-time_combined.pdf`
and `output/web/index.html`. Both should include the synthetic
holdout results, the forecast plot, and the bibliography. Stage 04 writes
`output/reports/validation_report.json` and
`output/reports/rendered_provenance.json`. The latter records the source
paths and evidence that the renderer consumed.

See the pinned [project integration guide](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/projects/README.md#L208-L223),
[manuscript resolver](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/infrastructure/core/project_paths.py#L59-L75),
and [render and validation stages](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/RUN_GUIDE.md#L563-L574).
