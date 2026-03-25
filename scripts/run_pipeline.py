#!/usr/bin/env python3
"""Thin orchestrator for the full multi-time pipeline with output and visualization.

Delegates to MultiTimePipeline for: validate → describe → test → transform →
forecast → evaluate. Saves report JSON and produces summary visualizations.

Usage:
    python scripts/run_pipeline.py --input data.csv --config config.yaml
    python scripts/run_pipeline.py -i data.csv --test-size 30 -o output/pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from multi_time.config import MultiTimeConfig, load_config, setup_logging, get_logger
from multi_time.data import load_csv_series
from multi_time.pipeline import MultiTimePipeline

logger = get_logger(__name__)

# Visualization imports (optional)
try:
    from multi_time.visualization import (
        plot_series,
        plot_forecast,
        plot_diagnostics,
        plot_model_comparison,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full multi-time analysis pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to CSV file")
    parser.add_argument("--config", "-c", help="Path to YAML config file")
    parser.add_argument("--column", help="Column name")
    parser.add_argument("--test-size", type=int, help="Hold-out test size")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger.info("=== Multi-Time Full Pipeline ===")

    config = load_config(args.config) if args.config else MultiTimeConfig()
    config.output_dir = args.output_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_csv_series(args.input, column=args.column)
    logger.info("Loaded %d observations from %s", len(data), args.input)

    if args.test_size and args.test_size < len(data):
        y_train = data.iloc[: -args.test_size]
        y_test = data.iloc[-args.test_size:]
        logger.info("Train/test split: %d / %d", len(y_train), len(y_test))
    else:
        y_train = data
        y_test = None

    pipeline = MultiTimePipeline(config=config)
    result = pipeline.run(y_train, y_test)

    # ── VISUALIZE ───────────────────────────────────────────────────────────────
    n_plots = 0
    if HAS_VIZ and not args.no_plots:
        # Training data diagnostics
        plot_diagnostics(y_train, title="Training Data",
                         save_path=out_dir / "diagnostics.png")
        n_plots += 1

        # Series overview
        plot_series(y_train, title="Training Series",
                    save_path=out_dir / "series.png")
        n_plots += 1

        # Forecast plots (if forecast results available)
        if result.forecast_results:
            first_model = list(result.forecast_results.keys())[0]
            preds_data = result.forecast_results[first_model]
            # Convert to Series if it's a dict
            if isinstance(preds_data, dict) and "predictions" in preds_data:
                preds = pd.Series(preds_data["predictions"]) if not isinstance(preds_data["predictions"], pd.Series) else preds_data["predictions"]
            elif isinstance(preds_data, pd.Series):
                preds = preds_data
            else:
                preds = None

            if preds is not None:
                plot_forecast(
                    y_train, preds, y_test=y_test,
                    title=f"Pipeline Forecast: {first_model}",
                    save_path=out_dir / "forecast.png",
                )
                n_plots += 1
            # Model comparison (if multiple models)
            if len(result.forecast_results) > 1 and y_test is not None:
                try:
                    plot_model_comparison(
                        y_test, result.forecast_results,
                        metrics=result.evaluation_results,
                        title="Pipeline Model Comparison",
                        save_path=out_dir / "model_comparison.png",
                    )
                    n_plots += 1
                except Exception as e:
                    logger.warning("Model comparison plot skipped: %s", e)

        logger.info("[VISUALIZE] Generated %d plots to %s", n_plots, out_dir)

    # ── EXPORT ──────────────────────────────────────────────────────────────────
    summary = {
        "stages_completed": len(result.pipeline_log),
        "validation_passed": result.validation.get("is_valid", None),
        "models_evaluated": list(result.evaluation_results.keys()),
        "evaluation": result.evaluation_results,
        "log": result.pipeline_log,
        "n_plots": n_plots,
    }
    print(json.dumps(summary, indent=2, default=str))

    # Save full pipeline result
    result.save(out_dir / "pipeline_result.json")
    logger.info("Pipeline result saved to %s", out_dir / "pipeline_result.json")

    # Save summary
    report_path = out_dir / "pipeline_summary.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
