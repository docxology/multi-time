#!/usr/bin/env python3
"""Thin orchestrator for time series data validation with visualization.

Usage:
    python scripts/run_validation.py --input data.csv [--config config.yaml]
    python scripts/run_validation.py -i data.csv --output-dir output/validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from multi_time.config import MultiTimeConfig, load_config, setup_logging, get_logger
from multi_time.data import load_csv_series
from multi_time.validate import validate_series, detect_frequency, assess_patchiness

logger = get_logger(__name__)

# Visualization imports (optional)
try:
    from multi_time.visualization import (
        plot_validation_summary,
        plot_missing_data,
        plot_series,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate time series data")
    parser.add_argument("--input", "-i", required=True, help="Path to CSV file")
    parser.add_argument("--config", "-c", help="Path to YAML config file")
    parser.add_argument("--column", help="Column name to analyze (default: first numeric)")
    parser.add_argument("--output", "-o", help="Path to save JSON results")
    parser.add_argument("--output-dir", help="Output directory for plots and results")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger.info("=== Multi-Time Validation ===")

    config = load_config(args.config) if args.config else MultiTimeConfig()
    data = load_csv_series(args.input, column=args.column)
    logger.info("Loaded %d observations from %s", len(data), args.input)

    # ── VALIDATE ────────────────────────────────────────────────────────────────
    validation = validate_series(data)
    frequency = detect_frequency(data)
    patchiness = assess_patchiness(data)

    logger.info(
        "[VALIDATE] valid=%s, freq=%s, missing=%.1f%%, gaps=%d",
        validation.is_valid, frequency["inferred_freq"],
        validation.missing_pct, patchiness.n_gaps,
    )

    results = {
        "validation": validation.to_dict(),
        "frequency": {k: str(v) for k, v in frequency.items()},
        "patchiness": patchiness.to_dict(),
    }

    # ── VISUALIZE ───────────────────────────────────────────────────────────────
    n_plots = 0
    if HAS_VIZ and not args.no_plots:
        out_dir = Path(args.output_dir) if args.output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

            # Validation summary dashboard
            plot_validation_summary(
                data,
                validation_result=validation.to_dict(),
                freq_result=frequency,
                title="Data Quality Summary",
                save_path=out_dir / "validation_summary.png",
            )
            n_plots += 1

            # Missing data analysis
            if data.isna().any():
                plot_missing_data(data, save_path=out_dir / "missing_data.png")
                n_plots += 1

            # Raw series overview
            plot_series(data, title="Raw Series", save_path=out_dir / "raw_series.png")
            n_plots += 1

            logger.info("[VISUALIZE] Generated %d plots to %s", n_plots, out_dir)

    # ── EXPORT ──────────────────────────────────────────────────────────────────
    results["n_plots"] = n_plots
    print(json.dumps(results, indent=2, default=str))

    json_path = args.output
    if not json_path and args.output_dir:
        json_path = str(Path(args.output_dir) / "validation_results.json")
    if json_path:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", json_path)

    return 0 if validation.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
