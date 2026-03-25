#!/usr/bin/env python3
"""Thin orchestrator for descriptive time series analysis with visualization.

Demonstrates: summarize_series, ACF/PACF, rolling statistics, decomposition,
distribution analysis, stationarity summary, lag scatter, and boxplot-by-period.

Usage:
    python scripts/run_descriptive.py --input data.csv [--config config.yaml]
    python scripts/run_descriptive.py -i data.csv --output-dir output/descriptive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from multi_time.config import MultiTimeConfig, load_config, setup_logging, get_logger
from multi_time.data import load_csv_series
from multi_time.stats import summarize_series, compute_seasonal_decomposition
from multi_time.transform import build_transform_pipeline, apply_transform

logger = get_logger(__name__)

# Visualization imports (optional — all descriptive-relevant functions)
try:
    from multi_time.visualization import (
        plot_series,
        plot_diagnostics,
        plot_acf_pacf,
        plot_rolling_statistics,
        plot_distribution,
        plot_decomposition,
        plot_stationarity_summary,
        plot_lag_scatter,
        plot_boxplot_by_period,
        plot_validation_summary,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descriptive time series analysis")
    parser.add_argument("--input", "-i", required=True, help="Path to CSV file")
    parser.add_argument("--config", "-c", help="Path to YAML config file")
    parser.add_argument("--column", help="Column name to analyze")
    parser.add_argument("--output", "-o", help="Path to save JSON results")
    parser.add_argument("--output-dir", help="Output directory for plots and results")
    parser.add_argument("--nlags", type=int, default=40, help="Number of ACF/PACF lags")
    parser.add_argument("--rolling-window", type=int, default=12, help="Rolling window size")
    parser.add_argument("--seasonal-period", "-s", type=int, default=7, help="Seasonal period")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger.info("=== Multi-Time Descriptive Analysis ===")

    config = load_config(args.config) if args.config else MultiTimeConfig()
    data = load_csv_series(args.input, column=args.column)
    logger.info("Loaded %d observations from %s", len(data), args.input)

    # Impute if needed for analysis
    if data.isna().any():
        pipeline = build_transform_pipeline(["impute"])
        clean = apply_transform(pipeline, data)
        logger.info("Imputed %d missing values for analysis", data.isna().sum())
    else:
        clean = data

    nlags = min(args.nlags, len(clean) // 3)
    summary = summarize_series(clean, nlags=nlags, rolling_window=args.rolling_window)
    stats = summary["descriptive"]

    logger.info(
        "[DESCRIBE] mean=%.2f, std=%.2f, skew=%.4f, kurt=%.4f",
        stats["mean"], stats["std"], stats["skewness"], stats["kurtosis"],
    )

    # ── VISUALIZE ───────────────────────────────────────────────────────────────
    n_plots = 0
    if HAS_VIZ and not args.no_plots:
        out_dir = Path(args.output_dir) if args.output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1. Series overview
            plot_series(clean, title="Time Series", save_path=out_dir / "series.png")
            n_plots += 1

            # 2. 4-panel diagnostics
            plot_diagnostics(clean, title="Diagnostics", save_path=out_dir / "diagnostics.png")
            n_plots += 1

            # 3. ACF/PACF
            acf_data = summary["acf_pacf"]
            plot_acf_pacf(
                acf_data["acf_values"], acf_data["pacf_values"],
                nlags=acf_data["nlags"], save_path=out_dir / "acf_pacf.png",
            )
            n_plots += 1

            # 4. Rolling statistics
            plot_rolling_statistics(
                clean, window=args.rolling_window,
                title=f"Rolling Statistics (window={args.rolling_window})",
                save_path=out_dir / "rolling_statistics.png",
            )
            n_plots += 1

            # 5. Distribution analysis
            plot_distribution(clean, title="Distribution Analysis",
                              save_path=out_dir / "distribution.png")
            n_plots += 1

            # 6. Stationarity visual check (4-panel)
            plot_stationarity_summary(
                clean, window=args.rolling_window,
                save_path=out_dir / "stationarity.png",
            )
            n_plots += 1

            # 7. Lag scatter
            sp = args.seasonal_period
            max_lag = min(max(sp, 12), len(clean) // 4)
            lags = sorted(set([1, sp, max_lag]))
            plot_lag_scatter(clean, lags=lags, save_path=out_dir / "lag_scatter.png")
            n_plots += 1

            # 8. Boxplot by period
            if isinstance(clean.index, pd.DatetimeIndex):
                for period in ["month", "dayofweek"]:
                    try:
                        plot_boxplot_by_period(
                            clean, period=period,
                            title=f"Distribution by {period}",
                            save_path=out_dir / f"boxplot_{period}.png",
                        )
                        n_plots += 1
                    except Exception as e:
                        logger.warning("Boxplot by %s skipped: %s", period, e)

            # 9. Seasonal decomposition
            if len(clean) >= sp * 2:
                try:
                    decomp = compute_seasonal_decomposition(clean, period=sp)
                    plot_decomposition(decomp, save_path=out_dir / "decomposition.png")
                    n_plots += 1
                except Exception as e:
                    logger.warning("Decomposition skipped: %s", e)

            # 10. Validation summary (data quality)
            from multi_time.validate import validate_series, detect_frequency
            val = validate_series(data)
            freq = detect_frequency(data)
            plot_validation_summary(
                data, validation_result=val.to_dict(), freq_result=freq,
                save_path=out_dir / "validation_summary.png",
            )
            n_plots += 1

            logger.info("[VISUALIZE] Generated %d plots to %s", n_plots, out_dir)

    # ── EXPORT ──────────────────────────────────────────────────────────────────
    summary["n_plots"] = n_plots
    print(json.dumps(summary, indent=2, default=str))

    json_path = args.output
    if not json_path and args.output_dir:
        json_path = str(Path(args.output_dir) / "descriptive_results.json")
    if json_path:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Results saved to %s", json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
