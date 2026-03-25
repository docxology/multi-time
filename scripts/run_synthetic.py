#!/usr/bin/env python3
"""Thin orchestrator for generating synthetic time series data with visualization.

Usage:
    python scripts/run_synthetic.py --type daily --n 365 --output output/synthetic.csv
    python scripts/run_synthetic.py --type configurable --n 200 --trend 0.5 --output-dir output/synthetic
    python scripts/run_synthetic.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from multi_time.config import setup_logging, get_logger
from multi_time.data import GENERATOR_REGISTRY, list_generators

logger = get_logger(__name__)

# Visualization imports (optional)
try:
    from multi_time.visualization import (
        plot_series,
        plot_diagnostics,
        plot_distribution,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic time series data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available generators: {', '.join(list_generators())}",
    )
    parser.add_argument("--list", action="store_true", help="List available generators")
    parser.add_argument("--type", "-t", default="daily", help="Generator type")
    parser.add_argument("--n", type=int, default=365, help="Number of observations")
    parser.add_argument("--start", default="2023-01-01", help="Start date")
    parser.add_argument("--freq", default="D", help="Frequency (D, H, W, MS, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--output-dir", default="output", help="Output directory")

    # Configurable series params
    parser.add_argument("--trend", type=float, default=0.1, help="Trend slope")
    parser.add_argument("--seasonal-period", type=int, help="Seasonal period")
    parser.add_argument("--seasonal-amplitude", type=float, default=0.0, help="Seasonal amplitude")
    parser.add_argument("--noise-std", type=float, default=1.0, help="Noise std dev")
    parser.add_argument("--baseline", type=float, default=100.0, help="Baseline level")
    parser.add_argument("--outlier-fraction", type=float, default=0.0, help="Outlier fraction (0-1)")
    parser.add_argument("--gap-fraction", type=float, default=0.0, help="Missing data fraction (0-1)")

    # Multi-series params
    parser.add_argument("--n-series", type=int, default=3, help="Number of series (multivariate)")
    parser.add_argument("--correlation", type=float, default=0.6, help="Series correlation (multivariate)")

    parser.add_argument("--no-plots", action="store_true", help="Skip visualization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)

    if args.list:
        print("Available generators:")
        for name in list_generators():
            fn = GENERATOR_REGISTRY[name]
            doc_line = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {name:20s} — {doc_line}")
        return 0

    logger.info("=== Multi-Time Synthetic Data Generation ===")

    gen_type = args.type
    if gen_type not in GENERATOR_REGISTRY:
        logger.error("Unknown generator: %s. Use --list to see available.", gen_type)
        return 1

    # Build kwargs based on generator type
    kwargs: dict = {"seed": args.seed}
    if gen_type in ("daily", "hourly", "weekly", "monthly"):
        kwargs.update(n=args.n, start=args.start, trend=args.trend, noise_std=args.noise_std)
        if gen_type == "monthly":
            kwargs["seasonal_amplitude"] = args.seasonal_amplitude or 10.0
        elif gen_type == "hourly":
            kwargs["daily_amplitude"] = args.seasonal_amplitude or 5.0
            kwargs["baseline"] = args.baseline
            kwargs.pop("trend", None)
        elif gen_type == "weekly":
            kwargs["annual_amplitude"] = args.seasonal_amplitude or 8.0
    elif gen_type == "patchy":
        kwargs.update(n=args.n, start=args.start)
    elif gen_type == "irregular":
        kwargs.update(n=args.n)
    elif gen_type == "random_walk":
        kwargs.update(
            n=args.n, start=args.start, freq=args.freq,
            drift=args.trend, volatility=args.noise_std,
            initial_value=args.baseline,
        )
    elif gen_type == "multi_seasonal":
        kwargs.update(
            n=args.n, start=args.start, freq=args.freq,
            trend=args.trend, noise_std=args.noise_std,
        )
        if args.seasonal_period:
            kwargs["periods"] = [args.seasonal_period]
            kwargs["amplitudes"] = [args.seasonal_amplitude]
    elif gen_type == "multivariate":
        kwargs.update(
            n=args.n, start=args.start, freq=args.freq,
            n_series=args.n_series, correlation=args.correlation,
        )
    elif gen_type == "configurable":
        kwargs.update(
            n=args.n, start=args.start, freq=args.freq,
            baseline=args.baseline, trend=args.trend,
            seasonal_period=args.seasonal_period,
            seasonal_amplitude=args.seasonal_amplitude,
            noise_std=args.noise_std,
            outlier_fraction=args.outlier_fraction,
            gap_fraction=args.gap_fraction,
        )

    data = GENERATOR_REGISTRY[gen_type](**kwargs)
    if isinstance(data, pd.DataFrame):
        print(f"Generated {gen_type}: {data.shape[0]} rows × {data.shape[1]} columns")
        print(data.describe().to_string())
    else:
        print(f"Generated {gen_type}: {len(data)} observations")
        print(f"  Range: [{data.dropna().min():.2f}, {data.dropna().max():.2f}]")
        print(f"  Mean: {data.dropna().mean():.2f}, Std: {data.dropna().std():.2f}")
        if data.isna().any():
            print(f"  Missing: {data.isna().sum()} ({data.isna().mean() * 100:.1f}%)")

    # Save CSV
    output_path = args.output
    if not output_path:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"synthetic_{gen_type}.csv")

    if isinstance(data, pd.DataFrame):
        data.to_csv(output_path)
    else:
        data.to_csv(output_path, header=True)
    logger.info("Saved to %s", output_path)
    print(f"\nSaved: {output_path}")

    # ── VISUALIZE ───────────────────────────────────────────────────────────────
    n_plots = 0
    if HAS_VIZ and not args.no_plots:
        # Determine plot output dir (sibling to CSV)
        plot_dir = Path(output_path).parent
        plot_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.Series):
            # Series overview
            plot_series(data, title=f"Synthetic: {gen_type} ({len(data)} obs)",
                        save_path=plot_dir / "preview_series.png")
            n_plots += 1

            # Quick diagnostics
            clean = data.dropna()
            if len(clean) > 10:
                plot_diagnostics(clean, title=f"Diagnostics: {gen_type}",
                                 save_path=plot_dir / "preview_diagnostics.png")
                n_plots += 1

                plot_distribution(clean, title=f"Distribution: {gen_type}",
                                  save_path=plot_dir / "preview_distribution.png")
                n_plots += 1
        elif isinstance(data, pd.DataFrame):
            # Plot first column
            first_col = data.iloc[:, 0]
            plot_series(first_col, title=f"Synthetic: {gen_type} (col 0)",
                        save_path=plot_dir / "preview_series.png")
            n_plots += 1

        logger.info("[VISUALIZE] Generated %d preview plots to %s", n_plots, plot_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
