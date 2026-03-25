#!/usr/bin/env python3
"""Thin orchestrator for multi-series coordination and analysis.

Generates multiple synthetic time series at different frequencies and partially
overlapping intervals, harmonizes them, performs cross-series analysis, trains
independent and comparative models, and produces comprehensive visualizations.

Demonstrates: GENERATOR_REGISTRY, harmonize_frequencies, multi-series panel
plots, cross-correlation, per-series forecasting, and comparative evaluation.

Usage:
    python scripts/run_multi_series.py -o output/multi_series
    python scripts/run_multi_series.py --n-series 4 --overlap 0.5 --models naive theta
    python scripts/run_multi_series.py --harmonize-freq D --seed 123
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from multi_time.config import setup_logging, get_logger
from multi_time.data import GENERATOR_REGISTRY
from multi_time.validate import validate_series, detect_frequency, assess_patchiness, harmonize_frequencies
from multi_time.stats import summarize_series, test_stationarity, test_normality
from multi_time.transform import build_transform_pipeline, apply_transform
from multi_time.modeling import create_forecaster, run_forecast, FORECASTER_REGISTRY
from multi_time.evaluate import evaluate_forecast

logger = get_logger(__name__)

# Visualization imports (optional)
try:
    from multi_time.visualization import (
        plot_series,
        plot_multi_series_panel,
        plot_series_correlation,
        plot_diagnostics,
        plot_forecast,
        plot_model_comparison,
        plot_acf_pacf,
        plot_correlation_heatmap,
        plot_distribution,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# ── Series generation definitions ───────────────────────────────────────────
# Each spec defines a generator type, its kwargs, intended start date offset,
# and frequency. Different frequencies + overlapping dates = realistic scenario.
DEFAULT_SERIES_SPECS = [
    {
        "name": "daily_sales",
        "generator": "configurable",
        "freq": "D",
        "kwargs": {
            "n": 365, "start": "2023-01-01", "freq": "D",
            "baseline": 200, "trend": 0.5, "seasonal_period": 7,
            "seasonal_amplitude": 30, "noise_std": 10, "gap_fraction": 0.02,
        },
    },
    {
        "name": "hourly_traffic",
        "generator": "hourly",
        "freq": "H",
        "kwargs": {
            "n": 2160, "start": "2023-03-15",  # 90 days, starts later
            "daily_amplitude": 15, "baseline": 500, "noise_std": 20,
        },
    },
    {
        "name": "weekly_revenue",
        "generator": "weekly",
        "freq": "W",
        "kwargs": {
            "n": 78, "start": "2022-06-01",  # 78 weeks, starts earlier
            "trend": 1.5, "annual_amplitude": 50, "noise_std": 8,
        },
    },
    {
        "name": "monthly_metrics",
        "generator": "monthly",
        "freq": "MS",
        "kwargs": {
            "n": 36, "start": "2022-01-01",  # 3 years, longest span
            "seasonal_amplitude": 15, "trend": 0.8, "noise_std": 3,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-series coordination: generate, harmonize, analyze, forecast, compare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-series", type=int, default=None,
                        help="Number of series to generate (default: 4 predefined)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--harmonize-freq", default="D",
                        help="Target frequency for harmonization (D, W, MS, etc.)")
    parser.add_argument("--models", "-m", nargs="*", default=["naive", "theta"],
                        help=f"Forecaster names: {list(FORECASTER_REGISTRY.keys())}")
    parser.add_argument("--horizon", "-H", type=int, default=14, help="Forecast horizon")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--output-dir", "-o", default="output/multi_series", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def _generate_series(specs: list[dict], seed: int) -> dict[str, pd.Series]:
    """Generate all series from specs."""
    series_dict: dict[str, pd.Series] = {}
    for spec in specs:
        gen_fn = GENERATOR_REGISTRY[spec["generator"]]
        kwargs = {**spec["kwargs"], "seed": seed}
        s = gen_fn(**kwargs)
        # Handle multivariate -> take first column
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = spec["name"]
        series_dict[spec["name"]] = s
        seed += 1  # Different seed per series
    return series_dict


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger.info("=== Multi-Time Multi-Series Coordination ===")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. GENERATE ─────────────────────────────────────────────────────────────
    if args.n_series and args.n_series != len(DEFAULT_SERIES_SPECS):
        # Generate N configurable series with varying frequencies
        freqs = ["D", "H", "W", "MS", "D", "H"]
        specs = []
        for i in range(args.n_series):
            freq = freqs[i % len(freqs)]
            n = {"D": 365, "H": 2160, "W": 78, "MS": 36}[freq]
            offset_days = i * 30  # Stagger starts for partial overlap
            start = pd.Timestamp("2023-01-01") + pd.Timedelta(days=offset_days)
            specs.append({
                "name": f"series_{i}_{freq}",
                "generator": "configurable",
                "freq": freq,
                "kwargs": {
                    "n": n, "start": str(start.date()), "freq": freq,
                    "baseline": 100 + i * 50, "trend": 0.1 * (i + 1),
                    "noise_std": 5 + i * 2,
                },
            })
    else:
        specs = DEFAULT_SERIES_SPECS

    raw_series = _generate_series(specs, args.seed)
    logger.info("[GENERATE] Created %d series", len(raw_series))

    for name, s in raw_series.items():
        info = f"n={len(s)}, range=[{s.index.min()}, {s.index.max()}]"
        if s.isna().any():
            info += f", missing={s.isna().sum()}"
        logger.info("  %s: %s", name, info)

    # Save raw data
    for name, s in raw_series.items():
        s.to_csv(out_dir / f"{name}.csv", header=True)
    logger.info("Saved %d CSV files to %s", len(raw_series), out_dir)

    # ── 2. VALIDATE ─────────────────────────────────────────────────────────────
    validation_results = {}
    for name, s in raw_series.items():
        val = validate_series(s)
        freq = detect_frequency(s)
        patchiness = assess_patchiness(s)
        validation_results[name] = {
            "is_valid": val.is_valid,
            "freq": freq["inferred_freq"],
            "n": len(s),
            "missing_pct": val.missing_pct,
            "gaps": patchiness.n_gaps,
        }
        logger.info("[VALIDATE] %s: valid=%s, freq=%s, missing=%.1f%%",
                    name, val.is_valid, freq["inferred_freq"], val.missing_pct)

    # ── 3. HARMONIZE ────────────────────────────────────────────────────────────
    # Clean NaN before harmonization
    clean_series = {}
    for name, s in raw_series.items():
        if s.isna().any():
            pipeline = build_transform_pipeline(["impute"])
            clean_series[name] = apply_transform(pipeline, s)
        else:
            clean_series[name] = s

    harmonized = harmonize_frequencies(
        list(clean_series.values()),
        target_freq=args.harmonize_freq,
    )
    harmonized_dict = {name: h for name, h in zip(clean_series.keys(), harmonized)}
    logger.info("[HARMONIZE] Resampled %d series to freq=%s", len(harmonized), args.harmonize_freq)

    for name, h in harmonized_dict.items():
        logger.info("  %s: %d → %d points", name, len(clean_series[name]), len(h))

    # ── 4. CROSS-SERIES ANALYSIS ────────────────────────────────────────────────
    # Build aligned DataFrame for pairwise analysis
    aligned_df = pd.DataFrame(harmonized_dict).dropna()
    logger.info("[ANALYSIS] %d overlapping observations across %d series", len(aligned_df), len(harmonized_dict))

    # Cross-correlation
    cross_corr = aligned_df.corr(method="pearson")
    logger.info("[ANALYSIS] Cross-correlation matrix computed (%dx%d)", *cross_corr.shape)

    # Per-series summaries
    series_summaries = {}
    for name, h in harmonized_dict.items():
        clean = h.dropna()
        if len(clean) > 10:
            nlags = min(20, len(clean) // 3)
            summary = summarize_series(clean, nlags=nlags)
            series_summaries[name] = summary["descriptive"]
            logger.info("  %s: mean=%.2f, std=%.2f",
                        name, summary["descriptive"]["mean"], summary["descriptive"]["std"])

    # ── 5. FORECAST EACH SERIES ─────────────────────────────────────────────────
    forecast_results: dict[str, dict] = {}
    all_forecasts: dict[str, dict[str, pd.Series]] = {}

    for name, h in harmonized_dict.items():
        clean = h.dropna()
        if len(clean) < 30:
            logger.warning("[FORECAST] %s: too short (%d pts), skipped", name, len(clean))
            continue

        test_size = max(1, min(args.horizon, int(len(clean) * args.test_fraction)))
        y_train = clean.iloc[:-test_size]
        y_test = clean.iloc[-test_size:]
        fh = len(y_test)

        series_preds = {}
        series_metrics = {}
        for model_name in args.models:
            try:
                f = create_forecaster(model_name)
                preds = run_forecast(f, y_train, fh=fh)
                metrics = evaluate_forecast(y_test, preds, metrics_list=["mae", "mape"])
                series_preds[model_name] = preds
                series_metrics[model_name] = metrics
                logger.info("[FORECAST] %s/%s: MAE=%.4f", name, model_name, metrics.get("mae", 0))
            except Exception as e:
                logger.warning("[FORECAST] %s/%s failed: %s", name, model_name, e)
                series_metrics[model_name] = {"error": str(e)}

        forecast_results[name] = {
            "metrics": series_metrics,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        all_forecasts[name] = series_preds

    # ── 6. VISUALIZE ────────────────────────────────────────────────────────────
    n_plots = 0
    if HAS_VIZ and not args.no_plots:
        logger.info("[VISUALIZE] Generating multi-series plots...")

        # 1. Multi-series panel (each series in its own panel, overlap highlighted)
        plot_multi_series_panel(
            raw_series, title="Multi-Series Panel (Raw)",
            save_path=out_dir / "01_multi_series_panel.png",
        )
        n_plots += 1

        # 2. Harmonized overlay (all on same axes after resampling)
        plot_series(
            *harmonized_dict.values(),
            labels=list(harmonized_dict.keys()),
            title=f"Harmonized Series (freq={args.harmonize_freq})",
            save_path=out_dir / "02_harmonized_overlay.png",
        )
        n_plots += 1

        # 3. Cross-correlation heatmap
        if len(aligned_df) > 3:
            plot_series_correlation(
                harmonized_dict, title="Cross-Series Correlation",
                save_path=out_dir / "03_cross_correlation.png",
            )
            n_plots += 1

        # 4. Correlation heatmap (on aligned + lagged data)
        if len(aligned_df) > 10:
            plot_correlation_heatmap(
                aligned_df, title="Aligned Series Correlation",
                save_path=out_dir / "04_correlation_heatmap.png",
            )
            n_plots += 1

        # 5. Per-series diagnostics
        for name, h in harmonized_dict.items():
            clean = h.dropna()
            if len(clean) > 20:
                plot_diagnostics(
                    clean, title=f"Diagnostics: {name}",
                    save_path=out_dir / f"05_diagnostics_{name}.png",
                )
                n_plots += 1

        # 6. Per-series distribution
        for name, h in harmonized_dict.items():
            clean = h.dropna()
            if len(clean) > 20:
                plot_distribution(
                    clean, title=f"Distribution: {name}",
                    save_path=out_dir / f"06_distribution_{name}.png",
                )
                n_plots += 1

        # 7. Per-series ACF/PACF
        for name, summ in series_summaries.items():
            try:
                h = harmonized_dict[name].dropna()
                nlags = min(20, len(h) // 3)
                from multi_time.stats import compute_acf_pacf
                acf_data = compute_acf_pacf(h, nlags=nlags)
                plot_acf_pacf(
                    acf_data["acf_values"], acf_data["pacf_values"],
                    nlags=acf_data["nlags"],
                    save_path=out_dir / f"07_acf_pacf_{name}.png",
                )
                n_plots += 1
            except Exception:
                pass

        # 8. Forecast plots per series
        for name, preds_dict in all_forecasts.items():
            if not preds_dict:
                continue
            h = harmonized_dict[name].dropna()
            test_size = max(1, min(args.horizon, int(len(h) * args.test_fraction)))
            y_train = h.iloc[:-test_size]
            y_test = h.iloc[-test_size:]
            best_model = min(
                (k for k in preds_dict if k in forecast_results[name]["metrics"]
                 and "error" not in forecast_results[name]["metrics"][k]),
                key=lambda k: forecast_results[name]["metrics"][k].get("mae", float("inf")),
                default=None,
            )
            if best_model:
                plot_forecast(
                    y_train, preds_dict[best_model], y_test=y_test,
                    title=f"Forecast: {name} ({best_model})",
                    save_path=out_dir / f"08_forecast_{name}.png",
                )
                n_plots += 1

        # 9. Model comparison (if multiple models and multiple series)
        if len(args.models) > 1:
            for name, preds_dict in all_forecasts.items():
                if len(preds_dict) > 1:
                    h = harmonized_dict[name].dropna()
                    test_size = max(1, min(args.horizon, int(len(h) * args.test_fraction)))
                    y_test = h.iloc[-test_size:]
                    try:
                        plot_model_comparison(
                            y_test, preds_dict,
                            metrics=forecast_results[name]["metrics"],
                            title=f"Model Comparison: {name}",
                            save_path=out_dir / f"09_comparison_{name}.png",
                        )
                        n_plots += 1
                    except Exception:
                        pass

        logger.info("[VISUALIZE] Generated %d plots to %s", n_plots, out_dir)

    # ── 7. EXPORT ───────────────────────────────────────────────────────────────
    report = {
        "n_series": len(raw_series),
        "series_specs": [
            {"name": s["name"], "generator": s["generator"], "freq": s["freq"]}
            for s in specs
        ],
        "harmonize_freq": args.harmonize_freq,
        "n_overlapping": len(aligned_df),
        "validation": validation_results,
        "cross_correlation": cross_corr.to_dict() if len(aligned_df) > 0 else {},
        "per_series_stats": series_summaries,
        "forecast_results": forecast_results,
        "n_plots": n_plots,
    }
    report_path = out_dir / "multi_series_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("[EXPORT] Report saved to %s", report_path)

    # Print summary
    print(f"\n{'='*70}")
    print(f" Multi-Series Coordination Complete")
    print(f"{'='*70}")
    print(f" Series generated: {len(raw_series)}")
    for name, info in validation_results.items():
        print(f"   {name:25s}  freq={info['freq']:5s}  n={info['n']:5d}  missing={info['missing_pct']:.1f}%")
    print(f"\n Harmonized to: {args.harmonize_freq} ({len(aligned_df)} overlapping points)")
    if len(cross_corr) > 0 and len(aligned_df) > 0:
        print(f"\n Cross-Correlation ({len(aligned_df)} overlap pts):")
        for i, name_i in enumerate(cross_corr.columns):
            for j, name_j in enumerate(cross_corr.columns):
                if j > i:
                    print(f"   {name_i} ↔ {name_j}: {cross_corr.iloc[i, j]:.3f}")
    print(f"\n Forecast Results:")
    print(f" {'Series':<25s} {'Model':<15s} {'MAE':>8s} {'MAPE':>8s}")
    print(f" {'-'*25} {'-'*15} {'-'*8} {'-'*8}")
    for name, res in forecast_results.items():
        for model, metrics in res["metrics"].items():
            if "error" in metrics:
                print(f" {name:<25s} {model:<15s} {'ERROR':>8s}")
            else:
                print(f" {name:<25s} {model:<15s} {metrics.get('mae', 0):>8.3f} {metrics.get('mape', 0):>8.3f}")
    print(f"\n Plots: {n_plots}")
    print(f" Output: {out_dir.resolve()}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
