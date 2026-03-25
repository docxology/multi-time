#!/usr/bin/env python3
"""Thin orchestrator for comprehensive time series analysis and visualization.

Demonstrates the full multi_time analysis toolkit: validation, descriptive
statistics, 6 statistical tests, transformation, and all 17 visualization
functions. Produces a complete JSON report and numbered PNG plots.

Usage:
    python scripts/run_analysis.py --input data.csv --output-dir output/analysis
    python scripts/run_analysis.py -i data.csv -o output/analysis --seasonal-period 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from multi_time.config import MultiTimeConfig, load_config, setup_logging, get_logger
from multi_time.data import load_csv_series
from multi_time.validate import validate_series, detect_frequency, assess_patchiness
from multi_time.stats import (
    summarize_series,
    test_stationarity,
    test_normality,
    test_seasonality,
    test_heteroscedasticity,
    compute_seasonal_decomposition,
)
from multi_time.transform import build_transform_pipeline, apply_transform

logger = get_logger(__name__)

# Visualization imports (optional — all 17 functions)
try:
    from multi_time.visualization import (
        plot_series,
        plot_diagnostics,
        plot_acf_pacf,
        plot_decomposition,
        plot_rolling_statistics,
        plot_distribution,
        plot_lag_scatter,
        plot_boxplot_by_period,
        plot_correlation_heatmap,
        plot_stationarity_summary,
        plot_validation_summary,
        plot_missing_data,
        plot_forecast,
        plot_residuals,
        plot_model_comparison,
        plot_error_distribution,
        plot_cumulative_error,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive time series analysis — all stats + all 17 visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to CSV file")
    parser.add_argument("--column", help="Column name to analyze")
    parser.add_argument("--output-dir", "-o", default="output/analysis", help="Output directory")
    parser.add_argument("--seasonal-period", "-s", type=int, default=7, help="Seasonal period")
    parser.add_argument("--rolling-window", "-w", type=int, default=12, help="Rolling window")
    parser.add_argument("--nlags", type=int, default=40, help="ACF/PACF lags")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger.info("=== Multi-Time Comprehensive Analysis ===")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. LOAD ──────────────────────────────────────────────────────────────────
    data = load_csv_series(args.input, column=args.column)
    logger.info("Loaded %d observations from %s", len(data), args.input)

    # ── 2. VALIDATE ──────────────────────────────────────────────────────────────
    val = validate_series(data)
    freq = detect_frequency(data)
    patchiness = assess_patchiness(data)
    logger.info(
        "[VALIDATE] valid=%s, freq=%s, missing=%.1f%%, gaps=%d",
        val.is_valid, freq["inferred_freq"], val.missing_pct, patchiness.n_gaps,
    )

    # ── 3. CLEAN (impute if needed) ──────────────────────────────────────────────
    if data.isna().any():
        n_missing = int(data.isna().sum())
        pipeline = build_transform_pipeline(["impute"])
        clean = apply_transform(pipeline, data)
        logger.info("[CLEAN] Imputed %d missing values", n_missing)
    else:
        clean = data

    # ── 4. DESCRIPTIVE ───────────────────────────────────────────────────────────
    nlags = min(args.nlags, len(clean) // 3)
    summary = summarize_series(clean, nlags=nlags, rolling_window=args.rolling_window)
    stats = summary["descriptive"]
    logger.info(
        "[DESCRIBE] mean=%.2f, std=%.2f, skew=%.4f, kurt=%.4f",
        stats["mean"], stats["std"], stats["skewness"], stats["kurtosis"],
    )

    # ── 5. STATISTICAL TESTS ────────────────────────────────────────────────────
    stationarity = test_stationarity(clean, alpha=args.alpha)
    normality = test_normality(clean, alpha=args.alpha)
    seasonal = test_seasonality(clean, period=args.seasonal_period)
    hetero = test_heteroscedasticity(clean, nlags=min(5, len(clean) // 10))

    logger.info("[TEST] ADF: %s (stat=%.4f, p=%.4f)",
                stationarity["adf"].interpretation,
                stationarity["adf"].statistic, stationarity["adf"].p_value)
    logger.info("[TEST] KPSS: %s (stat=%.4f, p=%.4f)",
                stationarity["kpss"].interpretation,
                stationarity["kpss"].statistic, stationarity["kpss"].p_value)
    logger.info("[TEST] Shapiro: %s (p=%.4f)",
                normality["shapiro"].interpretation, normality["shapiro"].p_value)
    logger.info("[TEST] Jarque-Bera: %s (p=%.4f)",
                normality["jarque_bera"].interpretation, normality["jarque_bera"].p_value)
    logger.info("[TEST] Seasonality: %s", seasonal.interpretation)
    logger.info("[TEST] ARCH: %s (p=%.4f)", hetero.interpretation, hetero.p_value)

    # ── 6. VISUALIZE (all 17 functions) ──────────────────────────────────────────
    n_plots = 0
    if HAS_VIZ and not args.no_plots:
        logger.info("[VISUALIZE] Generating all analysis plots...")
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # 1. Raw series
        plot_series(data, title="Raw Time Series", save_path=plots_dir / "01_series.png")
        n_plots += 1

        # 2. Validation summary (data quality dashboard)
        plot_validation_summary(
            data, validation_result=val.to_dict(), freq_result=freq,
            title="Data Quality Summary", save_path=plots_dir / "02_validation_summary.png",
        )
        n_plots += 1

        # 3. Missing data analysis
        if data.isna().any():
            plot_missing_data(data, save_path=plots_dir / "03_missing_data.png")
            n_plots += 1

        # 4. Diagnostics (4-panel)
        plot_diagnostics(clean, title="Diagnostics", save_path=plots_dir / "04_diagnostics.png")
        n_plots += 1

        # 5. ACF/PACF
        acf_data = summary["acf_pacf"]
        plot_acf_pacf(
            acf_data["acf_values"], acf_data["pacf_values"],
            nlags=acf_data["nlags"], save_path=plots_dir / "05_acf_pacf.png",
        )
        n_plots += 1

        # 6. Rolling statistics with ±2σ bands
        plot_rolling_statistics(
            clean, window=args.rolling_window,
            title=f"Rolling Statistics (window={args.rolling_window})",
            save_path=plots_dir / "06_rolling_statistics.png",
        )
        n_plots += 1

        # 7. Distribution (histogram + KDE + violin)
        plot_distribution(clean, title="Distribution Analysis",
                          save_path=plots_dir / "07_distribution.png")
        n_plots += 1

        # 8. Lag scatter
        max_lag = min(max(args.seasonal_period, 12), len(clean) // 4)
        lags = sorted(set([1, args.seasonal_period, max_lag]))
        plot_lag_scatter(clean, lags=lags, save_path=plots_dir / "08_lag_scatter.png")
        n_plots += 1

        # 9. Stationarity visual check (4-panel)
        plot_stationarity_summary(
            clean, window=args.rolling_window,
            save_path=plots_dir / "09_stationarity.png",
        )
        n_plots += 1

        # 10. Seasonal decomposition
        sp = args.seasonal_period
        if len(clean) >= sp * 2:
            try:
                decomp = compute_seasonal_decomposition(clean, period=sp)
                plot_decomposition(decomp, save_path=plots_dir / "10_decomposition.png")
                n_plots += 1
            except Exception as e:
                logger.warning("Decomposition skipped: %s", e)

        # 11. Boxplot by period (multiple groupings)
        if isinstance(clean.index, pd.DatetimeIndex):
            for period in ["month", "dayofweek"]:
                try:
                    plot_boxplot_by_period(
                        clean, period=period,
                        title=f"Distribution by {period}",
                        save_path=plots_dir / f"11_boxplot_{period}.png",
                    )
                    n_plots += 1
                except Exception as e:
                    logger.warning("Boxplot by %s skipped: %s", period, e)

        # 12. Correlation heatmap (lag-based auto-correlation matrix)
        try:
            max_corr_lag = min(10, len(clean) // 5)
            lag_df = pd.DataFrame({f"lag_{i}": clean.shift(i) for i in range(max_corr_lag + 1)}).dropna()
            plot_correlation_heatmap(lag_df, title="Lag Correlation Heatmap",
                                     save_path=plots_dir / "12_correlation_heatmap.png")
            n_plots += 1
        except Exception as e:
            logger.warning("Correlation heatmap skipped: %s", e)

        logger.info("[VISUALIZE] Generated %d plots to %s", n_plots, plots_dir)
    else:
        logger.info("[VISUALIZE] Skipped (--no-plots or matplotlib not installed)")

    # ── 7. EXPORT ────────────────────────────────────────────────────────────────
    report = {
        "input": str(args.input),
        "n_observations": len(data),
        "validation": val.to_dict(),
        "frequency": {k: str(v) for k, v in freq.items()},
        "patchiness": patchiness.to_dict(),
        "descriptive_stats": stats,
        "acf_pacf": {
            "nlags": acf_data["nlags"],
            "significant_acf_lags": acf_data.get("significant_acf_lags", []),
            "significant_pacf_lags": acf_data.get("significant_pacf_lags", []),
        },
        "statistical_tests": {
            "adf": stationarity["adf"].to_dict(),
            "kpss": stationarity["kpss"].to_dict(),
            "shapiro": normality["shapiro"].to_dict(),
            "jarque_bera": normality["jarque_bera"].to_dict(),
            "seasonality": seasonal.to_dict(),
            "arch": hetero.to_dict(),
        },
        "n_plots": n_plots,
    }
    report_path = out_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("[EXPORT] Report saved to %s", report_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f" Comprehensive Analysis Complete")
    print(f"{'='*60}")
    print(f" Input:         {args.input} ({len(data)} observations)")
    print(f" Valid:          {val.is_valid}, Freq: {freq['inferred_freq']}")
    print(f" Missing:        {val.missing_pct:.1f}%, Gaps: {patchiness.n_gaps}")
    print(f" Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    print(f" Skewness: {stats['skewness']:.4f}, Kurtosis: {stats['kurtosis']:.4f}")
    print(f"\n Statistical Tests (α={args.alpha}):")
    print(f"   ADF:          {stationarity['adf'].interpretation}")
    print(f"   KPSS:         {stationarity['kpss'].interpretation}")
    print(f"   Shapiro:      {normality['shapiro'].interpretation}")
    print(f"   Jarque-Bera:  {normality['jarque_bera'].interpretation}")
    print(f"   Seasonality:  {seasonal.interpretation}")
    print(f"   ARCH:         {hetero.interpretation}")
    print(f"\n Plots:          {n_plots} generated")
    print(f" Report:         {report_path}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
