#!/usr/bin/env python3
"""Thin orchestrator for end-to-end analysis: generate → validate → stats → model → visualize.

Demonstrates the full multi-time toolkit in a single command: generates synthetic
data, runs validation + all statistical tests, trains multiple models with ensemble,
produces prediction intervals, temporal CV, and generates all 17 visualization types.

Usage:
    python scripts/run_end_to_end.py --type daily --n 365 --models naive theta -o output/
    python scripts/run_end_to_end.py --type configurable --n 500 --trend 0.3 --seasonal-period 12 --seasonal-amplitude 8 --models naive theta exp_smoothing --ensemble -o output/e2e/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from multi_time.config import MultiTimeConfig, load_config, setup_logging, get_logger
from multi_time.data import GENERATOR_REGISTRY, list_generators
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
from multi_time.modeling import (
    create_forecaster,
    run_forecast,
    create_ensemble,
    predict_intervals,
    evaluate_forecaster,
    FORECASTER_REGISTRY,
)
from multi_time.evaluate import evaluate_forecast

logger = get_logger(__name__)

# All 17 visualization imports (optional)
try:
    from multi_time.visualization import (
        plot_series,
        plot_forecast,
        plot_diagnostics,
        plot_decomposition,
        plot_acf_pacf,
        plot_residuals,
        plot_rolling_statistics,
        plot_distribution,
        plot_stationarity_summary,
        plot_lag_scatter,
        plot_boxplot_by_period,
        plot_correlation_heatmap,
        plot_validation_summary,
        plot_missing_data,
        plot_model_comparison,
        plot_error_distribution,
        plot_cumulative_error,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end: generate → validate → stats → train → visualize (all 17 plots)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data generation
    parser.add_argument("--type", "-t", default="daily", help=f"Generator: {list_generators()}")
    parser.add_argument("--n", type=int, default=365, help="Observations to generate")
    parser.add_argument("--start", default="2023-01-01", help="Start date")
    parser.add_argument("--freq", default="D", help="Frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Configurable series params
    parser.add_argument("--trend", type=float, default=0.1, help="Trend slope")
    parser.add_argument("--seasonal-period", type=int, help="Seasonal period")
    parser.add_argument("--seasonal-amplitude", type=float, default=0.0, help="Seasonal amplitude")
    parser.add_argument("--noise-std", type=float, default=2.0, help="Noise std dev")
    parser.add_argument("--baseline", type=float, default=100.0, help="Baseline level")
    parser.add_argument("--outlier-fraction", type=float, default=0.0, help="Outlier fraction")
    parser.add_argument("--gap-fraction", type=float, default=0.0, help="Missing fraction")

    # Model + evaluation
    parser.add_argument(
        "--models", "-m", nargs="*", default=["naive", "theta"],
        help=f"Model names: {list(FORECASTER_REGISTRY.keys())}",
    )
    parser.add_argument("--horizon", "-H", type=int, default=30, help="Forecast horizon")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--ensemble", action="store_true", help="Also run ensemble of models")
    parser.add_argument("--coverage", type=float, default=0.95, help="Prediction interval coverage")

    # Config + output
    parser.add_argument("--config", "-c", help="YAML config file")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    return parser.parse_args()


def _build_gen_kwargs(args: argparse.Namespace) -> dict:
    """Build generator kwargs from CLI args."""
    gen_type = args.type
    kwargs: dict = {"seed": args.seed}

    if gen_type in ("daily", "weekly"):
        kwargs.update(n=args.n, start=args.start, trend=args.trend, noise_std=args.noise_std)
        if gen_type == "weekly":
            kwargs["annual_amplitude"] = args.seasonal_amplitude or 8.0
    elif gen_type == "monthly":
        kwargs.update(
            n=args.n, start=args.start, trend=args.trend, noise_std=args.noise_std,
            seasonal_amplitude=args.seasonal_amplitude or 10.0,
        )
    elif gen_type == "hourly":
        kwargs.update(
            n=args.n, start=args.start, daily_amplitude=args.seasonal_amplitude or 5.0,
            baseline=args.baseline, noise_std=args.noise_std,
        )
    elif gen_type == "random_walk":
        kwargs.update(
            n=args.n, start=args.start, freq=args.freq,
            drift=args.trend, volatility=args.noise_std, initial_value=args.baseline,
        )
    elif gen_type == "configurable":
        kwargs.update(
            n=args.n, start=args.start, freq=args.freq, baseline=args.baseline,
            trend=args.trend, seasonal_period=args.seasonal_period,
            seasonal_amplitude=args.seasonal_amplitude, noise_std=args.noise_std,
            outlier_fraction=args.outlier_fraction, gap_fraction=args.gap_fraction,
        )
    else:
        kwargs.update(n=args.n, start=args.start)

    return kwargs


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger.info("=== Multi-Time End-to-End Pipeline ===")

    # Load config if provided
    config = load_config(args.config) if args.config else MultiTimeConfig(
        models=args.models,
        forecast_horizon=args.horizon,
        output_dir=args.output_dir,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. GENERATE ─────────────────────────────────────────────────────────────
    gen_type = args.type
    if gen_type not in GENERATOR_REGISTRY:
        logger.error("Unknown generator: %s", gen_type)
        return 1

    gen_kwargs = _build_gen_kwargs(args)
    data = GENERATOR_REGISTRY[gen_type](**gen_kwargs)

    # Handle multivariate → take first column
    if isinstance(data, pd.DataFrame):
        logger.info("Multivariate data: using first column for univariate pipeline")
        data = data.iloc[:, 0]

    logger.info("[GENERATE] %s: %d observations", gen_type, len(data))

    # Save raw data
    data.to_csv(out_dir / "synthetic_data.csv", header=True)
    logger.info("Saved synthetic data to %s", out_dir / "synthetic_data.csv")

    # ── 2. VALIDATE ─────────────────────────────────────────────────────────────
    val = validate_series(data)
    freq_info = detect_frequency(data)
    patchiness = assess_patchiness(data)
    logger.info(
        "[VALIDATE] valid=%s, freq=%s, missing=%.1f%%, gaps=%d",
        val.is_valid, freq_info["inferred_freq"], val.missing_pct, patchiness.n_gaps,
    )

    # ── 3. DESCRIBE ─────────────────────────────────────────────────────────────
    nlags = min(40, len(data.dropna()) // 3)
    summary = summarize_series(data.dropna(), nlags=nlags, rolling_window=7)
    stats = summary["descriptive"]
    logger.info(
        "[DESCRIBE] mean=%.2f, std=%.2f, skew=%.4f, kurt=%.4f",
        stats["mean"], stats["std"], stats["skewness"], stats["kurtosis"],
    )

    # ── 4. TEST ─────────────────────────────────────────────────────────────────
    clean = data.dropna()
    stationarity = test_stationarity(clean)
    normality = test_normality(clean)
    sp = args.seasonal_period or 7
    seasonal = test_seasonality(clean, period=sp)
    hetero = test_heteroscedasticity(clean, nlags=min(5, len(clean) // 10))
    logger.info("[TEST] ADF: %s (p=%.4f)", stationarity["adf"].interpretation, stationarity["adf"].p_value)
    logger.info("[TEST] KPSS: %s (p=%.4f)", stationarity["kpss"].interpretation, stationarity["kpss"].p_value)
    logger.info("[TEST] Normality: %s (p=%.4f)", normality["shapiro"].interpretation, normality["shapiro"].p_value)
    logger.info("[TEST] Seasonality: %s", seasonal.interpretation)
    logger.info("[TEST] ARCH: %s (p=%.4f)", hetero.interpretation, hetero.p_value)

    # ── 5. TRANSFORM (impute if needed) ─────────────────────────────────────────
    if data.isna().any():
        pipeline = build_transform_pipeline(["impute"])
        data_clean = apply_transform(pipeline, data)
        logger.info("[TRANSFORM] Imputed %d missing values", data.isna().sum())
    else:
        data_clean = data

    # ── 6. FORECAST ─────────────────────────────────────────────────────────────
    test_size = max(1, int(len(data_clean) * args.test_fraction))
    test_size = min(test_size, args.horizon)
    y_train = data_clean.iloc[:-test_size]
    y_test = data_clean.iloc[-test_size:]
    fh = len(y_test)
    logger.info("[FORECAST] train=%d, test=%d, fh=%d", len(y_train), len(y_test), fh)

    model_results = {}
    all_predictions = {}
    all_intervals = {}
    model_names = config.models if args.config else args.models

    for model_name in model_names:
        try:
            f = create_forecaster(model_name)
            preds = run_forecast(f, y_train, fh=fh)
            metrics = evaluate_forecast(y_test, preds, metrics_list=config.metrics)
            model_results[model_name] = metrics
            all_predictions[model_name] = preds

            # Prediction intervals
            try:
                f_int = create_forecaster(model_name)
                intervals = predict_intervals(f_int, y_train, fh=fh, coverage=args.coverage)
                all_intervals[model_name] = intervals
            except Exception:
                all_intervals[model_name] = None

            logger.info(
                "[FORECAST] %s: MAE=%.4f, RMSE=%.4f, MAPE=%.4f",
                model_name, metrics.get("mae", float("nan")),
                metrics.get("rmse", float("nan")), metrics.get("mape", float("nan")),
            )
        except Exception as e:
            logger.warning("[FORECAST] %s failed: %s", model_name, e)
            model_results[model_name] = {"error": str(e)}

    # Ensemble
    if args.ensemble and len(model_names) > 1:
        try:
            ens = create_ensemble(model_names)
            ens_preds = run_forecast(ens, y_train, fh=fh)
            ens_metrics = evaluate_forecast(y_test, ens_preds, metrics_list=config.metrics)
            model_results["ensemble"] = ens_metrics
            all_predictions["ensemble"] = ens_preds
            all_intervals["ensemble"] = None
            logger.info("[FORECAST] ensemble: MAE=%.4f", ens_metrics.get("mae", float("nan")))
        except Exception as e:
            logger.warning("[FORECAST] ensemble failed: %s", e)

    # ── 7. DETERMINE BEST MODEL ─────────────────────────────────────────────────
    best_model = None
    if all_predictions:
        valid = [k for k in model_results if "error" not in model_results[k]]
        if valid:
            best_model = min(valid, key=lambda k: model_results[k].get("mae", float("inf")))

    # ── 8. TEMPORAL CV (best model) ─────────────────────────────────────────────
    cv_summary: dict = {}
    if best_model and len(y_train) > 50:
        try:
            f_cv = create_forecaster(best_model)
            cv_results = evaluate_forecaster(f_cv, y_train, initial_window=max(10, int(len(y_train) * 0.6)), fh=1)
            cv_summary = {
                "model": best_model,
                "n_folds": len(cv_results),
                "mean_test_score": float(cv_results["test_MeanAbsoluteError"].mean()) if "test_MeanAbsoluteError" in cv_results.columns else None,
            }
            logger.info("[CV] %s: %d folds, mean MAE=%.4f",
                        best_model, cv_summary["n_folds"], cv_summary.get("mean_test_score", 0))
        except Exception as e:
            logger.warning("[CV] Temporal cross-validation skipped: %s", e)

    # ── 9. VISUALIZE (all 17 functions) ─────────────────────────────────────────
    n_plots = 0
    if HAS_VIZ and not args.no_plots:
        logger.info("[VISUALIZE] Generating all plots...")

        # 1. Raw series
        plot_series(data, title=f"Synthetic {gen_type} Series", save_path=out_dir / "01_series.png")
        n_plots += 1

        # 2. Validation summary
        plot_validation_summary(
            data, validation_result=val.to_dict(), freq_result=freq_info,
            save_path=out_dir / "02_validation_summary.png",
        )
        n_plots += 1

        # 3. Missing data (if gaps)
        if data.isna().any():
            plot_missing_data(data, save_path=out_dir / "03_missing_data.png")
            n_plots += 1

        # 4. Diagnostics
        plot_diagnostics(clean, title="Diagnostics", save_path=out_dir / "04_diagnostics.png")
        n_plots += 1

        # 5. ACF/PACF
        acf_data = summary["acf_pacf"]
        plot_acf_pacf(
            acf_data["acf_values"], acf_data["pacf_values"],
            nlags=acf_data["nlags"], save_path=out_dir / "05_acf_pacf.png",
        )
        n_plots += 1

        # 6. Rolling statistics
        plot_rolling_statistics(
            clean, window=max(7, len(clean) // 30),
            title="Rolling Statistics", save_path=out_dir / "06_rolling_statistics.png",
        )
        n_plots += 1

        # 7. Distribution
        plot_distribution(clean, title="Distribution Analysis", save_path=out_dir / "07_distribution.png")
        n_plots += 1

        # 8. Stationarity
        plot_stationarity_summary(
            clean, window=max(7, len(clean) // 30), save_path=out_dir / "08_stationarity.png",
        )
        n_plots += 1

        # 9. Lag scatter
        plot_lag_scatter(clean, lags=[1, sp], save_path=out_dir / "09_lag_scatter.png")
        n_plots += 1

        # 10. Boxplot by period
        if isinstance(clean.index, pd.DatetimeIndex):
            for period in ["month", "dayofweek"]:
                try:
                    plot_boxplot_by_period(clean, period=period, save_path=out_dir / f"10_boxplot_{period}.png")
                    n_plots += 1
                except Exception:
                    pass

        # 11. Decomposition
        if len(clean) >= sp * 2:
            try:
                decomp = compute_seasonal_decomposition(clean, period=sp)
                plot_decomposition(decomp, save_path=out_dir / "11_decomposition.png")
                n_plots += 1
            except Exception as e:
                logger.warning("Decomposition plot skipped: %s", e)

        # 12. Correlation heatmap (lag-based)
        try:
            max_corr_lag = min(10, len(clean) // 5)
            lag_df = pd.DataFrame({f"lag_{i}": clean.shift(i) for i in range(max_corr_lag + 1)}).dropna()
            plot_correlation_heatmap(lag_df, title="Lag Correlation", save_path=out_dir / "12_correlation.png")
            n_plots += 1
        except Exception:
            pass

        # 13. Forecast plot (best model with intervals)
        if best_model:
            plot_forecast(
                y_train, all_predictions[best_model], y_test=y_test,
                intervals=all_intervals.get(best_model),
                title=f"Best: {best_model}", save_path=out_dir / "13_forecast.png",
            )
            n_plots += 1

            # 14. Residuals
            residuals = y_test - all_predictions[best_model].loc[y_test.index]
            plot_residuals(residuals, save_path=out_dir / "14_residuals.png")
            n_plots += 1

        # 15-17. Multi-model comparison
        if len(all_predictions) > 1:
            try:
                plot_model_comparison(
                    y_test, all_predictions, metrics=model_results,
                    save_path=out_dir / "15_model_comparison.png",
                )
                n_plots += 1
                plot_error_distribution(
                    y_test, all_predictions, save_path=out_dir / "16_error_distribution.png",
                )
                n_plots += 1
                plot_cumulative_error(
                    y_test, all_predictions, save_path=out_dir / "17_cumulative_error.png",
                )
                n_plots += 1
            except Exception as e:
                logger.warning("Comparison plots skipped: %s", e)

        logger.info("[VISUALIZE] Saved %d plots to %s", n_plots, out_dir)
    else:
        logger.info("[VISUALIZE] Skipped (--no-plots or matplotlib not installed)")

    # ── 10. EXPORT ──────────────────────────────────────────────────────────────
    report = {
        "generator": gen_type,
        "n_observations": len(data),
        "validation": val.to_dict(),
        "frequency": {k: str(v) for k, v in freq_info.items()},
        "patchiness": patchiness.to_dict(),
        "descriptive_stats": stats,
        "statistical_tests": {
            "adf": stationarity["adf"].to_dict(),
            "kpss": stationarity["kpss"].to_dict(),
            "shapiro": normality["shapiro"].to_dict(),
            "jarque_bera": normality["jarque_bera"].to_dict(),
            "seasonality": seasonal.to_dict(),
            "arch": hetero.to_dict(),
        },
        "models": model_results,
        "best_model": best_model,
        "temporal_cv": cv_summary if cv_summary else None,
        "n_plots": n_plots,
    }
    report_path = out_dir / "e2e_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("[EXPORT] Report saved to %s", report_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f" End-to-End Analysis Complete — {gen_type}")
    print(f"{'='*60}")
    print(f" Observations: {len(data)}")
    print(f" Valid: {val.is_valid}, Frequency: {freq_info['inferred_freq']}")
    print(f" Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    print(f" ADF: {stationarity['adf'].interpretation}")
    print(f" KPSS: {stationarity['kpss'].interpretation}")
    print(f" ARCH: {hetero.interpretation}")
    print(f"\n {'Model':<20s} {'MAE':>8s} {'RMSE':>8s} {'MAPE':>8s}")
    print(f" {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for name, res in model_results.items():
        if "error" in res:
            print(f" {name:<20s} {'ERROR':>8s}")
        else:
            print(f" {name:<20s} {res.get('mae', 0):>8.4f} {res.get('rmse', 0):>8.4f} {res.get('mape', 0):>8.4f}")
    if best_model:
        print(f"\n Best: {best_model}")
    if cv_summary:
        print(f" CV:   {cv_summary['n_folds']} folds, mean MAE={cv_summary.get('mean_test_score', 0):.4f}")
    print(f" Plots: {n_plots}")
    print(f" Output: {out_dir.resolve()}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
