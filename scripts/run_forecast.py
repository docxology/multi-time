#!/usr/bin/env python3
"""Thin orchestrator for time series forecasting with multi-model comparison.

Demonstrates: forecaster registry, ensemble, prediction intervals, temporal
cross-validation, model comparison, error distribution, and cumulative error
visualizations. Supports single or multi-model runs.

Usage:
    python scripts/run_forecast.py -i data.csv --model theta --horizon 30
    python scripts/run_forecast.py -i data.csv --models naive theta exp_smoothing --ensemble
    python scripts/run_forecast.py -i data.csv --models naive theta --test-size 30 -o output/forecast
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
from multi_time.modeling import (
    create_forecaster, create_ensemble, run_forecast, predict_intervals,
    evaluate_forecaster, FORECASTER_REGISTRY,
)
from multi_time.evaluate import evaluate_forecast

logger = get_logger(__name__)

# Visualization imports (optional — all forecasting-relevant functions)
try:
    from multi_time.visualization import (
        plot_series,
        plot_forecast,
        plot_residuals,
        plot_model_comparison,
        plot_error_distribution,
        plot_cumulative_error,
        plot_diagnostics,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-model forecasting with comparison and temporal CV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to CSV file")
    parser.add_argument("--config", "-c", help="Path to YAML config file")
    parser.add_argument("--column", help="Column name")
    parser.add_argument(
        "--model", "-m", default=None,
        help=f"Single model: {list(FORECASTER_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Multiple models for comparison (overrides --model)",
    )
    parser.add_argument("--ensemble", action="store_true", help="Also run ensemble of all models")
    parser.add_argument("--horizon", "-H", type=int, default=12, help="Forecast horizon")
    parser.add_argument("--test-size", type=int, help="Hold-out test size for evaluation")
    parser.add_argument("--coverage", type=float, default=0.95, help="Prediction interval coverage")
    parser.add_argument("--output", help="Path to save JSON results")
    parser.add_argument("--output-dir", "-o", help="Output directory for plots and results")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger.info("=== Multi-Time Forecast ===")

    config = load_config(args.config) if args.config else MultiTimeConfig()
    data = load_csv_series(args.input, column=args.column)
    logger.info("Loaded %d observations from %s", len(data), args.input)

    # Auto-impute missing values so models that cannot handle NaN work
    if data.isna().any():
        n_missing = int(data.isna().sum())
        from multi_time.transform import build_transform_pipeline, apply_transform
        pipeline = build_transform_pipeline(["impute"])
        data = apply_transform(pipeline, data)
        logger.info("Imputed %d missing values before forecasting", n_missing)

    # Ensure DatetimeIndex has freq set (required by many sktime forecasters)
    if isinstance(data.index, pd.DatetimeIndex) and data.index.freq is None:
        inferred = pd.infer_freq(data.index)
        if inferred:
            data.index = pd.DatetimeIndex(data.index, freq=inferred)
            logger.info("Set index freq to '%s'", inferred)

    # Determine model list
    model_names: list[str] = []
    if args.models:
        model_names = args.models
    elif args.model:
        model_names = [args.model]
    else:
        model_names = ["naive"]  # default

    # Train/test split
    if args.test_size and args.test_size < len(data):
        y_train = data.iloc[: -args.test_size]
        y_test = data.iloc[-args.test_size:]
    else:
        y_train = data
        y_test = None

    fh = args.horizon

    # ── FORECAST (all models) ────────────────────────────────────────────────────
    model_results: dict[str, dict] = {}
    all_predictions: dict[str, pd.Series] = {}
    all_intervals: dict[str, pd.DataFrame | None] = {}

    for model_name in model_names:
        try:
            f = create_forecaster(model_name)
            preds = run_forecast(f, y_train, fh=fh)
            all_predictions[model_name] = preds

            # Prediction intervals
            try:
                f_int = create_forecaster(model_name)
                intervals = predict_intervals(f_int, y_train, fh=fh, coverage=args.coverage)
                all_intervals[model_name] = intervals
            except Exception:
                all_intervals[model_name] = None

            # Evaluation (if test data available)
            metrics: dict = {}
            if y_test is not None:
                common_idx = y_test.index.intersection(preds.index)
                if len(common_idx) > 0:
                    metrics = evaluate_forecast(
                        y_test.loc[common_idx], preds.loc[common_idx],
                        metrics_list=config.metrics,
                    )

            model_results[model_name] = metrics
            logger.info(
                "[FORECAST] %s: MAE=%.4f, RMSE=%.4f, MAPE=%.4f",
                model_name,
                metrics.get("mae", float("nan")),
                metrics.get("rmse", float("nan")),
                metrics.get("mape", float("nan")),
            )
        except Exception as e:
            logger.warning("[FORECAST] %s failed: %s", model_name, e)
            model_results[model_name] = {"error": str(e)}

    # Ensemble (if requested and multiple models)
    if args.ensemble and len(model_names) > 1:
        try:
            ens = create_ensemble(model_names)
            ens_preds = run_forecast(ens, y_train, fh=fh)
            all_predictions["ensemble"] = ens_preds
            all_intervals["ensemble"] = None
            if y_test is not None:
                common_idx = y_test.index.intersection(ens_preds.index)
                if len(common_idx) > 0:
                    ens_metrics = evaluate_forecast(
                        y_test.loc[common_idx], ens_preds.loc[common_idx],
                        metrics_list=config.metrics,
                    )
                    model_results["ensemble"] = ens_metrics
                    logger.info("[FORECAST] ensemble: MAE=%.4f", ens_metrics.get("mae", float("nan")))
        except Exception as e:
            logger.warning("[FORECAST] ensemble failed: %s", e)

    # Determine best model
    best_model = None
    if all_predictions and y_test is not None:
        valid_models = [k for k in model_results if "error" not in model_results[k] and model_results[k]]
        if valid_models:
            best_model = min(valid_models, key=lambda k: model_results[k].get("mae", float("inf")))
            logger.info("[BEST] %s (MAE=%.4f)", best_model, model_results[best_model].get("mae", 0))

    # ── TEMPORAL CV (optional evaluation summary) ────────────────────────────────
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

    # ── VISUALIZE ────────────────────────────────────────────────────────────────
    n_plots = 0
    if HAS_VIZ and not args.no_plots:
        out_dir = Path(args.output_dir) if args.output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1. Best model forecast with intervals
            if best_model and best_model in all_predictions:
                plot_forecast(
                    y_train, all_predictions[best_model], y_test=y_test,
                    intervals=all_intervals.get(best_model),
                    title=f"Best: {best_model} (h={fh})",
                    save_path=out_dir / "forecast.png",
                )
                n_plots += 1

                # 2. Residuals
                if y_test is not None:
                    common_idx = y_test.index.intersection(all_predictions[best_model].index)
                    if len(common_idx) > 0:
                        residuals = y_test.loc[common_idx] - all_predictions[best_model].loc[common_idx]
                        plot_residuals(residuals, title=f"Residuals: {best_model}",
                                       save_path=out_dir / "residuals.png")
                        n_plots += 1

            # 3. Train + forecast overlay
            if all_predictions:
                first_model = list(all_predictions.keys())[0]
                plot_series(y_train, all_predictions[first_model],
                            labels=["Train", "Forecast"],
                            title="Train vs Forecast",
                            save_path=out_dir / "series.png")
                n_plots += 1

            # 4-6. Multi-model comparison (if multiple models)
            if len(all_predictions) > 1 and y_test is not None:
                try:
                    plot_model_comparison(
                        y_test, all_predictions, metrics=model_results,
                        title="Model Comparison",
                        save_path=out_dir / "model_comparison.png",
                    )
                    n_plots += 1

                    plot_error_distribution(
                        y_test, all_predictions,
                        save_path=out_dir / "error_distribution.png",
                    )
                    n_plots += 1

                    plot_cumulative_error(
                        y_test, all_predictions,
                        save_path=out_dir / "cumulative_error.png",
                    )
                    n_plots += 1
                except Exception as e:
                    logger.warning("Comparison plots skipped: %s", e)

            # 7. Diagnostics of training data
            plot_diagnostics(y_train, title="Training Data Diagnostics",
                             save_path=out_dir / "diagnostics.png")
            n_plots += 1

            logger.info("[VISUALIZE] Generated %d plots to %s", n_plots, out_dir)

    # ── EXPORT ───────────────────────────────────────────────────────────────────
    results: dict = {
        "model_names": model_names,
        "horizon": fh,
        "n_train": len(y_train),
        "n_test": len(y_test) if y_test is not None else 0,
        "models": model_results,
        "best_model": best_model,
        "temporal_cv": cv_summary if cv_summary else None,
        "coverage": args.coverage,
        "n_plots": n_plots,
    }

    # Include predictions
    for name, preds in all_predictions.items():
        results.setdefault("predictions", {})[name] = {str(k): v for k, v in preds.to_dict().items()}

    print(json.dumps(results, indent=2, default=str))

    json_path = args.output
    if not json_path and args.output_dir:
        json_path = str(Path(args.output_dir) / "forecast_results.json")
    if json_path:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", json_path)

    # Print summary table
    if model_results:
        print(f"\n{'='*60}")
        print(f" Forecast Results")
        print(f"{'='*60}")
        print(f" {'Model':<20s} {'MAE':>8s} {'RMSE':>8s} {'MAPE':>8s}")
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
        print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
