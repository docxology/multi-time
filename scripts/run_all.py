#!/usr/bin/env python3
"""Run all thin orchestrator scripts, each outputting to a named subfolder.

Each script runs with real, configurable parameters and saves all output
(data, stats, plots, reports) to output/<script_name>/.

Usage (from anywhere):
    python scripts/run_all.py
    uv run python scripts/run_all.py
    cd scripts && python run_all.py
    python scripts/run_all.py --output-dir output --skip validation
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _find_project_root() -> Path:
    """Find the project root by walking up from this script to find pyproject.toml."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback: assume scripts/ is one level below root
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()


def _find_uv() -> str | None:
    """Find the `uv` executable."""
    return shutil.which("uv")


def _build_python_cmd() -> list[str]:
    """Build the Python invocation command.

    Prefers `uv run python` (ensures correct venv), falls back to
    the project's .venv/bin/python, then sys.executable.
    """
    uv = _find_uv()
    if uv:
        return [uv, "run", "python"]

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python)]

    return [sys.executable]


PYTHON_CMD = _build_python_cmd()

SCRIPTS = {
    "synthetic": {
        "script": "scripts/run_synthetic.py",
        "args": [
            "--type", "configurable",
            "--n", "365",
            "--trend", "0.3",
            "--seasonal-period", "12",
            "--seasonal-amplitude", "8",
            "--noise-std", "2.0",
            "--gap-fraction", "0.05",
        ],
        "description": "Generate configurable synthetic time series (365 obs)",
    },
    "validation": {
        "script": "scripts/run_validation.py",
        "args": ["--input", "{output_dir}/synthetic/synthetic_configurable.csv"],
        "description": "Validate data quality, frequency, and patchiness",
        "depends_on": "synthetic",
    },
    "descriptive": {
        "script": "scripts/run_descriptive.py",
        "args": ["--input", "{output_dir}/synthetic/synthetic_configurable.csv"],
        "description": "Compute descriptive statistics and ACF/PACF",
        "depends_on": "synthetic",
    },
    "analysis": {
        "script": "scripts/run_analysis.py",
        "args": [
            "--input", "{output_dir}/synthetic/synthetic_configurable.csv",
            "--seasonal-period", "12",
            "--rolling-window", "14",
        ],
        "description": "Full analysis with 6 statistical tests + up to 14 visualization plots",
        "depends_on": "synthetic",
    },
    "forecast": {
        "script": "scripts/run_forecast.py",
        "args": [
            "--input", "{output_dir}/synthetic/synthetic_configurable.csv",
            "--models", "naive", "theta",
            "--ensemble",
            "--horizon", "30",
            "--test-size", "30",
        ],
        "description": "Multi-model forecasting with ensemble + temporal CV + 7 viz plots",
        "depends_on": "synthetic",
    },
    "pipeline": {
        "script": "scripts/run_pipeline.py",
        "args": [
            "--input", "{output_dir}/synthetic/synthetic_configurable.csv",
        ],
        "description": "Full pipeline (validate → stats → forecast → evaluate)",
        "depends_on": "synthetic",
    },
    "end_to_end": {
        "script": "scripts/run_end_to_end.py",
        "args": [
            "--type", "configurable",
            "--n", "365",
            "--trend", "0.2",
            "--seasonal-period", "7",
            "--seasonal-amplitude", "5",
            "--noise-std", "1.5",
            "--models", "naive", "theta", "exp_smoothing",
            "--ensemble",
            "--horizon", "30",
        ],
        "description": "End-to-end: generate → validate → 6 tests → train → all 17 viz plots",
    },
    "multi_series": {
        "script": "scripts/run_multi_series.py",
        "args": [
            "--n-series", "4",
            "--harmonize-freq", "D",
            "--models", "naive", "theta",
            "--horizon", "14",
        ],
        "description": "Multi-series: overlap, cross-correlation, individual forecasting, 30+ viz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all multi-time orchestrator scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", "-o", default="output", help="Root output directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help=f"Scripts to skip: {list(SCRIPTS.keys())}",
    )
    parser.add_argument("--only", nargs="*", help="Run only these scripts")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    return parser.parse_args()


def run_script(
    name: str,
    spec: dict,
    output_dir: Path,
    log_level: str,
    dry_run: bool = False,
) -> dict:
    """Run a single orchestrator script and capture results."""
    script_output = output_dir / name
    script_output.mkdir(parents=True, exist_ok=True)

    # Build command — always use uv/venv python, resolve script relative to project root
    script_path = str(PROJECT_ROOT / spec["script"])
    cmd = [*PYTHON_CMD, script_path]

    # Substitute output_dir in args (use absolute path for reliability)
    abs_output = str(output_dir.resolve())
    args = []
    for arg in spec["args"]:
        args.append(arg.format(output_dir=abs_output))
    cmd.extend(args)

    # Add output dir and log level
    if "--output" not in " ".join(spec["args"]) and "--output-dir" not in " ".join(spec["args"]):
        if "run_synthetic" in spec["script"]:
            cmd.extend(["--output", str(script_output.resolve() / "synthetic_configurable.csv")])
        else:
            # All other scripts use --output-dir for both JSON + plots
            cmd.extend(["--output-dir", str(script_output.resolve())])
    cmd.extend(["--log-level", log_level])

    # Log file
    log_file = script_output / f"{name}.log"

    result = {
        "name": name,
        "description": spec["description"],
        "command": " ".join(cmd),
        "output_dir": str(script_output.resolve()),
        "log_file": str(log_file.resolve()),
    }

    if dry_run:
        result["status"] = "dry_run"
        return result

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),  # Always run from project root
        )
        elapsed = time.time() - start
        result["elapsed_seconds"] = round(elapsed, 2)
        result["returncode"] = proc.returncode
        result["status"] = "success" if proc.returncode == 0 else "failed"

        # Save stdout/stderr to log
        with open(log_file, "w") as f:
            f.write(f"=== {name} — {spec['description']} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Project root: {PROJECT_ROOT}\n")
            f.write(f"Return code: {proc.returncode}\n")
            f.write(f"Elapsed: {elapsed:.2f}s\n\n")
            f.write("--- STDOUT ---\n")
            f.write(proc.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(proc.stderr)

        if proc.returncode != 0:
            result["error"] = proc.stderr[-500:] if proc.stderr else "Unknown error"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["elapsed_seconds"] = 120.0
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main() -> int:
    args = parse_args()

    # Resolve output_dir relative to project root (not CWD)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which scripts to run
    if args.only:
        script_names = [s for s in args.only if s in SCRIPTS]
    else:
        script_names = [s for s in SCRIPTS if s not in args.skip]

    print(f"\n{'='*60}")
    print(f" Multi-Time: Run All Orchestrators")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Project: {PROJECT_ROOT}")
    print(f" Python:  {' '.join(PYTHON_CMD)}")
    print(f" Output:  {output_dir.resolve()}")
    print(f" Scripts: {len(script_names)}")
    print(f"{'='*60}\n")

    results = []
    passed = 0
    failed = 0

    for i, name in enumerate(script_names, 1):
        spec = SCRIPTS[name]

        # Check dependency — skip if dependency failed
        dep = spec.get("depends_on")
        if dep:
            dep_result = next((r for r in results if r["name"] == dep), None)
            if dep_result and dep_result["status"] != "success":
                print(f"[{i}/{len(script_names)}] {name:20s} — SKIPPED (dependency '{dep}' failed)")
                results.append({
                    "name": name,
                    "status": "skipped",
                    "description": spec["description"],
                    "error": f"Dependency '{dep}' failed",
                })
                failed += 1
                continue

        print(f"[{i}/{len(script_names)}] {name:20s} — {spec['description']}")

        result = run_script(name, spec, output_dir, args.log_level, args.dry_run)
        results.append(result)

        status = result["status"]
        elapsed = result.get("elapsed_seconds", 0)
        if status == "success":
            passed += 1
            print(f"   ✅ PASS ({elapsed:.1f}s)")
        elif status == "dry_run":
            print(f"   🔍 {result['command']}")
        else:
            failed += 1
            err_msg = result.get("error", "Unknown")
            # Show last 80 chars of error, but try to find the actual error line
            for line in reversed(err_msg.strip().split("\n")):
                line = line.strip()
                if line and not line.startswith("File ") and not line.startswith("^"):
                    err_msg = line[:80]
                    break
            print(f"   ❌ {status.upper()}: {err_msg}")

    # Save master report
    report = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python_cmd": " ".join(PYTHON_CMD),
        "output_dir": str(output_dir.resolve()),
        "total": len(script_names),
        "passed": passed,
        "failed": failed,
        "scripts": results,
    }
    report_path = output_dir / "run_all_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Count output files
    total_files = len(list(output_dir.rglob("*")))

    print(f"\n{'='*60}")
    print(f" Summary: {passed} passed, {failed} failed")
    print(f" Output files: {total_files} in {output_dir.resolve()}")
    print(f" Report: {report_path.resolve()}")
    print(f"{'='*60}\n")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
