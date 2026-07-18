"""Measure the real adapter-summary footprint and project formal storage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .experiment import write_json


REQUIRED_RUN_FILES = {
    "config.json",
    "manifest.json",
    "runtime_validity.json",
    "summary.json",
    "summary.csv",
    "stdout.log",
}
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".png"}
FORMAL_NEW_RUNS = 620


def build_storage_report(run_root: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(run_root)
    run_dirs = sorted(path.parent for path in root.rglob("summary.json") if "runs" in path.parts)
    if not run_dirs:
        raise ValueError(f"no adapter-summary runs found below {root}")
    rows = []
    for run_dir in run_dirs:
        names = {path.name for path in run_dir.iterdir() if path.is_file()}
        missing = sorted(REQUIRED_RUN_FILES - names)
        if missing:
            raise ValueError(f"{run_dir} is missing {missing}")
        forbidden = sorted(
            path.name for path in run_dir.iterdir() if path.is_file() and path.suffix in FORBIDDEN_SUFFIXES
        )
        if forbidden:
            raise ValueError(f"{run_dir} contains forbidden heavy artifacts: {forbidden}")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("storage_schema") != "adapter_summary_v1":
            raise ValueError(f"{run_dir} does not use adapter_summary_v1")
        size = sum((run_dir / name).stat().st_size for name in REQUIRED_RUN_FILES)
        rows.append(
            {
                "run_dir": str(run_dir),
                "environment": summary.get("environment"),
                "condition": summary.get("condition"),
                "adapter": summary.get("adapter_name"),
                "bytes": size,
            }
        )
    inventory = pd.DataFrame(rows)
    per_run = int(inventory["bytes"].max())
    expected = FORMAL_NEW_RUNS * per_run
    report = {
        "schema": "adapter_storage_pilot_v1",
        "status": "valid",
        "pilot_root": str(root.resolve()),
        "pilot_runs": len(inventory),
        "mean_run_bytes": float(inventory["bytes"].mean()),
        "adapter_summary_bytes_per_run": per_run,
        "formal_new_runs": FORMAL_NEW_RUNS,
        "expected_output_bytes": expected,
        "estimated_peak_disk_bytes": int(math.ceil(expected * 1.25)),
    }
    return report, inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inventory-csv", required=True)
    args = parser.parse_args()
    report, inventory = build_storage_report(args.run_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    inventory_path = Path(args.inventory_csv)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(inventory_path, index=False)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
