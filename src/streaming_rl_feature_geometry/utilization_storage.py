"""Measure the real adapter-summary footprint and project formal storage."""

from __future__ import annotations

import argparse
import json
import shutil
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
FORMAL_NEW_RUNS = 1290
FORMAL_RUNS_BY_ADAPTER = {
    "identity": 600,
    "residual_rff": 600,
    "tile_coding": 90,
}
FORMAL_RUNS_BY_STAGE = {
    "stage-a": {"identity": 560, "residual_rff": 560, "tile_coding": 0},
    "stage-b1": {"identity": 40, "residual_rff": 40, "tile_coding": 40},
    "stage-b2": {"identity": 0, "residual_rff": 0, "tile_coding": 50},
}


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
    by_adapter = {
        adapter: int(group["bytes"].max())
        for adapter, group in inventory.groupby("adapter")
    }
    missing_adapters = sorted(set(FORMAL_RUNS_BY_ADAPTER) - set(by_adapter))
    if missing_adapters:
        raise ValueError(f"storage pilot is missing adapters: {missing_adapters}")
    expected = sum(
        FORMAL_RUNS_BY_ADAPTER[adapter] * by_adapter[adapter]
        for adapter in FORMAL_RUNS_BY_ADAPTER
    )
    stage_projection = {
        stage: sum(counts[adapter] * by_adapter[adapter] for adapter in counts)
        for stage, counts in FORMAL_RUNS_BY_STAGE.items()
    }
    aggregate_bytes = 5 * 1024 * 1024
    package_bytes = expected + aggregate_bytes
    peak = expected + aggregate_bytes + package_bytes
    free = shutil.disk_usage(root.resolve()).free
    safety = 10 * 1024**3
    report = {
        "schema": "adapter_storage_pilot_v1",
        "status": "valid",
        "pilot_root": str(root.resolve()),
        "pilot_runs": len(inventory),
        "mean_run_bytes": float(inventory["bytes"].mean()),
        "adapter_summary_bytes_per_run": per_run,
        "bytes_per_run_by_adapter": by_adapter,
        "formal_new_runs": FORMAL_NEW_RUNS,
        "formal_runs_by_adapter": FORMAL_RUNS_BY_ADAPTER,
        "stage_projection_bytes": stage_projection,
        "expected_output_bytes": expected,
        "aggregate_and_figure_bytes": aggregate_bytes,
        "estimated_package_bytes": package_bytes,
        "estimated_peak_disk_bytes": peak,
        "peak_plus_10_gib_bytes": peak + safety,
        "current_free_space_bytes": free,
        "launch_allowed": free >= peak + safety,
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
