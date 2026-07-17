"""Pilot-calibrated storage inventory and formal-suite peak projection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alpha_tuning import candidate_alphas
from .compact_storage import COMPACT_SCHEMA_VERSION
from .cross_experiment import load_cross_config
from .experiment import config_hash, write_json
from .production_runtime import DEFAULT_SAFE_MARGIN_BYTES, DEFAULT_TARGET_PEAK_BYTES


INTERACTION_SCALED_CATEGORIES = {"compact_traces"}
ANALYSIS_LOG_BUDGET_BYTES = 16 * 1024**2
TAR_HEADER_BUDGET_PER_FILE = 1024
EXPECTED_FORMAL_PROFILES = {
    "cross_extension_fixed_full",
    "cross_extension_lr_tune_full",
    "cross_extension_lr_eval_full",
    "cross_extension_norm_scaled_full",
}


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _category(path: Path) -> str:
    name = path.name
    if name in {"strided_trace.npz", "decision_event_trace.npz", "disturbance_event_trace.npz"}:
        return "compact_traces"
    if name in {"model_state.npz", "prediction_feature_summary.npz"}:
        return "model_transform_state"
    if name == "diagnostic_samples.npz":
        return "selected_diagnostic_samples"
    if path.suffix == ".csv":
        return "scalar_and_probe_csv"
    if path.suffix in {".json", ".log", ".md"}:
        return "manifests_and_logs"
    if path.suffix in {".png", ".svg", ".pdf"}:
        return "figures"
    if path.suffix in {".gz", ".tar"} or name.endswith(".sha256"):
        return "packages"
    return "other"


def _pilot_records(pilot_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted((pilot_root / "runs").rglob("summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = summary_path.parent
        category_bytes: dict[str, int] = defaultdict(int)
        for path in run_dir.rglob("*"):
            if path.is_file():
                category_bytes[_category(path)] += path.stat().st_size
        total = sum(category_bytes.values())
        interactions = int(summary["interactions"])
        records.append(
            {
                "environment": str(summary["environment"]),
                "condition": str(summary["condition"]),
                "seed": int(summary["seed"]),
                "interactions": interactions,
                "run_bytes": total,
                "bytes_per_interaction": total / interactions,
                **{f"bytes__{key}": value for key, value in category_bytes.items()},
            }
        )
    if not records:
        raise ValueError(f"storage pilot has no per-run summaries: {pilot_root}")
    return records


def _pilot_validity_evidence(pilot_root: Path, expected_runs: int) -> dict[str, Any]:
    manifests = sorted((pilot_root / "runs").rglob("manifest.json"))
    if len(manifests) != expected_runs:
        raise ValueError(
            f"storage pilot manifest count mismatch: expected={expected_runs} observed={len(manifests)}"
        )
    commits: set[str] = set()
    dirty_flags: set[bool] = set()
    config_hashes: set[str] = set()
    schema_versions: set[int] = set()
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        runtime_path = path.parent / "runtime_validity.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if (
            manifest.get("exit_status") != "ok"
            or not manifest.get("resume_eligible", False)
            or runtime.get("status") != "valid"
            or not runtime.get("resume_eligible", False)
        ):
            raise ValueError(f"storage pilot contains invalid or incomplete run: {path.parent}")
        commits.add(str(manifest.get("git_commit", "unknown")))
        dirty_flags.add(bool(manifest.get("git_dirty", True)))
        config_hashes.add(str(manifest.get("config_hash", "unknown")))
        schema_versions.add(int(manifest.get("result_schema_version", -1)))
    if len(commits) != 1 or "unknown" in commits:
        raise ValueError(f"storage pilot must use one known Git commit: {sorted(commits)}")
    if len(config_hashes) != 1 or "unknown" in config_hashes:
        raise ValueError(
            f"storage pilot must use one known config hash: {sorted(config_hashes)}"
        )
    return {
        "pilot_git_commits": sorted(commits),
        "pilot_git_dirty_flags": sorted(dirty_flags),
        "pilot_config_hashes": sorted(config_hashes),
        "pilot_result_schema_versions": sorted(schema_versions),
        "pilot_valid_runs": expected_runs,
    }


def project_storage(
    pilot_root: str | Path,
    formal_configs: list[str | Path],
    *,
    safety_factor: float = 1.35,
    package_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Project the full suite from actual per-environment/condition pilot files."""

    if safety_factor < 1.0:
        raise ValueError("storage safety_factor must be at least one")
    pilot = Path(pilot_root).resolve()
    records = _pilot_records(pilot)
    pilot_evidence = _pilot_validity_evidence(pilot, len(records))
    category_names = sorted(
        {
            key.removeprefix("bytes__")
            for record in records
            for key in record
            if key.startswith("bytes__")
        }
    )
    calibration: dict[tuple[str, str], dict[str, float]] = {}
    for record in records:
        key = (record["environment"], record["condition"])
        values = calibration.setdefault(key, {})
        for category in category_names:
            observed = float(record.get(f"bytes__{category}", 0))
            if category in INTERACTION_SCALED_CATEGORIES:
                observed /= int(record["interactions"])
            values[category] = max(values.get(category, 0.0), observed)

    pilot_root_file_bytes = sum(
        path.stat().st_size for path in pilot.iterdir() if path.is_file()
    )
    pilot_root_file_count = sum(1 for path in pilot.iterdir() if path.is_file())
    pilot_figure_root = pilot / "figures"
    pilot_figure_bytes = _tree_bytes(pilot_figure_root) if pilot_figure_root.is_dir() else 0
    pilot_figure_count = (
        sum(1 for path in pilot_figure_root.rglob("*") if path.is_file())
        if pilot_figure_root.is_dir()
        else 0
    )
    pilot_run_file_count = sum(
        1 for path in (pilot / "runs").rglob("*") if path.is_file()
    )
    files_per_run = pilot_run_file_count / len(records)
    stages: list[dict[str, Any]] = []
    projected_results = 0.0
    projected_analysis = 0.0
    projected_runs = 0
    projected_category_bytes: dict[str, float] = defaultdict(float)
    formal_config_evidence: list[dict[str, Any]] = []
    for config_path in formal_configs:
        config = load_cross_config(config_path)
        formal_config_evidence.append(
            {
                "path": str(config_path),
                "profile": str(config["profile"]),
                "config_hash": config_hash(config),
                "result_schema_version": int(config.get("result_schema_version", -1)),
                "storage_schema": str(config.get("storage_schema", "legacy_csv")),
            }
        )
        multiplier = 1
        if config.get("experiment_stage") == "lr_tune":
            base = float(config["control_alpha"])
            multiplier = len(
                candidate_alphas(base, list(map(float, config["alpha_tuning"]["multipliers"])))
            )
        stage_run_bytes = 0.0
        stage_category_bytes: dict[str, float] = defaultdict(float)
        stage_runs = 0
        for environment, specification in config["environments"].items():
            interactions = int(specification["interactions"])
            for condition in specification["conditions"]:
                key = (environment, condition)
                if key not in calibration:
                    raise ValueError(f"storage pilot missing environment/condition {key}")
                count = len(config["seeds"]) * multiplier
                stage_runs += count
                for category, observed in calibration[key].items():
                    value = observed * count
                    if category in INTERACTION_SCALED_CATEGORIES:
                        value *= interactions
                    stage_category_bytes[category] += value
                    stage_run_bytes += value
        # Aggregate CSVs grow with run count, while the figure set and other
        # rendering outputs are stage-level.  Treat both as uncompressed source
        # bytes and apply the same safety factor as the run partitions.
        stage_analysis_bytes = (
            pilot_root_file_bytes * stage_runs / len(records) + pilot_figure_bytes
        )
        stage_bytes = (stage_run_bytes + stage_analysis_bytes) * safety_factor
        projected_results += stage_bytes
        projected_analysis += stage_analysis_bytes * safety_factor
        for category, value in stage_category_bytes.items():
            projected_category_bytes[category] += value * safety_factor
        projected_runs += stage_runs
        stages.append(
            {
                "config": str(config_path),
                "profile": config["profile"],
                "runs": stage_runs,
                "projected_result_bytes": int(stage_bytes),
            }
        )
    projected_result_bytes = int(projected_results)
    projected_analysis_bytes = int(projected_analysis)
    # Peak accounting does not assume that gzip saves any bytes.  The archive
    # estimates use the uncompressed source size plus explicit tar headers;
    # actual pilot archive sizes remain in the report as calibration evidence.
    pilot_package_root = Path(package_dir).resolve() if package_dir else None
    pilot_full_packages = (
        sorted(pilot_package_root.glob("*-full.tar.gz")) if pilot_package_root else []
    )
    pilot_analysis_packages = (
        sorted(pilot_package_root.glob("*-analysis-core.tar.gz")) if pilot_package_root else []
    )
    pilot_full_package_bytes = sum(path.stat().st_size for path in pilot_full_packages)
    pilot_analysis_core_bytes = sum(path.stat().st_size for path in pilot_analysis_packages)
    pilot_tree_bytes = _tree_bytes(pilot)
    projected_source_file_count = int(
        np.ceil(
            files_per_run * projected_runs
            + (pilot_root_file_count + pilot_figure_count) * len(formal_configs)
        )
    )
    full_tar_overhead = (
        projected_source_file_count * TAR_HEADER_BUDGET_PER_FILE + 1024**2
    )
    projected_full_package = projected_result_bytes + full_tar_overhead
    projected_analysis_file_count = (
        (pilot_root_file_count + pilot_figure_count) * len(formal_configs) + 32
    )
    analysis_tar_overhead = (
        projected_analysis_file_count * TAR_HEADER_BUDGET_PER_FILE + 1024**2
    )
    projected_analysis_core = (
        projected_analysis_bytes + ANALYSIS_LOG_BUDGET_BYTES + analysis_tar_overhead
    )
    projected_peak = (
        projected_result_bytes + projected_full_package + projected_analysis_core
    )
    category_totals: dict[str, int] = defaultdict(int)
    for record in records:
        for key, value in record.items():
            if key.startswith("bytes__"):
                category_totals[key.removeprefix("bytes__")] += int(value)
    largest_categories = [
        {
            "category": category,
            "pilot_bytes": size,
            "projected_bytes": int(projected_category_bytes.get(category, 0.0)),
            "scaling": (
                "per_interaction"
                if category in INTERACTION_SCALED_CATEGORIES
                else "per_run"
            ),
        }
        for category, size in sorted(category_totals.items(), key=lambda item: item[1], reverse=True)
    ]
    largest_categories.append(
        {
            "category": "stage_level_analysis",
            "pilot_bytes": pilot_root_file_bytes + pilot_figure_bytes,
            "projected_bytes": projected_analysis_bytes,
            "scaling": "aggregate_rows_plus_per_stage_figures",
        }
    )
    largest_categories.sort(key=lambda item: item["projected_bytes"], reverse=True)
    return {
        "pilot_root": str(pilot),
        "pilot_runs": len(records),
        "pilot_tree_bytes": pilot_tree_bytes,
        "pilot_package_dir": str(pilot_package_root) if pilot_package_root else None,
        "pilot_full_package_bytes": pilot_full_package_bytes,
        "pilot_analysis_core_bytes": pilot_analysis_core_bytes,
        "pilot_root_file_bytes": pilot_root_file_bytes,
        "pilot_figure_bytes": pilot_figure_bytes,
        "interaction_scaled_categories": sorted(INTERACTION_SCALED_CATEGORIES),
        "package_projection_method": "uncompressed_source_plus_tar_header_budget",
        "safety_factor": safety_factor,
        "formal_runs": projected_runs,
        "formal_configs": formal_config_evidence,
        "stages": stages,
        "projected_result_bytes": projected_result_bytes,
        "projected_aggregation_bytes": projected_analysis_bytes,
        "projected_full_package_bytes": projected_full_package,
        "projected_analysis_core_bytes": projected_analysis_core,
        "projected_peak_bytes": projected_peak,
        "target_peak_bytes": DEFAULT_TARGET_PEAK_BYTES,
        "required_safe_margin_bytes": DEFAULT_SAFE_MARGIN_BYTES,
        "within_50_gib_target": projected_peak <= DEFAULT_TARGET_PEAK_BYTES,
        "largest_file_categories": largest_categories,
        "pilot_run_inventory": records,
        **pilot_evidence,
    }


def validated_production_peak_bytes(
    report_path: str | Path, expected_commit: str
) -> int:
    """Validate that a storage report is safe and belongs to this production commit."""

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not report.get("within_50_gib_target", False):
        raise ValueError("storage projection does not satisfy the 50 GiB target")
    if report.get("formal_runs") != 4200:
        raise ValueError(f"storage projection run count mismatch: {report.get('formal_runs')}")
    if report.get("pilot_runs") != 50 or report.get("pilot_valid_runs") != 50:
        raise ValueError("storage projection is not backed by 50 valid pilot runs")
    if report.get("pilot_git_commits") != [expected_commit]:
        raise ValueError(
            f"storage pilot commit mismatch: expected={expected_commit} "
            f"observed={report.get('pilot_git_commits')}"
        )
    if report.get("pilot_git_dirty_flags") != [False]:
        raise ValueError("storage pilot was produced from a dirty worktree")
    if report.get("pilot_result_schema_versions") != [COMPACT_SCHEMA_VERSION]:
        raise ValueError("storage pilot result schema is not compact v2")
    formal = report.get("formal_configs", [])
    profiles = {row.get("profile") for row in formal}
    if profiles != EXPECTED_FORMAL_PROFILES or any(
        row.get("storage_schema") != "compact_v2"
        or row.get("result_schema_version") != COMPACT_SCHEMA_VERSION
        for row in formal
    ):
        raise ValueError(
            "storage projection does not describe the four compact-v2 formal configs"
        )
    if report.get("package_projection_method") != "uncompressed_source_plus_tar_header_budget":
        raise ValueError(
            "storage projection does not use the required conservative package budget"
        )
    peak = int(report["projected_peak_bytes"])
    if peak <= 0 or peak > int(report["target_peak_bytes"]):
        raise ValueError("storage projection peak is invalid or exceeds its target")
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--formal-config", action="append", required=True)
    parser.add_argument("--package-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--inventory-csv")
    parser.add_argument("--safety-factor", type=float, default=1.35)
    arguments = parser.parse_args()
    report = project_storage(
        arguments.pilot_dir,
        arguments.formal_config,
        safety_factor=arguments.safety_factor,
        package_dir=arguments.package_dir,
    )
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    if arguments.inventory_csv:
        inventory_path = Path(arguments.inventory_csv)
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report["pilot_run_inventory"]).to_csv(inventory_path, index=False)
    print(json.dumps(report, sort_keys=True))
    if not report["within_50_gib_target"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
