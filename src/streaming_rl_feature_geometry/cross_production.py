"""Production validation, reporting, and packaging for cross extensions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alpha_tuning import load_selected_learning_rates
from .cross_experiment import (
    aggregate_cross,
    config_hash,
    expected_cross_run_count,
    load_cross_config,
    validate_cross_results,
)
from .experiment import git_value, write_json
from .production_runtime import contained_path, repository_root


ANALYSIS_FILENAMES = {
    "aggregate_summary.csv",
    "condition_summary.csv",
    "aggregate_representation.csv",
    "aggregate_task_information.csv",
    "aggregate_decision_probe_by_position.csv",
    "robust_condition_summary.csv",
    "selected_learning_rates.csv",
    "catastrophic_failure_summary.csv",
    "config.json",
    "manifest.json",
    "analysis_manifest.json",
    "production_validation_violations.csv",
    "production_validation_violations.json",
    "trace_partition_inventory.json",
    "storage_inventory.json",
    "RESULTS_SUMMARY_ZH.md",
    "remote_launcher.log",
}


class ProductionValidationError(AssertionError):
    """Production validation failure with a persisted machine-readable report."""


def _validation_violations(
    root: Path,
    config: dict[str, Any],
    summaries: pd.DataFrame,
    finite_columns: list[str],
) -> list[dict[str, Any]]:
    threshold = float(config.get("extreme_finite_limit", 1e12))
    stage = str(config.get("experiment_stage", "fixed"))
    violations: list[dict[str, Any]] = []
    for summary_index, row in summaries.iterrows():
        for metric in finite_columns:
            if metric not in summaries:
                violations.append(
                    {
                        "stage": stage,
                        "environment": row.get("environment"),
                        "condition": row.get("condition"),
                        "seed": row.get("seed"),
                        "candidate_alpha": row.get("candidate_alpha"),
                        "metric": metric,
                        "value": None,
                        "threshold": threshold,
                        "run_directory": row.get("run_dir"),
                        "summary_row": int(summary_index),
                        "first_offending_interaction": None,
                        "classification": "missing_required_metric",
                    }
                )
                continue
            value = float(row[metric])
            classification = None
            if not np.isfinite(value):
                classification = "non_finite_required_metric"
            elif abs(value) > threshold:
                classification = "finite_extreme_required_metric"
            if classification is None:
                continue
            first_interaction = None
            run_directory = Path(str(row.get("run_dir", "")))
            validity_path = run_directory / "runtime_validity.json"
            if validity_path.is_file():
                try:
                    validity = json.loads(validity_path.read_text(encoding="utf-8"))
                    first_interaction = (validity.get("first_failure") or {}).get(
                        "interaction_index"
                    )
                except (OSError, ValueError):
                    first_interaction = None
            violations.append(
                {
                    "stage": stage,
                    "environment": row.get("environment"),
                    "condition": row.get("condition"),
                    "seed": int(row.get("seed")),
                    "candidate_alpha": row.get("candidate_alpha"),
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "run_directory": str(row.get("run_dir", "")),
                    "summary_row": int(summary_index),
                    "first_offending_interaction": first_interaction,
                    "classification": classification,
                }
            )
    violations.sort(
        key=lambda item: abs(float(item["value"])) if item["value"] is not None else float("inf"),
        reverse=True,
    )
    return violations


def _write_validation_violations(root: Path, violations: list[dict[str, Any]]) -> None:
    write_json(root / "production_validation_violations.json", violations)
    pd.DataFrame(violations).to_csv(root / "production_validation_violations.csv", index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_manifest(root: Path, config: dict[str, Any], summaries: pd.DataFrame) -> dict[str, Any]:
    aggregate_files = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.name != "analysis_manifest.json"
        and path.suffix.lower() in {".csv", ".json", ".md", ".log"}
    ]
    if (root / "figures").is_dir():
        aggregate_files.extend(path for path in (root / "figures").rglob("*") if path.is_file())
    aggregate_files.sort()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "config_hash": config_hash(config),
        "profile": config["profile"],
        "experiment_stage": config.get("experiment_stage", "fixed"),
        "expected_runs": expected_cross_run_count(config),
        "observed_runs": int(len(summaries)),
        "catastrophic_failures": int(summaries["catastrophic_failure"].sum()),
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in aggregate_files
        ],
    }


def _write_chinese_summary(root: Path, config: dict[str, Any], summaries: pd.DataFrame) -> None:
    failure_count = int(summaries["catastrophic_failure"].sum())
    lines = [
        "# Cross-extension 结果摘要",
        "",
        f"- Profile：`{config['profile']}`",
        f"- 阶段：`{config.get('experiment_stage', 'fixed')}`",
        f"- 完成 run：{len(summaries)} / {expected_cross_run_count(config)}",
        f"- Catastrophic failure：{failure_count}",
        "- whitening 是二阶变换；gaussian_moment 是无分布保证的在线矩塑形探索项。",
        "- 本摘要只陈述完整性与文件位置，不把尚未审阅的数值解释为正结果。",
        "",
        "详细 seed 结果见 `aggregate_summary.csv`，稳健统计见 `robust_condition_summary.csv`。",
    ]
    (root / "RESULTS_SUMMARY_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_and_validate(
    config_path: str | Path,
    run_dir: str | Path,
    selected_learning_rates: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild aggregates and fail closed on incomplete or mixed-stage results."""

    config = load_cross_config(config_path)
    root = Path(run_dir)
    if config.get("enforce_repository_containment", False):
        root = contained_path(root, repository_root(), label="production_result_root")
    aggregate_cross(root, config)
    validation = validate_cross_results(root, config)
    summaries = pd.read_csv(root / "aggregate_summary.csv")
    expected_stage = config.get("experiment_stage", "fixed")
    if "experiment_stage" not in summaries or set(summaries["experiment_stage"]) != {expected_stage}:
        raise AssertionError("tuning/evaluation stage mixing detected in aggregate summaries")
    finite_columns = [
        "final_performance",
        "final_window_reward",
        "mean_control_update_norm",
        "max_control_update_norm",
        "final_parameter_norm",
    ]
    violations = _validation_violations(root, config, summaries, finite_columns)
    _write_validation_violations(root, violations)
    if violations:
        preview = "\n".join(
            f"- {item['stage']} {item['environment']}/{item['condition']} "
            f"seed={item['seed']} alpha={item['candidate_alpha']} "
            f"{item['metric']}={item['value']} threshold={item['threshold']} "
            f"run={item['run_directory']} first_interaction={item['first_offending_interaction']}"
            for item in violations[:10]
        )
        raise ProductionValidationError(
            f"{len(violations)} production metric violation(s); complete reports: "
            f"{root / 'production_validation_violations.json'} and "
            f"{root / 'production_validation_violations.csv'}\n{preview}"
        )
    if expected_stage == "lr_eval":
        if selected_learning_rates is None:
            raise ValueError("lr_eval aggregation requires selected_learning_rates.csv")
        expected_pairs = {
            (environment, condition)
            for environment, spec in config["environments"].items()
            for condition in spec["conditions"]
        }
        selected = load_selected_learning_rates(selected_learning_rates, expected_pairs)
        for row in summaries.itertuples(index=False):
            expected_alpha = selected[(row.environment, row.condition)]
            if not np.isclose(float(row.base_controller_alpha), expected_alpha):
                raise AssertionError(
                    f"evaluation alpha mismatch for {row.environment}/{row.condition}"
                )
    _write_chinese_summary(root, config, summaries)
    analysis_manifest = _analysis_manifest(root, config, summaries)
    write_json(root / "analysis_manifest.json", analysis_manifest)
    return {**validation, "analysis_manifest": str((root / "analysis_manifest.json").resolve())}


def _copy_analysis_core(suite_dir: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    stage_roots = sorted(path.parent for path in suite_dir.glob("*/config.json"))
    if not stage_roots and (suite_dir / "config.json").exists():
        stage_roots = [suite_dir]
    for stage_root in stage_roots:
        relative_stage = Path(stage_root.name) if stage_root != suite_dir else Path("stage")
        target = destination / relative_stage
        target.mkdir(parents=True, exist_ok=True)
        for name in ANALYSIS_FILENAMES:
            source = stage_root / name
            if source.is_file():
                shutil.copy2(source, target / name)
                copied.append(str(relative_stage / name))
        figures = stage_root / "figures"
        if figures.is_dir():
            shutil.copytree(figures, target / "figures")
            copied.extend(
                str(relative_stage / "figures" / path.name)
                for path in sorted(figures.iterdir())
                if path.is_file()
            )
        selected = stage_root / "selected_learning_rates.csv"
        if selected.is_file():
            shutil.copy2(selected, destination / "selected_learning_rates.csv")
    logs = suite_dir / "logs"
    if logs.is_dir():
        shutil.copytree(logs, destination / "remote_logs")
        copied.extend(
            str(Path("remote_logs") / path.relative_to(logs))
            for path in sorted(logs.rglob("*"))
            if path.is_file()
        )
    return copied


def _write_checksum(path: Path) -> Path:
    checksum = path.with_name(path.name + ".sha256")
    checksum.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="ascii")
    expected = checksum.read_text(encoding="ascii").split()[0]
    if sha256_file(path) != expected:
        raise AssertionError(f"SHA-256 verification failed for {path}")
    return checksum


def package_suite(
    suite_dir: str | Path, run_name: str, output_dir: str | Path | None = None
) -> dict[str, str]:
    """Create verified full and analysis-core archives without deleting results."""

    suite = Path(suite_dir).resolve()
    if not suite.is_dir():
        raise FileNotFoundError(f"suite directory does not exist: {suite}")
    destination = Path(output_dir).resolve() if output_dir else suite.parent / "packages"
    try:
        repository = repository_root(suite)
    except RuntimeError:
        repository = None
    if repository is not None:
        contained_path(suite, repository, label="suite_dir")
        destination = contained_path(destination, repository, label="package_output_dir")
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"cross-extension-{run_name}"
    full = destination / f"{stem}-full.tar.gz"
    analysis = destination / f"{stem}-analysis-core.tar.gz"
    temporary_parent = (
        (repository / ".runtime" / "tmp") if repository is not None else suite.parent
    )
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="cross-extension-package-", dir=temporary_parent
    ) as temporary:
        temporary_root = Path(temporary)
        temporary_full = temporary_root / full.name
        temporary_analysis = temporary_root / analysis.name

        def exclude_packages(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
            return None if "packages" in Path(member.name).parts else member

        with tarfile.open(temporary_full, "w:gz") as archive:
            archive.add(suite, arcname=stem, filter=exclude_packages)
        core = temporary_root / f"{stem}-analysis-core"
        core.mkdir()
        copied = _copy_analysis_core(suite, core)
        if not any(path.endswith("aggregate_summary.csv") for path in copied):
            raise AssertionError("analysis-core has no aggregate_summary.csv")
        suite_manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "source_suite": str(suite),
            "files": sorted(copied),
        }
        write_json(core / "analysis_manifest.json", suite_manifest)
        (core / "README_ZH.md").write_text(
            "# 分析轻量包\n\n各实验阶段保存在同名子目录；优先查看 robust_condition_summary.csv。\n",
            encoding="utf-8",
        )
        with tarfile.open(temporary_analysis, "w:gz") as archive:
            archive.add(core, arcname=core.name)
        temporary_full.replace(full)
        temporary_analysis.replace(analysis)
    full_sha = _write_checksum(full)
    analysis_sha = _write_checksum(analysis)
    with tarfile.open(full, "r:gz") as archive:
        if not archive.getmembers():
            raise AssertionError("full archive is empty")
    with tarfile.open(analysis, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
        if not any(name.endswith("analysis_manifest.json") for name in names):
            raise AssertionError("analysis archive is missing analysis_manifest.json")
    return {
        "full": str(full),
        "full_sha256": str(full_sha),
        "analysis_core": str(analysis),
        "analysis_core_sha256": str(analysis_sha),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", required=True)
    aggregate.add_argument("--run-dir", required=True)
    aggregate.add_argument("--selected-learning-rates")
    package = subparsers.add_parser("package")
    package.add_argument("--suite-dir", required=True)
    package.add_argument("--run-name", required=True)
    package.add_argument("--output-dir")
    args = parser.parse_args()
    if args.command == "aggregate":
        result = aggregate_and_validate(
            args.config, args.run_dir, args.selected_learning_rates
        )
    else:
        result = package_suite(args.suite_dir, args.run_name, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
