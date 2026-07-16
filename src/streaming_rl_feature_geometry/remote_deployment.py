"""Remote deployment planning, completeness checks, aggregation, and packaging.

This module orchestrates existing causal CPU experiment runners.  It does not
change agent updates or introduce a GPU execution path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .alpha_tuning import candidate_alphas
from .cross_experiment import aggregate_cross, validate_cross_results
from .experiment import aggregate, config_hash, dependency_versions, validate_results


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_kind(config: dict[str, Any]) -> str:
    if isinstance(config.get("environments"), dict):
        return "cross"
    if isinstance(config.get("conditions"), list):
        return "core"
    raise ValueError("config is not an executable core or cross-environment profile")


def selected_seeds(
    config: dict[str, Any], seed_start: int | None = None, seed_end: int | None = None
) -> list[int]:
    seeds = sorted(map(int, config.get("seeds", [])))
    if not seeds:
        raise ValueError("config has no seeds")
    if seed_start is not None:
        seeds = [seed for seed in seeds if seed >= seed_start]
    if seed_end is not None:
        seeds = [seed for seed in seeds if seed <= seed_end]
    if not seeds:
        raise ValueError("seed range selects no configured seeds")
    return seeds


def expected_run_count(config: dict[str, Any], seeds: list[int] | None = None) -> int:
    seeds = seeds or selected_seeds(config)
    if experiment_kind(config) == "core":
        return len(config["conditions"]) * len(seeds)
    multiplier = 1
    if config.get("experiment_stage") == "lr_tune":
        multiplier = len(config.get("alpha_tuning", {}).get("multipliers", []))
        if multiplier < 1:
            raise ValueError("lr_tune config has no alpha multipliers")
    return sum(
        len(specification["conditions"]) * len(seeds) * multiplier
        for specification in config["environments"].values()
    )


def config_plan(
    path: str | Path, seed_start: int | None = None, seed_end: int | None = None
) -> dict[str, Any]:
    config = read_json(path)
    kind = experiment_kind(config)
    seeds = selected_seeds(config, seed_start, seed_end)
    if kind == "core":
        environments = ["tmaze"]
        representations = list(config["conditions"])
        banks = [str(config.get("gvf_bank", "mixed"))]
        interactions: int | dict[str, int] = int(config["total_interactions"])
    else:
        environments = list(config["environments"])
        representations = sorted(
            {
                condition
                for specification in config["environments"].values()
                for condition in specification["conditions"]
            }
        )
        banks = sorted(
            {
                str(specification.get("bank", "mixed"))
                for specification in config["environments"].values()
            }
        )
        interactions = {
            environment: int(specification["interactions"])
            for environment, specification in config["environments"].items()
        }
    return {
        "config": str(path),
        "config_sha256": sha256_file(path),
        "normalized_config_hash": config_hash(config),
        "kind": kind,
        "profile": str(config["profile"]),
        "seeds": seeds,
        "environments": environments,
        "representations": representations,
        "banks": banks,
        "horizons": list(config.get("horizons", [])),
        "interactions": interactions,
        "expected_runs": expected_run_count(config, seeds),
        "configured_output_dir": str(config["output_dir"]),
    }


def _alpha_path(alpha: float) -> str:
    return f"alpha_{alpha:.12g}".replace("+", "").replace(".", "p")


def expected_run_directories(
    root: str | Path, config: dict[str, Any]
) -> list[tuple[str, Path]]:
    root = Path(root)
    records: list[tuple[str, Path]] = []
    if experiment_kind(config) == "core":
        for condition in config["conditions"]:
            for seed in selected_seeds(config):
                label = f"{condition}/seed_{seed:03d}"
                records.append((label, root / "runs" / condition / f"seed_{seed:03d}"))
        return records
    for environment, specification in config["environments"].items():
        for condition in specification["conditions"]:
            alpha_paths: list[str | None] = [None]
            if config.get("experiment_stage") == "lr_tune":
                base = float(specification.get("control_alpha", config["control_alpha"]))
                alpha_paths = [
                    _alpha_path(alpha)
                    for alpha, _ in candidate_alphas(
                        base, list(map(float, config["alpha_tuning"]["multipliers"]))
                    )
                ]
            for alpha_path in alpha_paths:
                for seed in selected_seeds(config):
                    parts = [environment, condition]
                    if alpha_path is not None:
                        parts.append(alpha_path)
                    parts.append(f"seed_{seed:03d}")
                    label = "/".join(parts)
                    records.append((label, root / "runs" / Path(*parts)))
    return records


def inspect_run(root: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(root)
    if config is None:
        config = read_json(root / "config.json")
    missing: list[str] = []
    failed: list[str] = []
    completed: list[str] = []
    for label, run_dir in expected_run_directories(root, config):
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.csv"
        if not manifest_path.is_file() or not summary_path.is_file():
            missing.append(label)
            continue
        try:
            manifest = read_json(manifest_path)
            with summary_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, ValueError, csv.Error) as error:
            failed.append(f"{label}:unreadable:{type(error).__name__}")
            continue
        status = rows[0].get("run_status") if len(rows) == 1 else "invalid-summary"
        if manifest.get("exit_status") == "ok" and status == "ok":
            completed.append(label)
        else:
            failed.append(
                f"{label}:manifest={manifest.get('exit_status')}:summary={status}"
            )
    return {
        "result_dir": str(root.resolve()),
        "expected_runs": len(missing) + len(failed) + len(completed),
        "completed_runs": len(completed),
        "missing_runs": missing,
        "failed_runs": failed,
        "ok": not missing and not failed,
    }


def aggregate_single(root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(root)
    config = read_json(config_path or root / "config.json")
    inspection = inspect_run(root, config)
    if not inspection["ok"]:
        return inspection
    if experiment_kind(config) == "cross":
        aggregate_cross(root, config)
        validation = validate_cross_results(root, config)
    else:
        aggregate(root)
        validation = validate_results(root, config)
    inspection["validation"] = validation
    return inspection


def _git_value(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cpu_model() -> str:
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _gpu_detection() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"nvidia_smi_available": False, "devices": [], "backend_used": "none"}
    process = subprocess.run(
        [executable, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    devices = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    return {
        "nvidia_smi_available": process.returncode == 0,
        "devices": devices,
        "backend_used": "none",
        "note": "GPU inventory is recorded only; experiments use CPU NumPy processes.",
    }


def host_metadata() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "dependency_versions": dependency_versions(),
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "gpu_detection": _gpu_detection(),
        "parallelism": "CPU process-level condition-by-seed tasks; BLAS threads capped at one",
    }


def write_launch_manifest(
    result_dir: str | Path,
    config_path: str | Path,
    command: str,
    workers: int,
    start_time: str,
    end_time: str,
    exit_status: str,
) -> dict[str, Any]:
    result_dir = Path(result_dir)
    value = {
        **host_metadata(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "exact_command": command,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "workers": int(workers),
        "start_time": start_time,
        "end_time": end_time,
        "exit_status": exit_status,
        "result_path": str(result_dir.resolve()),
    }
    write_json(result_dir / "remote_launch_manifest.json", value)
    return value


def load_suite_plan(
    suite_path: str | Path, selected_stages: set[str] | None = None
) -> list[dict[str, Any]]:
    suite_path = Path(suite_path)
    suite = read_json(suite_path)
    if suite.get("schema_version") != 1 or not isinstance(suite.get("stages"), list):
        raise ValueError("remote suite config must use schema_version 1 and a stages list")
    records: list[dict[str, Any]] = []
    for stage in suite["stages"]:
        stage_id = str(stage["id"])
        if not stage.get("enabled", True):
            continue
        if selected_stages is not None and stage_id not in selected_stages:
            continue
        if stage.get("kind") == "cross_extension":
            records.append(
                {
                    "batch_id": stage_id,
                    "stage": stage_id,
                    "kind": "cross_extension",
                    "config": "configs/cross_extension_fixed_full.json",
                    "run_name": stage_id,
                    "extension_stage": str(stage.get("extension_stage", "all")),
                }
            )
            continue
        plan_path = suite_path.parent.parent / stage["plan"]
        plan = read_json(plan_path)
        for run in plan.get("runs", []):
            run_id = str(run["id"])
            records.append(
                {
                    "batch_id": f"{stage_id}__{run_id}",
                    "stage": stage_id,
                    "kind": str(run["kind"]),
                    "config": str(run["config"]),
                    "run_name": run_id,
                    "extension_stage": "",
                }
            )
    if not records:
        raise ValueError("suite stage selection produced no batches")
    ids = [record["batch_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("suite batch ids must be unique")
    return records


def load_run_plan(plan_path: str | Path) -> list[dict[str, str]]:
    plan_path = Path(plan_path)
    plan = read_json(plan_path)
    records = []
    for run in plan.get("runs", []):
        record = {
            "id": str(run["id"]),
            "kind": str(run["kind"]),
            "config": str(run["config"]),
        }
        actual = plan_path.parent.parent / record["config"]
        configured_kind = experiment_kind(read_json(actual))
        if configured_kind != record["kind"]:
            raise ValueError(
                f"{record['id']} declares {record['kind']} but config is {configured_kind}"
            )
        records.append(record)
    if not records:
        raise ValueError("plan has no executable runs")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("plan run ids must be unique")
    return records


def initialize_suite(
    suite_dir: str | Path,
    run_name: str,
    suite_config: str | Path,
    workers: int,
    stages: set[str] | None,
) -> dict[str, Any]:
    suite_dir = Path(suite_dir)
    records = load_suite_plan(suite_config, stages)
    suite_dir.mkdir(parents=True, exist_ok=False)
    for record in records:
        record["status"] = "planned"
        record["result_dir"] = ""
    value = {
        **host_metadata(),
        "schema_version": 1,
        "run_name": run_name,
        "suite_config": str(suite_config),
        "suite_config_sha256": sha256_file(suite_config),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "workers": int(workers),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": None,
        "exit_status": "running",
        "batches": records,
    }
    write_json(suite_dir / "suite_manifest.json", value)
    return value


def record_batch(
    suite_dir: str | Path, batch_id: str, status: str, result_dir: str, error: str = ""
) -> dict[str, Any]:
    manifest_path = Path(suite_dir) / "suite_manifest.json"
    value = read_json(manifest_path)
    matches = [record for record in value["batches"] if record["batch_id"] == batch_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate batch id {batch_id}")
    matches[0].update(
        status=status,
        result_dir=result_dir,
        error=error,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    if status == "failed":
        value["exit_status"] = "failed"
    write_json(manifest_path, value)
    return value


def finalize_suite(suite_dir: str | Path, status: str) -> dict[str, Any]:
    manifest_path = Path(suite_dir) / "suite_manifest.json"
    value = read_json(manifest_path)
    value["exit_status"] = status
    value["end_time"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, value)
    return value


def _batch_roots(suite_dir: Path) -> list[Path]:
    roots = []
    for manifest in suite_dir.rglob("manifest.json"):
        parent = manifest.parent
        if (parent / "config.json").is_file() and (parent / "runs").is_dir():
            roots.append(parent)
    return sorted(set(roots))


def aggregate_suite(suite_dir: str | Path) -> dict[str, Any]:
    suite_dir = Path(suite_dir)
    suite_manifest_path = suite_dir / "suite_manifest.json"
    suite_manifest = read_json(suite_manifest_path) if suite_manifest_path.is_file() else {}
    inventory: list[dict[str, Any]] = []
    missing_rows: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []
    combined: list[pd.DataFrame] = []
    for root in _batch_roots(suite_dir):
        config = read_json(root / "config.json")
        inspection = inspect_run(root, config)
        for label in inspection["missing_runs"]:
            missing_rows.append({"batch": str(root.relative_to(suite_dir)), "run": label})
        for label in inspection["failed_runs"]:
            failed_rows.append({"batch": str(root.relative_to(suite_dir)), "run": label})
        validation: dict[str, Any] | None = None
        if inspection["ok"]:
            try:
                validation = aggregate_single(root, root / "config.json").get("validation")
            except Exception as error:  # preserve the exact failed batch in the report
                failed_rows.append(
                    {
                        "batch": str(root.relative_to(suite_dir)),
                        "run": f"aggregation:{type(error).__name__}:{error}",
                    }
                )
        summary_path = root / "aggregate_summary.csv"
        if summary_path.is_file():
            frame = pd.read_csv(summary_path)
            frame = pd.concat(
                [
                    pd.Series(
                        [str(root.relative_to(suite_dir))] * len(frame), name="batch"
                    ),
                    frame,
                ],
                axis=1,
            )
            combined.append(frame)
        inventory.append(
            {
                "batch": str(root.relative_to(suite_dir)),
                "profile": config.get("profile"),
                "kind": experiment_kind(config),
                "expected_runs": inspection["expected_runs"],
                "completed_runs": inspection["completed_runs"],
                "missing_runs": len(inspection["missing_runs"]),
                "failed_runs": len(inspection["failed_runs"]),
                "validated": validation is not None,
            }
        )
    planned_failed = [
        record
        for record in suite_manifest.get("batches", [])
        if record.get("status") not in {"ok", "completed"}
    ]
    for record in planned_failed:
        failed_rows.append(
            {
                "batch": str(record.get("batch_id", "unknown")),
                "run": f"suite-status:{record.get('status', 'missing')}",
            }
        )
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(suite_dir / "remote_aggregate_inventory.csv", index=False)
    pd.DataFrame(missing_rows, columns=["batch", "run"]).to_csv(
        suite_dir / "remote_missing_runs.csv", index=False
    )
    pd.DataFrame(failed_rows, columns=["batch", "run"]).to_csv(
        suite_dir / "remote_failed_runs.csv", index=False
    )
    if combined:
        pd.concat(combined, ignore_index=True, sort=False).to_csv(
            suite_dir / "remote_aggregate_all_runs.csv", index=False
        )
    figures = suite_dir / "figures"
    figures.mkdir(exist_ok=True)
    if len(inventory_frame):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(max(7, len(inventory_frame) * 1.1), 4.5))
        axis.bar(inventory_frame["batch"], inventory_frame["expected_runs"], label="expected")
        axis.bar(inventory_frame["batch"], inventory_frame["completed_runs"], label="completed")
        axis.set_ylabel("runs")
        axis.set_title("Remote batch completion")
        axis.tick_params(axis="x", rotation=35)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures / "remote_batch_completion.png", dpi=160)
        plt.close(figure)
    logs_summary = []
    for log in sorted(suite_dir.rglob("*.log")):
        if "packages" in log.parts or (
            "logs" not in log.parts and log.name not in {"remote_launcher.log"}
        ):
            continue
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        logs_summary.append(f"===== {log.relative_to(suite_dir)} =====\n" + "\n".join(lines[-100:]))
    (suite_dir / "logs_summary.txt").write_text("\n\n".join(logs_summary), encoding="utf-8")
    ok = bool(inventory) and not missing_rows and not failed_rows
    report = {
        "suite_dir": str(suite_dir.resolve()),
        "batches_found": len(inventory),
        "missing_runs": len(missing_rows),
        "failed_runs": len(failed_rows),
        "ok": ok,
    }
    write_json(suite_dir / "remote_aggregation_manifest.json", report)
    (suite_dir / "REMOTE_RESULTS_SUMMARY_ZH.md").write_text(
        "# 远程结果汇总\n\n"
        f"- 批次数：{len(inventory)}\n"
        f"- 缺失 runs：{len(missing_rows)}\n"
        f"- 失败 runs：{len(failed_rows)}\n"
        f"- 状态：{'通过' if ok else '失败或不完整'}\n\n"
        "失败与缺失不会被计入成功结果。详见 remote_missing_runs.csv 和 remote_failed_runs.csv。\n",
        encoding="utf-8",
    )
    return report


def package_suite(suite_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    suite_dir = Path(suite_dir)
    if not suite_dir.is_dir():
        raise FileNotFoundError(suite_dir)
    output_dir = Path(output_dir) if output_dir else suite_dir / "packages"
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_value("rev-parse", "--short=12", "HEAD")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = output_dir / f"remote-results-{suite_dir.name}-{commit}-{timestamp}.tar.gz"
    candidates: set[Path] = set()
    exact_names = {
        "suite_manifest.json",
        "remote_aggregation_manifest.json",
        "remote_aggregate_inventory.csv",
        "remote_aggregate_all_runs.csv",
        "remote_missing_runs.csv",
        "remote_failed_runs.csv",
        "REMOTE_RESULTS_SUMMARY_ZH.md",
        "logs_summary.txt",
        "remote_launch_manifest.json",
        "config.json",
        "manifest.json",
        "analysis_manifest.json",
        "selected_learning_rates.csv",
    }
    for path in suite_dir.rglob("*"):
        if not path.is_file() or output_dir in path.parents:
            continue
        if (
            path.name in exact_names
            or path.name.startswith("aggregate_")
            or path.name.endswith("summary.csv")
            or ("figures" in path.parts and "runs" not in path.parts)
        ):
            candidates.add(path)
    inventory = {
        "suite_dir": str(suite_dir.resolve()),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [str(path.relative_to(suite_dir)) for path in sorted(candidates)],
    }
    inventory_path = suite_dir / "package_inventory.json"
    write_json(inventory_path, inventory)
    candidates.add(inventory_path)
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(candidates):
            handle.add(path, arcname=f"{suite_dir.name}/{path.relative_to(suite_dir)}")
        repository_root = Path(__file__).resolve().parents[2]
        for config in sorted((repository_root / "configs").glob("remote_*.json")):
            handle.add(config, arcname=f"{suite_dir.name}/repository_configs/{config.name}")
    checksum = sha256_file(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    return {"archive": str(archive), "sha256": checksum, "checksum_file": str(checksum_path)}


def _parse_stages(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_config_parser = subparsers.add_parser("plan-config")
    plan_config_parser.add_argument("--config", required=True)
    plan_config_parser.add_argument("--seed-start", type=int)
    plan_config_parser.add_argument("--seed-end", type=int)
    plan_config_parser.add_argument("--lines", action="store_true")

    inspect_parser = subparsers.add_parser("inspect-run")
    inspect_parser.add_argument("--run-dir", required=True)
    inspect_parser.add_argument("--config")

    aggregate_parser = subparsers.add_parser("aggregate-run")
    aggregate_parser.add_argument("--run-dir", required=True)
    aggregate_parser.add_argument("--config")

    plan_suite_parser = subparsers.add_parser("plan-suite")
    plan_suite_parser.add_argument("--config", required=True)
    plan_suite_parser.add_argument("--stages")
    plan_suite_parser.add_argument("--tsv", action="store_true")

    plan_runs_parser = subparsers.add_parser("plan-runs")
    plan_runs_parser.add_argument("--config", required=True)
    plan_runs_parser.add_argument("--tsv", action="store_true")

    init_parser = subparsers.add_parser("init-suite")
    init_parser.add_argument("--suite-dir", required=True)
    init_parser.add_argument("--run-name", required=True)
    init_parser.add_argument("--config", required=True)
    init_parser.add_argument("--workers", required=True, type=int)
    init_parser.add_argument("--stages")

    record_parser = subparsers.add_parser("record-batch")
    record_parser.add_argument("--suite-dir", required=True)
    record_parser.add_argument("--batch-id", required=True)
    record_parser.add_argument("--status", choices=["ok", "failed"], required=True)
    record_parser.add_argument("--result-dir", default="")
    record_parser.add_argument("--error", default="")

    finalize_parser = subparsers.add_parser("finalize-suite")
    finalize_parser.add_argument("--suite-dir", required=True)
    finalize_parser.add_argument("--status", choices=["ok", "failed"], required=True)

    launch_parser = subparsers.add_parser("write-launch-manifest")
    launch_parser.add_argument("--result-dir", required=True)
    launch_parser.add_argument("--config", required=True)
    launch_parser.add_argument("--command", dest="exact_command", required=True)
    launch_parser.add_argument("--workers", required=True, type=int)
    launch_parser.add_argument("--start-time", required=True)
    launch_parser.add_argument("--end-time", required=True)
    launch_parser.add_argument("--status", choices=["ok", "failed"], required=True)

    aggregate_suite_parser = subparsers.add_parser("aggregate-suite")
    aggregate_suite_parser.add_argument("--suite-dir", required=True)

    package_parser = subparsers.add_parser("package-suite")
    package_parser.add_argument("--suite-dir", required=True)
    package_parser.add_argument("--output-dir")

    arguments = parser.parse_args()
    if arguments.command == "plan-config":
        result = config_plan(arguments.config, arguments.seed_start, arguments.seed_end)
        if arguments.lines:
            print(result["kind"])
            print(result["profile"])
            print(",".join(map(str, result["seeds"])))
            print(result["expected_runs"])
            print(result["configured_output_dir"])
        else:
            print(json.dumps(result, sort_keys=True))
        return
    if arguments.command == "inspect-run":
        config = read_json(arguments.config) if arguments.config else None
        result = inspect_run(arguments.run_dir, config)
    elif arguments.command == "aggregate-run":
        result = aggregate_single(arguments.run_dir, arguments.config)
    elif arguments.command == "plan-suite":
        result = load_suite_plan(arguments.config, _parse_stages(arguments.stages))
        if arguments.tsv:
            for record in result:
                print(
                    "\t".join(
                        str(record[key])
                        for key in (
                            "batch_id",
                            "stage",
                            "kind",
                            "config",
                            "run_name",
                            "extension_stage",
                        )
                    )
                )
            return
    elif arguments.command == "plan-runs":
        result = load_run_plan(arguments.config)
        if arguments.tsv:
            for record in result:
                print("\t".join(record[key] for key in ("id", "kind", "config")))
            return
    elif arguments.command == "init-suite":
        result = initialize_suite(
            arguments.suite_dir,
            arguments.run_name,
            arguments.config,
            arguments.workers,
            _parse_stages(arguments.stages),
        )
    elif arguments.command == "record-batch":
        result = record_batch(
            arguments.suite_dir,
            arguments.batch_id,
            arguments.status,
            arguments.result_dir,
            arguments.error,
        )
    elif arguments.command == "finalize-suite":
        result = finalize_suite(arguments.suite_dir, arguments.status)
    elif arguments.command == "write-launch-manifest":
        result = write_launch_manifest(
            arguments.result_dir,
            arguments.config,
            arguments.exact_command,
            arguments.workers,
            arguments.start_time,
            arguments.end_time,
            arguments.status,
        )
    elif arguments.command == "aggregate-suite":
        result = aggregate_suite(arguments.suite_dir)
    elif arguments.command == "package-suite":
        result = package_suite(arguments.suite_dir, arguments.output_dir)
    else:
        raise AssertionError(arguments.command)
    print(json.dumps(result, sort_keys=True))
    if isinstance(result, dict) and result.get("ok") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
