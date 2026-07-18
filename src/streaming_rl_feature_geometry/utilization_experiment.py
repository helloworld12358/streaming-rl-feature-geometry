"""Planning, reuse validation, and execution for utilization-adapter stages."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cross_experiment import load_cross_config, run_cross_one
from .experiment import config_hash, git_value, load_config, run_one, write_json
from .production_runtime import contained_path, repository_root, safe_worker_limit
from .utilization_adapters import normalize_adapter_config


STAGE_CONFIGS = {
    "stage-a": Path("configs/utilization_stage_a_full.json"),
    "stage-b1": Path("configs/utilization_stage_b1_full.json"),
    "stage-b2": Path("configs/utilization_stage_b2_full.json"),
    "smoke": Path("configs/utilization_adapter_smoke.json"),
}
FORMAL_STAGES = ("stage-a", "stage-b1", "stage-b2")
ESTIMATED_RUN_BYTES = 32 * 1024
BASELINE_GLOBAL_FIELDS = (
    "experiment_stage",
    "horizons",
    "trace_dim",
    "predictive_alpha",
    "control_alpha",
    "gamma",
    "lambda",
    "epsilon",
    "transform_eps",
    "transform_min_samples",
    "cov_update_every",
    "moment_beta",
    "moment_learning_rate",
    "covariance_shrinkage",
    "matrix_smoothing",
    "final_window",
    "controller_alpha_mode",
    "controller_alpha_norm",
)
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
BASELINE_ENV_FIELDS = (
    "interactions",
    "kwargs",
    "bank",
    "predictive_alpha",
    "control_alpha",
    "gamma",
    "lambda",
    "epsilon",
)


@dataclass(frozen=True)
class MatrixCell:
    stage: str
    engine: str
    environment: str
    condition: str
    adapter: str
    seed: int
    disposition: str

    @property
    def identity(self) -> str:
        return (
            f"{self.stage}/{self.environment}/{self.condition}/"
            f"{self.adapter}/seed_{self.seed:03d}"
        )


def _raw_config(stage: str) -> dict[str, Any]:
    return json.loads(STAGE_CONFIGS[stage].read_text(encoding="utf-8"))


def _adapter_configs(raw: dict[str, Any]) -> list[dict[str, Any]]:
    values = raw.get("utilization_adapters")
    if not isinstance(values, list) or not values:
        raise ValueError("utilization_adapters must be a non-empty list")
    canonical = [normalize_adapter_config(value) for value in values]
    names = [value["name"] for value in canonical]
    if len(names) != len(set(names)):
        raise ValueError("utilization_adapters names must be unique")
    return canonical


def stage_config(stage: str, adapter_name: str | None = None) -> dict[str, Any]:
    raw = _raw_config(stage)
    adapters = _adapter_configs(raw)
    selected = adapters[0] if adapter_name is None else next(
        (value for value in adapters if value["name"] == adapter_name), None
    )
    if selected is None:
        raise ValueError(f"{stage} does not register adapter {adapter_name!r}")
    config = (
        load_config(STAGE_CONFIGS[stage])
        if stage == "stage-b1"
        else load_cross_config(STAGE_CONFIGS[stage])
    )
    config.pop("utilization_adapters")
    config["utilization_adapter"] = selected
    allowed_envs = {"tmaze", "ringworld", "two_loop", "hidden_velocity"}
    if stage != "stage-b1" and set(config["environments"]) - allowed_envs:
        raise ValueError("utilization stages may use only the four registered environments")
    if config.get("controller_alpha_mode") != "norm_scaled":
        raise ValueError("formal utilization configs require norm_scaled controller alpha")
    if config.get("storage_schema") != "adapter_summary_v1":
        raise ValueError("formal utilization configs require adapter_summary_v1")
    return config


def build_matrix(stages: Iterable[str]) -> list[MatrixCell]:
    cells: list[MatrixCell] = []
    for stage in stages:
        raw = _raw_config(stage)
        adapters = [value["name"] for value in _adapter_configs(raw)]
        if stage == "stage-b1":
            pairs = [("tmaze", condition) for condition in raw["conditions"]]
            seeds = raw["seeds"]
            engine = "core"
        else:
            pairs = [
                (environment, condition)
                for environment, spec in raw["environments"].items()
                for condition in spec["conditions"]
            ]
            seeds = raw["seeds"]
            engine = "cross"
        for environment, condition in pairs:
            for adapter in adapters:
                for seed in seeds:
                    if stage == "smoke":
                        disposition = "new"
                    elif stage == "stage-a" and adapter == "identity":
                        disposition = "baseline_identity"
                    elif stage == "stage-a" and condition == "observation_only":
                        disposition = "structural_noop"
                    elif stage == "stage-b1" and condition == "observation_only" and adapter != "identity":
                        disposition = "structural_noop"
                    elif stage == "stage-b2" and adapter in {"identity", "residual_rff"}:
                        disposition = (
                            "stage_a_reuse" if adapter == "identity" or condition != "observation_only"
                            else "structural_noop"
                        )
                    elif stage == "stage-b2" and condition == "observation_only":
                        disposition = "structural_noop"
                    else:
                        disposition = "new"
                    cells.append(
                        MatrixCell(
                            stage, engine, environment, condition, adapter, int(seed), disposition
                        )
                    )
    identities = [cell.identity for cell in cells]
    if len(identities) != len(set(identities)):
        raise ValueError("logical utilization matrix contains duplicate cells")
    return cells


def _baseline_candidates(root: Path, cell: MatrixCell) -> list[Path]:
    suffix = Path("runs") / cell.environment / cell.condition / f"seed_{cell.seed:03d}"
    candidates = [root / suffix, root / "norm-scaled" / suffix]
    return [path for path in candidates if path.is_dir()]


def _field(config: dict[str, Any], environment: str, name: str) -> Any:
    if name in config.get("environments", {}).get(environment, {}):
        return config["environments"][environment][name]
    return config.get(name)


def validate_baseline_run(run_dir: Path, expected: dict[str, Any], cell: MatrixCell) -> dict[str, Any]:
    differences: list[str] = []
    required = {"config.json", "manifest.json", "runtime_validity.json", "summary.json"}
    missing = sorted(required - {path.name for path in run_dir.iterdir()})
    if missing:
        differences.append(f"missing files: {missing}")
        return {"status": "incompatible", "differences": differences, "source": str(run_dir)}
    try:
        observed = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        validity = json.loads((run_dir / "runtime_validity.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"status": "incompatible", "differences": [str(error)], "source": str(run_dir)}

    for name in BASELINE_GLOBAL_FIELDS:
        if observed.get(name) != expected.get(name):
            differences.append(f"{name}: {observed.get(name)!r} != {expected.get(name)!r}")
    for name in BASELINE_ENV_FIELDS:
        left = _field(observed, cell.environment, name)
        right = _field(expected, cell.environment, name)
        if left != right:
            differences.append(f"environment.{name}: {left!r} != {right!r}")
    if normalize_adapter_config(observed.get("utilization_adapter"))["name"] != "identity":
        differences.append("baseline utilization adapter is not identity-compatible")
    if manifest.get("environment") != cell.environment:
        differences.append("manifest environment mismatch")
    if manifest.get("condition") != cell.condition:
        differences.append("manifest condition mismatch")
    if int(manifest.get("seed", -1)) != cell.seed:
        differences.append("manifest seed mismatch")
    if manifest.get("config_hash") != config_hash(observed):
        differences.append("manifest config_hash mismatch")
    if manifest.get("git_commit") in {None, "", "unknown"}:
        differences.append("source commit identity is missing")
    try:
        raw_manifest_adapter = manifest["utilization_adapter"]
        if not isinstance(raw_manifest_adapter, dict):
            raise TypeError("manifest adapter must be explicit")
        manifest_adapter = normalize_adapter_config(raw_manifest_adapter)
    except (KeyError, TypeError, ValueError):
        manifest_adapter = {"name": "invalid"}
    if manifest_adapter["name"] != "identity":
        differences.append("source commit does not record identity adapter semantics")
    if manifest.get("controller_input_definition") != (
        "observation_plus_adapted_controller_state_plus_bias_v1"
    ):
        differences.append("source commit controller input identity is not verified")
    if manifest.get("exit_status") != "ok" or manifest.get("resume_eligible") is not True:
        differences.append("manifest run status is not strictly reusable")
    if validity.get("status") != "valid" or validity.get("resume_eligible") is not True:
        differences.append("runtime validity is not strictly reusable")
    if summary.get("run_status") not in {"ok", "valid"}:
        differences.append("summary run_status is not reusable")
    for name in ("final_performance", "nan_count", "inf_count", "divergence_flag"):
        try:
            if not np.isfinite(float(summary[name])):
                differences.append(f"summary {name} is non-finite")
        except (KeyError, TypeError, ValueError):
            differences.append(f"summary {name} is missing or invalid")
    return {
        "status": "compatible" if not differences else "incompatible",
        "differences": differences,
        "source": str(run_dir),
        "source_commit": manifest.get("git_commit"),
    }


def _storage_projection(new_runs: int, storage_report: Path | None) -> dict[str, Any]:
    per_run = ESTIMATED_RUN_BYTES
    source = "conservative_static_estimate"
    if storage_report is not None:
        report = json.loads(storage_report.read_text(encoding="utf-8"))
        if report.get("schema") != "adapter_storage_pilot_v1" or report.get("status") != "valid":
            raise ValueError("storage report is not a valid adapter_storage_pilot_v1 report")
        if int(report.get("pilot_runs", 0)) < 1:
            raise ValueError("storage report must contain at least one real pilot run")
        candidates = (
            report.get("adapter_summary_bytes_per_run"),
            report.get("bytes_per_run"),
            report.get("mean_run_bytes"),
        )
        observed = next((value for value in candidates if value is not None), None)
        if observed is None or int(observed) <= 0:
            raise ValueError("storage report lacks a positive per-run byte estimate")
        per_run = int(math.ceil(float(observed)))
        source = str(storage_report)
    expected = int(new_runs * per_run)
    return {
        "bytes_per_new_run": per_run,
        "expected_output_bytes": expected,
        "estimated_peak_disk_bytes": int(math.ceil(expected * 1.25)),
        "projection_source": source,
    }


def dry_run_report(
    stages: Iterable[str],
    *,
    baseline_root: Path | None = None,
    storage_report: Path | None = None,
    suite_root: Path | None = None,
) -> dict[str, Any]:
    selected = tuple(stages)
    cells = build_matrix(selected)
    reuse_records: list[dict[str, Any]] = []
    missing = incompatible = duplicates = reusable_identity = 0
    baseline_cells = [
        cell for cell in cells if cell.disposition == "baseline_identity"
    ]
    if "stage-b2" in selected and "stage-a" not in selected:
        baseline_cells.extend(
            MatrixCell(
                "stage-a", "cross", cell.environment, cell.condition, "identity", cell.seed,
                "baseline_identity",
            )
            for cell in cells
            if cell.stage == "stage-b2" and cell.adapter == "identity"
        )
    if baseline_cells:
        expected = stage_config("stage-a", "identity")
        for cell in baseline_cells:
            candidates = [] if baseline_root is None else _baseline_candidates(baseline_root, cell)
            if len(candidates) > 1:
                duplicates += 1
                reuse_records.append(
                    {"cell": cell.identity, "status": "duplicate", "sources": list(map(str, candidates))}
                )
            elif not candidates:
                missing += 1
                reuse_records.append({"cell": cell.identity, "status": "missing", "source": None})
            else:
                record = validate_baseline_run(candidates[0], expected, cell)
                record["cell"] = cell.identity
                reuse_records.append(record)
                if record["status"] == "compatible":
                    reusable_identity += 1
                else:
                    incompatible += 1

    missing_stage_a = 0
    if "stage-b2" in selected and "stage-a" not in selected:
        dependencies = [
            cell
            for cell in cells
            if cell.stage == "stage-b2"
            and cell.adapter == "residual_rff"
            and cell.condition != "observation_only"
        ]
        for cell in dependencies:
            source = MatrixCell(
                "stage-a", "cross", cell.environment, cell.condition, "residual_rff",
                cell.seed, "new",
            )
            path = None if suite_root is None else _new_run_dir(suite_root, source) / "summary.json"
            if path is None or not path.is_file():
                missing_stage_a += 1
                reuse_records.append(
                    {
                        "cell": cell.identity,
                        "status": "missing_stage_a",
                        "source": None if path is None else str(path),
                    }
                )

    structural = sum(cell.disposition == "structural_noop" for cell in cells)
    stage_reuse = sum(cell.disposition == "stage_a_reuse" for cell in cells)
    new_pending = sum(cell.disposition == "new" for cell in cells)
    report: dict[str, Any] = {
        "stages": list(selected),
        "logical_cells": len(cells),
        "reusable_identity_cells": reusable_identity,
        "structural_noop_reuse_cells": structural,
        "stage_a_reuse_cells": stage_reuse,
        "new_pending_runs": new_pending,
        "missing_baseline_cells": missing,
        "missing_stage_a_cells": missing_stage_a,
        "duplicate_cells": duplicates,
        "incompatible_cells": incompatible,
        "baseline_ready": missing == incompatible == duplicates == 0,
        "execution_ready": missing == incompatible == duplicates == missing_stage_a == 0,
        "stage_counts": {
            stage: {
                "logical_cells": sum(cell.stage == stage for cell in cells),
                "new_pending_runs": sum(cell.stage == stage and cell.disposition == "new" for cell in cells),
                "structural_noop_reuse_cells": sum(
                    cell.stage == stage and cell.disposition == "structural_noop" for cell in cells
                ),
                "baseline_identity_cells": sum(
                    cell.stage == stage and cell.disposition == "baseline_identity" for cell in cells
                ),
                "stage_a_reuse_cells": sum(
                    cell.stage == stage and cell.disposition == "stage_a_reuse" for cell in cells
                ),
            }
            for stage in selected
        },
        "reuse_validation": reuse_records,
    }
    report.update(_storage_projection(new_pending, storage_report))
    return report


def _new_run_dir(suite: Path, cell: MatrixCell) -> Path:
    adapter_root = suite / cell.stage / cell.adapter
    if cell.engine == "cross":
        return adapter_root / "runs" / cell.environment / cell.condition / f"seed_{cell.seed:03d}"
    return adapter_root / "runs" / cell.condition / f"seed_{cell.seed:03d}"


def _run_cell(task: tuple[MatrixCell, str, bool, bool]) -> dict[str, Any]:
    cell, suite_string, resume, retry_invalid = task
    suite = Path(suite_string)
    config = stage_config(cell.stage, cell.adapter)
    adapter_root = suite / cell.stage / cell.adapter
    run_dir = _new_run_dir(suite, cell)
    if cell.engine == "cross":
        options = {"resume": resume or retry_invalid, "retry_failed": retry_invalid}
        return run_cross_one(
            (config, cell.environment, cell.condition, cell.seed, str(adapter_root), options)
        )
    if run_dir.exists():
        manifest_path = run_dir / "manifest.json"
        valid = False
        if manifest_path.exists() and (run_dir / "summary.json").exists() and (
            run_dir / "runtime_validity.json"
        ).exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validity = json.loads((run_dir / "runtime_validity.json").read_text(encoding="utf-8"))
            valid = (
                manifest.get("exit_status") == "ok"
                and manifest.get("config_hash") == config_hash(config)
                and validity.get("status") == "valid"
            )
        if valid and (resume or retry_invalid):
            return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if not (retry_invalid if not valid else resume):
            raise FileExistsError(f"{run_dir} exists and is not reusable under requested mode")
        archive = suite / "failed_attempts" / cell.identity / datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(archive))
    adapter_root.mkdir(parents=True, exist_ok=True)
    return run_one((config, cell.condition, cell.seed, str(adapter_root)))


def _run_tasks(
    cells: list[MatrixCell], suite: Path, workers: int, resume: bool, retry_invalid: bool
) -> None:
    tasks = [(cell, str(suite), resume, retry_invalid) for cell in cells]
    count = min(max(1, int(workers)), len(tasks), cpu_count()) if tasks else 0
    if count == 1:
        list(map(_run_cell, tasks))
    elif count > 1:
        with Pool(count) as pool:
            pool.map(_run_cell, tasks)


def _summary_path_for_cell(
    suite: Path, cell: MatrixCell, baseline_root: Path | None
) -> tuple[Path, str]:
    if cell.disposition == "new":
        return _new_run_dir(suite, cell) / "summary.json", "new"
    if cell.stage == "stage-a":
        assert baseline_root is not None
        identity = MatrixCell(
            cell.stage, cell.engine, cell.environment, cell.condition, "identity", cell.seed,
            "baseline_identity",
        )
        source = _baseline_candidates(baseline_root, identity)
        if len(source) != 1:
            raise RuntimeError(f"baseline source became unavailable for {cell.identity}")
        return source[0] / "summary.json", (
            "baseline_identity" if cell.adapter == "identity" else "structural_noop"
        )
    if cell.stage == "stage-b1":
        identity = MatrixCell(
            cell.stage, cell.engine, cell.environment, cell.condition, "identity", cell.seed, "new"
        )
        return _new_run_dir(suite, identity) / "summary.json", "structural_noop"
    # Stage B2 identity/RFF values are the exact Stage A cells; tile observation
    # is also a state-empty alias of Stage A identity.
    source_adapter = "identity" if cell.adapter == "tile_coding" else cell.adapter
    stage_a = MatrixCell(
        "stage-a", "cross", cell.environment, cell.condition, source_adapter, cell.seed,
        "baseline_identity" if source_adapter == "identity" else "new",
    )
    return _summary_path_for_cell(suite, stage_a, baseline_root)[0], (
        "structural_noop" if cell.disposition == "structural_noop" else "stage_a_reuse"
    )


def aggregate_suite(
    suite: Path,
    stages: Iterable[str],
    baseline_root: Path | None,
    reuse_validation: list[dict[str, Any]],
) -> None:
    cells = build_matrix(stages)
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    for cell in cells:
        path, reuse_type = _summary_path_for_cell(suite, cell, baseline_root)
        if not path.is_file():
            failed_rows.append({"cell": cell.identity, "reason": "missing summary", "path": str(path)})
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("run_status") not in {"ok", "valid"}:
            failed_rows.append({"cell": cell.identity, "reason": "invalid run_status", "path": str(path)})
            continue
        row = dict(value)
        row.update(
            stage=cell.stage,
            environment=cell.environment,
            condition=cell.condition,
            adapter=cell.adapter,
            seed=cell.seed,
            reuse_type=reuse_type,
            source_summary=str(path),
        )
        rows.append(row)
        manifest_rows.append(
            {**asdict(cell), "identity": cell.identity, "reuse_type": reuse_type, "source": str(path)}
        )
    suite.mkdir(parents=True, exist_ok=True)
    write_json(suite / "run_manifest.json", manifest_rows)
    summaries = pd.DataFrame(rows)
    summaries.to_csv(suite / "run_summaries.csv", index=False)
    metric = "final_performance"
    aggregate = summaries.groupby(
        ["stage", "environment", "condition", "adapter"], as_index=False
    )[metric].agg(["mean", "sem"]).reset_index()
    aggregate.to_csv(suite / "aggregate_performance.csv", index=False)
    paired = summaries.pivot_table(
        index=["stage", "environment", "condition", "seed"],
        columns="adapter",
        values=metric,
        aggfunc="first",
    ).reset_index()
    for adapter in ("residual_rff", "tile_coding"):
        if adapter in paired and "identity" in paired:
            paired[f"{adapter}_minus_identity"] = paired[adapter] - paired["identity"]
    paired.to_csv(suite / "paired_adapter_differences.csv", index=False)
    runtime = summaries.groupby(["stage", "adapter"], as_index=False)[
        ["nan_count", "inf_count", "divergence_flag", "wall_seconds"]
    ].sum()
    runtime.to_csv(suite / "runtime_summary.csv", index=False)
    pd.DataFrame(reuse_validation).to_csv(suite / "reuse_validation.csv", index=False)
    pd.DataFrame(failed_rows, columns=["cell", "reason", "path"]).to_csv(
        suite / "failed_runs.csv", index=False
    )
    (suite / "experiment_summary.md").write_text(
        "# Utilization adapter experiment summary\n\n"
        f"- Logical cells: {len(cells)}\n"
        f"- Valid summaries: {len(rows)}\n"
        f"- Failed or missing: {len(failed_rows)}\n",
        encoding="utf-8",
    )
    if failed_rows:
        raise RuntimeError(f"utilization aggregation found {len(failed_rows)} failed/missing cells")
    if set(stages) == set(FORMAL_STAGES):
        make_formal_figures(summaries, suite / "figures")


def make_formal_figures(summaries: pd.DataFrame, output_dir: Path) -> None:
    """Generate only the four preregistered aggregate figures."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    stage_a = summaries[summaries["stage"] == "stage-a"]
    grouped = stage_a.groupby(["environment", "adapter"])["final_performance"].mean().unstack()
    ax = grouped[["identity", "residual_rff"]].plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("Final performance")
    ax.set_title("Stage A: identity vs residual RFF")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "01_stage_a_identity_vs_rff.png", dpi=160)
    plt.close(ax.figure)

    paired = stage_a.pivot_table(
        index=["environment", "condition", "seed"],
        columns="adapter",
        values="final_performance",
        aggfunc="first",
    )
    differences = (paired["residual_rff"] - paired["identity"]).groupby("environment").mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    differences.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Paired RFF - identity")
    ax.set_title("Stage A paired adapter difference")
    fig.tight_layout()
    fig.savefig(output_dir / "02_stage_a_paired_rff_minus_identity.png", dpi=160)
    plt.close(fig)

    core = summaries[
        (summaries["stage"] == "stage-b1") & (summaries["condition"] == "trace_only")
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    core.groupby("adapter")["final_performance"].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("Final-window accuracy")
    ax.set_title("Core T-maze trace-only utilization")
    fig.tight_layout()
    fig.savefig(output_dir / "03_core_tmaze_trace_only.png", dpi=160)
    plt.close(fig)

    hidden = summaries[
        (summaries["stage"] == "stage-b2")
        & summaries["condition"].isin(["raw", "whitened", "matched"])
    ]
    grouped = hidden.groupby(["condition", "adapter"])["final_performance"].mean().unstack()
    ax = grouped.plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("Final reward")
    ax.set_title("Hidden velocity local nonlinear utilization")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "04_hidden_velocity_adapters.png", dpi=160)
    plt.close(ax.figure)


def run_experiment(
    stages: Iterable[str],
    *,
    baseline_root: Path | None,
    workers: int,
    run_name: str,
    output_root: Path,
    storage_report: Path | None,
    allow_full_run: bool,
    resume: bool,
    retry_invalid: bool,
) -> Path:
    selected = tuple(stages)
    formal = any(stage in FORMAL_STAGES for stage in selected)
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("run_name may contain only letters, digits, dot, underscore, and hyphen")
    report = dry_run_report(
        selected,
        baseline_root=baseline_root,
        storage_report=storage_report,
        suite_root=output_root / run_name,
    )
    if formal and (not allow_full_run or os.environ.get("RL_RUN_CONTEXT") != "remote"):
        raise RuntimeError("formal utilization run requires RL_RUN_CONTEXT=remote and --allow-full-run")
    if formal and storage_report is None:
        raise RuntimeError("formal utilization run requires --storage-report from a real pilot")
    if formal and not report["baseline_ready"] and "stage-a" in selected:
        raise RuntimeError("Stage A baseline is missing, duplicated, or incompatible; refusing new runs")
    if formal and not report["execution_ready"]:
        raise RuntimeError("formal utilization dependencies are missing or incompatible; refusing new runs")
    if formal:
        safe_max, effective_cpus, quota = safe_worker_limit()
        if workers < 1 or workers > safe_max:
            raise RuntimeError(
                f"workers={workers} exceeds safe maximum={safe_max}; "
                f"effective_cpus={effective_cpus}, cgroup_quota={quota}"
            )
        output_root = contained_path(
            output_root, repository_root(), label="utilization_output_root"
        )
    suite = output_root / run_name
    if suite.exists() and not (resume or retry_invalid):
        raise FileExistsError(f"{suite} exists; use --resume or --retry-invalid")
    suite.mkdir(parents=True, exist_ok=resume or retry_invalid)
    write_json(suite / "dry_run_report.json", report)
    write_json(
        suite / "suite_manifest.json",
        {
            "run_name": run_name,
            "stages": list(selected),
            "git_commit": git_value("rev-parse", "HEAD"),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "storage_report": None if storage_report is None else str(storage_report),
        },
    )
    matrix = build_matrix(selected)
    # Stages are deliberately sequential; process-level parallelism is confined
    # within one stage so Stage B reuse can never race Stage A production.
    for stage in selected:
        pending = [
            cell for cell in matrix if cell.stage == stage and cell.disposition == "new"
        ]
        _run_tasks(pending, suite, workers, resume, retry_invalid)
    aggregate_suite(suite, selected, baseline_root, report["reuse_validation"])
    return suite


def _expand_stages(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("--stage all cannot be combined with another stage")
        return FORMAL_STAGES
    if "smoke" in values and len(values) != 1:
        raise ValueError("smoke must run separately from formal stages")
    ordered = tuple(dict.fromkeys(values))
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", action="append", required=True,
        choices=["stage-a", "stage-b1", "stage-b2", "all", "smoke"],
    )
    parser.add_argument("--baseline-root")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="results/utilization_adapters")
    parser.add_argument("--storage-report")
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        stages = _expand_stages(args.stage)
        baseline = None if args.baseline_root is None else Path(args.baseline_root)
        storage = None if args.storage_report is None else Path(args.storage_report)
        report = dry_run_report(
            stages,
            baseline_root=baseline,
            storage_report=storage,
            suite_root=Path(args.output_root) / args.run_name,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run:
        printable = dict(report)
        records = printable.pop("reuse_validation")
        printable["reuse_validation_records"] = len(records)
        printable["incompatible_details"] = [
            record for record in records if record.get("status") in {"incompatible", "duplicate"}
        ]
        print(json.dumps(printable, indent=2, sort_keys=True))
        if not report["execution_ready"]:
            raise SystemExit(2)
        return
    result = run_experiment(
        stages,
        baseline_root=baseline,
        workers=args.workers,
        run_name=args.run_name,
        output_root=Path(args.output_root),
        storage_report=storage,
        allow_full_run=args.allow_full_run,
        resume=args.resume or args.retry_invalid,
        retry_invalid=args.retry_invalid,
    )
    print(f"RESULT_DIR={result.resolve()}")


if __name__ == "__main__":
    main()
