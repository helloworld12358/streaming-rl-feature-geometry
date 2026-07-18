"""Planning, reuse validation, and execution for utilization-adapter stages."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from functools import lru_cache
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
    "storage-pilot": Path("configs/utilization_storage_pilot.json"),
}
FORMAL_STAGES = ("stage-a", "stage-b1", "stage-b2")
ESTIMATED_RUN_BYTES = 32 * 1024
CONTROLLER_INPUT_DEFINITION = (
    "adapter_of_complete_controller_input_"
    "observation_then_condition_state_then_bias_v2"
)
KNOWN_IDENTITY_COMMITS = {
    "2940182f41c9c855a26b4fd740e41907856f2f7f",
}
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@lru_cache(maxsize=1)
def _current_commit() -> str:
    return git_value("rev-parse", "HEAD")


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
    if config.get("controller_alpha_mode") != "fixed":
        raise ValueError("utilization configs must preserve their fixed-alpha source")
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
                        disposition = "baseline_candidate"
                    elif stage == "stage-b2" and adapter in {"identity", "residual_rff"}:
                        disposition = "stage_a_candidate"
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


def _stage_a_source_path(root: Path, cell: MatrixCell) -> Path:
    """Resolve a Stage A logical cell, including a validated external baseline source."""

    summaries_path = root / "run_summaries.csv"
    if summaries_path.is_file():
        frame = pd.read_csv(summaries_path)
        required = {"stage", "environment", "condition", "adapter", "seed", "source_summary"}
        if required <= set(frame.columns):
            matches = frame[
                (frame["stage"] == "stage-a")
                & (frame["environment"] == cell.environment)
                & (frame["condition"] == cell.condition)
                & (frame["adapter"] == cell.adapter)
                & (frame["seed"].astype(int) == cell.seed)
            ]
            if len(matches) > 1:
                raise ValueError(f"duplicate Stage A logical source for {cell.identity}")
            if len(matches) == 1:
                return Path(str(matches.iloc[0]["source_summary"])).parent
    return _new_run_dir(root, cell)


def cell_signature(
    config: dict[str, Any],
    cell: MatrixCell,
    *,
    controller_input_definition: str = CONTROLLER_INPUT_DEFINITION,
) -> dict[str, Any]:
    """Return the normalized per-cell scientific and schema identity."""

    adapter = normalize_adapter_config(config.get("utilization_adapter"))
    alpha_mode = str(config.get("controller_alpha_mode", "fixed"))
    if cell.engine == "core":
        interactions = int(config["total_interactions"])
        final_window = int(config["final_window_trials"])
        control_alpha = float(config["control_alpha"])
        predictive_alpha = float(config["gvf_alpha"])
        gamma = float(config["gamma"])
        lam = float(config["lambda"])
        epsilon = float(config["epsilon"])
        bank = config.get("gvf_bank", "mixed")
        environment_kwargs = {"corridor_length": int(config["corridor_length"])}
        evaluation = "decision_time_final_trial_accuracy"
    else:
        env_spec = config["environments"][cell.environment]
        interactions = int(env_spec["interactions"])
        final_window = int(config["final_window"])
        control_alpha = float(env_spec.get("control_alpha", config["control_alpha"]))
        predictive_alpha = float(
            env_spec.get("predictive_alpha", config["predictive_alpha"])
        )
        gamma = float(env_spec.get("gamma", config["gamma"]))
        lam = float(env_spec.get("lambda", config["lambda"]))
        epsilon = float(env_spec.get("epsilon", config["epsilon"]))
        bank = env_spec.get("bank", "mixed")
        environment_kwargs = env_spec.get("kwargs", {})
        evaluation = (
            "final_window_reward"
            if cell.environment == "hidden_velocity"
            else "decision_time_final_window_accuracy"
        )
    return {
        "environment": cell.environment,
        "condition": cell.condition,
        "seed": int(cell.seed),
        "interactions": interactions,
        "final_window": final_window,
        "gamma": gamma,
        "lambda": lam,
        "epsilon": epsilon,
        "control_alpha": control_alpha,
        "controller_alpha_mode": alpha_mode,
        "controller_alpha_norm": (
            config.get("controller_alpha_norm", {}) if alpha_mode == "norm_scaled" else None
        ),
        "predictive_alpha": predictive_alpha,
        "bank": bank,
        "horizons": list(config.get("horizons", [])),
        "trace_dim": int(config.get("trace_dim", 0)),
        "transform": {
            name: config.get(name)
            for name in (
                "transform_eps",
                "transform_min_samples",
                "cov_update_every",
                "moment_beta",
                "moment_learning_rate",
                "covariance_shrinkage",
                "matrix_smoothing",
            )
        },
        "controller_input_definition": controller_input_definition,
        "environment_kwargs": environment_kwargs,
        "evaluation_definition": evaluation,
        "identity_semantics": adapter["name"] == "identity",
        "adapter": adapter,
        "result_schema": config.get("storage_schema", "legacy_csv"),
        "result_schema_version": int(config.get("result_schema_version", 0)),
    }


def validate_baseline_run(run_dir: Path, expected: dict[str, Any], cell: MatrixCell) -> dict[str, Any]:
    differences: list[str] = []
    required = {"config.json", "manifest.json", "runtime_validity.json", "summary.json"}
    missing = sorted(required - {path.name for path in run_dir.iterdir()})
    if missing:
        differences.append(f"missing files: {missing}")
        return {"status": "incompatible", "differences": differences, "source": str(run_dir)}
    try:
        raw_observed = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        validity = json.loads((run_dir / "runtime_validity.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"status": "incompatible", "differences": [str(error)], "source": str(run_dir)}

    observed_hash = config_hash(raw_observed)
    try:
        observed = load_cross_config(run_dir / "config.json")
    except (OSError, TypeError, ValueError) as error:
        return {
            "status": "incompatible",
            "differences": [f"effective config: {error}"],
            "source": str(run_dir),
        }
    try:
        observed_adapter = normalize_adapter_config(observed.get("utilization_adapter"))
    except (TypeError, ValueError) as error:
        differences.append(f"utilization_adapter: invalid ({error})")
        observed_adapter = {"name": "invalid"}
    observed["utilization_adapter"] = observed_adapter
    if manifest.get("environment") != cell.environment:
        differences.append("manifest environment mismatch")
    if manifest.get("condition") != cell.condition:
        differences.append("manifest condition mismatch")
    if int(manifest.get("seed", -1)) != cell.seed:
        differences.append("manifest seed mismatch")
    if manifest.get("config_hash") != observed_hash:
        differences.append(
            f"config_hash: {manifest.get('config_hash')!r} != {observed_hash!r}"
        )
    source_commit = manifest.get("source_commit", manifest.get("git_commit"))
    current_commit = _current_commit()
    if source_commit not in KNOWN_IDENTITY_COMMITS | {current_commit}:
        differences.append(
            f"source_commit: {source_commit!r} is not a verified identity implementation"
        )
    input_definition = manifest.get("controller_input_definition")
    if input_definition != CONTROLLER_INPUT_DEFINITION:
        if source_commit in KNOWN_IDENTITY_COMMITS and observed_adapter["name"] == "identity":
            input_definition = CONTROLLER_INPUT_DEFINITION
        else:
            input_definition = str(input_definition)
    observed_signature = cell_signature(
        observed, cell, controller_input_definition=input_definition
    )
    expected_signature = cell_signature(expected, cell)
    for name, right in expected_signature.items():
        left = observed_signature.get(name)
        if left != right:
            differences.append(f"{name}: {left!r} != {right!r}")
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
    if int(summary.get("nan_count", 1)) != 0:
        differences.append(f"summary nan_count: {summary.get('nan_count')!r} != 0")
    if int(summary.get("inf_count", 1)) != 0:
        differences.append(f"summary inf_count: {summary.get('inf_count')!r} != 0")
    if int(summary.get("divergence_flag", 1)) != 0:
        differences.append(
            f"summary divergence_flag: {summary.get('divergence_flag')!r} != 0"
        )
    return {
        "status": "compatible" if not differences else "incompatible",
        "differences": differences,
        "source": str(run_dir),
        "source_commit": manifest.get("git_commit"),
    }


def _storage_projection(
    new_runs: int,
    storage_report: Path | None,
    pending_by_adapter: dict[str, int] | None = None,
) -> dict[str, Any]:
    per_run = ESTIMATED_RUN_BYTES
    adapter_bytes = {
        "identity": per_run,
        "residual_rff": per_run,
        "tile_coding": per_run,
    }
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
        observed_adapters = report.get("bytes_per_run_by_adapter", {})
        for adapter in adapter_bytes:
            value = observed_adapters.get(adapter, per_run)
            if int(value) <= 0:
                raise ValueError(f"storage report has invalid {adapter} byte estimate")
            adapter_bytes[adapter] = int(math.ceil(float(value)))
        source = str(storage_report)
    counts = pending_by_adapter or {
        "identity": 0,
        "residual_rff": 0,
        "tile_coding": 0,
    }
    expected = (
        int(sum(counts.get(adapter, 0) * size for adapter, size in adapter_bytes.items()))
        if pending_by_adapter is not None
        else int(new_runs * per_run)
    )
    aggregate_bytes = 5 * 1024 * 1024
    package_bytes = expected + aggregate_bytes
    peak = expected + aggregate_bytes + package_bytes
    free = shutil.disk_usage(repository_root()).free
    safety = 10 * 1024**3
    return {
        "bytes_per_new_run": per_run,
        "bytes_per_run_by_adapter": adapter_bytes,
        "expected_output_bytes": expected,
        "aggregate_and_figure_bytes": aggregate_bytes,
        "estimated_package_bytes": package_bytes,
        "estimated_peak_disk_bytes": peak,
        "peak_plus_10_gib_bytes": peak + safety,
        "current_free_space_bytes": free,
        "launch_allowed": free >= peak + safety,
        "projection_source": source,
    }


def validate_suite_run(
    run_dir: Path, expected_config: dict[str, Any], cell: MatrixCell
) -> dict[str, Any]:
    """Validate an adapter result for resume or cross-stage reuse."""

    differences: list[str] = []
    required = {"config.json", "manifest.json", "runtime_validity.json", "summary.json"}
    if not run_dir.is_dir():
        return {"status": "missing", "source": str(run_dir), "differences": []}
    missing = sorted(required - {path.name for path in run_dir.iterdir()})
    if missing:
        return {
            "status": "incompatible",
            "source": str(run_dir),
            "differences": [f"missing files: {missing}"],
        }
    try:
        observed = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        validity = json.loads((run_dir / "runtime_validity.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        observed = (
            load_config(run_dir / "config.json")
            if cell.engine == "core"
            else load_cross_config(run_dir / "config.json")
        )
    except (OSError, TypeError, ValueError) as error:
        return {
            "status": "incompatible",
            "source": str(run_dir),
            "differences": [str(error)],
        }
    observed_signature = cell_signature(observed, cell)
    expected_signature = cell_signature(expected_config, cell)
    for name, right in expected_signature.items():
        left = observed_signature.get(name)
        if left != right:
            differences.append(f"{name}: {left!r} != {right!r}")
    raw_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if manifest.get("config_hash") != config_hash(raw_config):
        differences.append("manifest config_hash does not match source config")
    if manifest.get("git_commit") != _current_commit():
        differences.append(
            f"git_commit: {manifest.get('git_commit')!r} != {_current_commit()!r}"
        )
    if manifest.get("exit_status") != "ok" or manifest.get("resume_eligible") is not True:
        differences.append("manifest status is not reusable")
    if validity.get("status") != "valid" or validity.get("resume_eligible") is not True:
        differences.append("runtime validity is not reusable")
    if summary.get("run_status") not in {"ok", "valid"}:
        differences.append("summary run_status is not reusable")
    for name in ("final_performance", "nan_count", "inf_count", "divergence_flag"):
        try:
            if not np.isfinite(float(summary[name])):
                differences.append(f"summary {name} is non-finite")
        except (KeyError, TypeError, ValueError):
            differences.append(f"summary {name} is missing or invalid")
    if any(int(summary.get(name, 1)) != 0 for name in ("nan_count", "inf_count", "divergence_flag")):
        differences.append("summary numerical validity counters are nonzero")
    return {
        "status": "compatible" if not differences else "incompatible",
        "source": str(run_dir),
        "differences": differences,
    }


def dry_run_report(
    stages: Iterable[str],
    *,
    baseline_root: Path | None = None,
    storage_report: Path | None = None,
    suite_root: Path | None = None,
    stage_a_root: Path | None = None,
) -> dict[str, Any]:
    selected = tuple(stages)
    cells = build_matrix(selected)
    plans: list[dict[str, Any]] = []
    reuse_records: list[dict[str, Any]] = []
    missing_baseline = incompatible_baseline = duplicate_baseline = 0
    missing_stage_a = incompatible_stage_a = 0

    for cell in cells:
        direct_path = None if suite_root is None else _new_run_dir(suite_root, cell)
        direct = (
            None
            if direct_path is None
            else validate_suite_run(direct_path, stage_config(cell.stage, cell.adapter), cell)
        )
        if direct is not None and direct["status"] == "compatible":
            plan = {
                "cell": cell.identity,
                "status": "reused_existing",
                "source": str(direct_path / "summary.json"),
                "physical_cell": cell.identity,
            }
            plans.append(plan)
            reuse_records.append(plan)
            continue

        if cell.disposition == "baseline_candidate":
            candidates = [] if baseline_root is None else _baseline_candidates(baseline_root, cell)
            if len(candidates) > 1:
                duplicate_baseline += 1
                record = {
                    "cell": cell.identity,
                    "status": "duplicate",
                    "sources": list(map(str, candidates)),
                }
                reuse_records.append(record)
            elif not candidates:
                missing_baseline += 1
                reuse_records.append(
                    {"cell": cell.identity, "status": "missing", "source": None}
                )
            else:
                record = validate_baseline_run(
                    candidates[0], stage_config("stage-a", "identity"), cell
                )
                record["cell"] = cell.identity
                reuse_records.append(record)
                if record["status"] == "compatible":
                    plans.append(
                        {
                            "cell": cell.identity,
                            "status": "reused_baseline",
                            "source": str(candidates[0] / "summary.json"),
                            "physical_cell": cell.identity,
                        }
                    )
                    continue
                incompatible_baseline += 1

        if cell.disposition == "stage_a_candidate":
            source_cell = MatrixCell(
                "stage-a",
                "cross",
                cell.environment,
                cell.condition,
                cell.adapter,
                cell.seed,
                "new",
            )
            left = cell_signature(stage_config("stage-a", cell.adapter), source_cell)
            right = cell_signature(stage_config("stage-b2", cell.adapter), cell)
            signatures_match = left == right
            if "stage-a" in selected and signatures_match:
                source_plan = next(
                    (
                        record
                        for record in plans
                        if record["cell"] == source_cell.identity
                    ),
                    None,
                )
                source_summary = (
                    source_plan.get("source") if source_plan is not None else None
                )
                if source_summary is None and suite_root is not None:
                    source_summary = str(
                        _new_run_dir(suite_root, source_cell) / "summary.json"
                    )
                plans.append(
                    {
                        "cell": cell.identity,
                        "status": "planned_stage_a_reuse",
                        "source": source_summary or source_cell.identity,
                        "physical_cell": source_cell.identity,
                    }
                )
                continue
            if stage_a_root is not None and signatures_match:
                stage_a_path = _stage_a_source_path(stage_a_root, source_cell)
                record = validate_suite_run(
                    stage_a_path, stage_config("stage-a", cell.adapter), source_cell
                )
                record["cell"] = cell.identity
                reuse_records.append(record)
                if record["status"] == "compatible":
                    plans.append(
                        {
                            "cell": cell.identity,
                            "status": "reused_stage_a",
                            "source": str(stage_a_path / "summary.json"),
                            "physical_cell": source_cell.identity,
                        }
                    )
                    continue
                if record["status"] == "missing":
                    missing_stage_a += 1
                else:
                    incompatible_stage_a += 1
            elif "stage-a" not in selected:
                missing_stage_a += 1
                reuse_records.append(
                    {"cell": cell.identity, "status": "missing_stage_a", "source": None}
                )

        plans.append(
            {
                "cell": cell.identity,
                "status": "pending",
                "source": None if direct_path is None else str(direct_path / "summary.json"),
                "physical_cell": cell.identity,
            }
        )

    pending = [record for record in plans if record["status"] == "pending"]
    reused = [record for record in plans if record["status"].startswith("reused")]
    planned_reuse = [
        record for record in plans if record["status"] == "planned_stage_a_reuse"
    ]
    pending_by_adapter = {
        adapter: sum(f"/{adapter}/" in record["cell"] for record in pending)
        for adapter in ("identity", "residual_rff", "tile_coding")
    }
    stage_counts: dict[str, dict[str, Any]] = {}
    for stage in selected:
        stage_plans = [record for record in plans if record["cell"].startswith(f"{stage}/")]
        stage_counts[stage] = {
            "logical_cells": len(stage_plans),
            "reused_cells": sum(record["status"].startswith("reused") for record in stage_plans),
            "planned_cross_stage_reuse_cells": sum(
                record["status"] == "planned_stage_a_reuse" for record in stage_plans
            ),
            "pending_runs": sum(record["status"] == "pending" for record in stage_plans),
            "new_identity_runs": sum(
                record["status"] == "pending" and "/identity/" in record["cell"]
                for record in stage_plans
            ),
            "new_rff_runs": sum(
                record["status"] == "pending" and "/residual_rff/" in record["cell"]
                for record in stage_plans
            ),
            "new_tile_runs": sum(
                record["status"] == "pending" and "/tile_coding/" in record["cell"]
                for record in stage_plans
            ),
        }
    report: dict[str, Any] = {
        "stages": list(selected),
        "logical_cells": len(cells),
        "reused_cells": len(reused),
        "reusable_identity_cells": sum(
            record["status"] == "reused_baseline" for record in plans
        ),
        "stage_a_reuse_cells": len(planned_reuse)
        + sum(record["status"] == "reused_stage_a" for record in plans),
        "cross_stage_deduplicated_cells": len(planned_reuse),
        "new_pending_runs": len(pending),
        "pending_by_adapter": pending_by_adapter,
        "missing_baseline_cells": missing_baseline,
        "missing_stage_a_cells": missing_stage_a,
        "duplicate_cells": duplicate_baseline,
        "incompatible_cells": incompatible_baseline + incompatible_stage_a,
        "incompatible_baseline_cells": incompatible_baseline,
        "incompatible_stage_a_cells": incompatible_stage_a,
        "baseline_ready": missing_baseline == incompatible_baseline == duplicate_baseline == 0,
        "execution_ready": True,
        "stage_counts": stage_counts,
        "config_paths": [str(STAGE_CONFIGS[stage]) for stage in selected],
        "launcher_command": (
            "python scripts/run_utilization_experiment.py "
            + " ".join(f"--stage {stage}" for stage in selected)
        ),
        "execution_plan": plans,
        "reuse_validation": reuse_records,
    }
    report.update(_storage_projection(len(pending), storage_report, pending_by_adapter))
    per_adapter = report["bytes_per_run_by_adapter"]
    report["estimated_stage_bytes"] = {
        stage: int(
            values["new_identity_runs"] * per_adapter["identity"]
            + values["new_rff_runs"] * per_adapter["residual_rff"]
            + values["new_tile_runs"] * per_adapter["tile_coding"]
        )
        for stage, values in stage_counts.items()
    }
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
                and manifest.get("git_commit") == _current_commit()
                and int(manifest.get("result_schema_version", -1))
                == int(config.get("result_schema_version", 1))
                and validity.get("status") == "valid"
                and validity.get("resume_eligible") is True
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
    suite: Path, cell: MatrixCell, report: dict[str, Any]
) -> tuple[Path, str]:
    matches = [
        record for record in report["execution_plan"] if record["cell"] == cell.identity
    ]
    if len(matches) != 1:
        raise RuntimeError(f"execution plan has {len(matches)} entries for {cell.identity}")
    record = matches[0]
    source = record.get("source")
    if source is None:
        source = str(_new_run_dir(suite, cell) / "summary.json")
    return Path(source), str(record["status"])


def aggregate_suite(
    suite: Path,
    stages: Iterable[str],
    report: dict[str, Any],
) -> None:
    cells = build_matrix(stages)
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    for cell in cells:
        path, reuse_type = _summary_path_for_cell(suite, cell, report)
        if not path.is_file():
            failed_rows.append({"cell": cell.identity, "reason": "missing summary", "path": str(path)})
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = path.parent / "manifest.json"
        config_path = path.parent / "config.json"
        validity_path = path.parent / "runtime_validity.json"
        if not manifest_path.is_file() or not config_path.is_file() or not validity_path.is_file():
            failed_rows.append(
                {"cell": cell.identity, "reason": "missing evidence file", "path": str(path)}
            )
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validity = json.loads(validity_path.read_text(encoding="utf-8"))
        if value.get("run_status") not in {"ok", "valid"}:
            failed_rows.append({"cell": cell.identity, "reason": "invalid run_status", "path": str(path)})
            continue
        if validity.get("status") != "valid" or validity.get("resume_eligible") is not True:
            failed_rows.append(
                {"cell": cell.identity, "reason": "runtime invalid", "path": str(path)}
            )
            continue
        if reuse_type != "reused_baseline":
            validation = validate_suite_run(
                path.parent, stage_config(cell.stage, cell.adapter), cell
            )
            if validation["status"] != "compatible":
                failed_rows.append(
                    {
                        "cell": cell.identity,
                        "reason": "; ".join(validation["differences"]),
                        "path": str(path),
                    }
                )
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
            source_commit=manifest.get("git_commit"),
            source_config_hash=manifest.get("config_hash"),
        )
        rows.append(row)
        manifest_rows.append(
            {
                **asdict(cell),
                "identity": cell.identity,
                "reuse_type": reuse_type,
                "source": str(path),
                "source_commit": manifest.get("git_commit"),
                "config_hash": manifest.get("config_hash"),
            }
        )
    suite.mkdir(parents=True, exist_ok=True)
    source_commits = {
        row["source_commit"] for row in manifest_rows if row.get("source_commit")
    }
    if len(source_commits) > 1:
        failed_rows.append(
            {
                "cell": "GLOBAL",
                "reason": f"mixed source commits: {sorted(source_commits)}",
                "path": str(suite),
            }
        )
    write_json(suite / "run_manifest.json", manifest_rows)
    summaries = pd.DataFrame(rows)
    summaries.to_csv(suite / "run_summaries.csv", index=False)
    metric = "final_performance"
    aggregate = summaries.groupby(
        ["stage", "environment", "condition", "adapter"], as_index=False
    )[metric].agg(mean="mean", sem="sem", valid_seed_count="count")
    aggregate.to_csv(suite / "aggregate_performance.csv", index=False)
    paired_rows: list[dict[str, Any]] = []
    for keys, group in summaries.groupby(["stage", "environment", "condition"]):
        stage, environment, condition = keys
        identity = group[group["adapter"] == "identity"].set_index("seed")[metric]
        for adapter in ("residual_rff", "tile_coding"):
            right = group[group["adapter"] == adapter].set_index("seed")[metric]
            common = identity.index.intersection(right.index)
            differences = right.loc[common] - identity.loc[common]
            paired_mean = float(differences.mean()) if len(differences) else np.nan
            paired_sem = float(differences.sem()) if len(differences) > 1 else np.nan
            for seed in common:
                paired_rows.append(
                    {
                        "stage": stage,
                        "environment": environment,
                        "condition": condition,
                        "left_adapter": "identity",
                        "right_adapter": adapter,
                        "seed": int(seed),
                        "left_performance": float(identity.loc[seed]),
                        "right_performance": float(right.loc[seed]),
                        "paired_difference": float(right.loc[seed] - identity.loc[seed]),
                        "paired_mean": paired_mean,
                        "paired_sem": paired_sem,
                        "valid_seed_count": int(len(common)),
                    }
                )
    pd.DataFrame(
        paired_rows,
        columns=[
            "stage", "environment", "condition", "left_adapter", "right_adapter",
            "seed", "left_performance", "right_performance", "paired_difference",
            "paired_mean", "paired_sem", "valid_seed_count",
        ],
    ).to_csv(suite / "paired_adapter_differences.csv", index=False)
    runtime = summaries.groupby(["stage", "adapter"], as_index=False)[
        ["nan_count", "inf_count", "divergence_flag", "wall_seconds"]
    ].sum()
    runtime.to_csv(suite / "runtime_summary.csv", index=False)
    pd.DataFrame(report["reuse_validation"]).to_csv(
        suite / "reuse_validation.csv", index=False
    )
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
    environments = ("tmaze", "ringworld", "two_loop", "hidden_velocity")
    adapters = ("identity", "residual_rff")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, environment in zip(axes.flat, environments, strict=True):
        frame = stage_a[stage_a["environment"] == environment]
        conditions = list(dict.fromkeys(frame["condition"].tolist()))
        x = np.arange(len(conditions), dtype=float)
        for offset, adapter in zip((-0.2, 0.2), adapters, strict=True):
            means = []
            sems = []
            for condition in conditions:
                values = frame[
                    (frame["condition"] == condition) & (frame["adapter"] == adapter)
                ]["final_performance"]
                means.append(float(values.mean()))
                sems.append(float(values.sem()) if len(values) > 1 else 0.0)
            ax.bar(x + offset, means, width=0.4, yerr=sems, label=adapter, capsize=2)
        ax.set_xticks(x, conditions, rotation=35, ha="right")
        ax.set_title(environment)
        ax.set_ylabel("Final-window reward" if environment == "hidden_velocity" else "Final-window accuracy")
    axes.flat[0].legend()
    fig.suptitle("Stage A: residual RFF versus identity (mean ± SEM)")
    fig.savefig(output_dir / "01_stage_a_identity_vs_rff.png", dpi=160)
    plt.close(fig)

    paired = stage_a.pivot(
        index=["environment", "condition", "seed"],
        columns="adapter",
        values="final_performance",
    )
    paired["difference"] = paired["residual_rff"] - paired["identity"]
    stats = paired.groupby(["environment", "condition"])["difference"].agg(
        mean="mean", sem="sem"
    ).reset_index()
    labels = [f"{row.environment}\n{row.condition}" for row in stats.itertuples()]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.45), 5))
    ax.bar(
        np.arange(len(stats)), stats["mean"],
        yerr=stats["sem"].fillna(0.0), capsize=2,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(stats)), labels, rotation=55, ha="right")
    ax.set_ylabel("Paired residual_rff - identity")
    ax.set_title("Stage A paired rescue; positive values favor residual RFF")
    fig.tight_layout()
    fig.savefig(output_dir / "02_stage_a_paired_rff_minus_identity.png", dpi=160)
    plt.close(fig)

    core = summaries[summaries["stage"] == "stage-b1"]
    requested = [
        ("trace_only", "identity"),
        ("trace_only", "residual_rff"),
        ("trace_only", "tile_coding"),
        ("oracle", "identity"),
    ]
    core_values = [
        core[(core["condition"] == condition) & (core["adapter"] == adapter)][
            "final_performance"
        ]
        for condition, adapter in requested
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        np.arange(4),
        [float(values.mean()) for values in core_values],
        yerr=[float(values.sem()) if len(values) > 1 else 0.0 for values in core_values],
        capsize=3,
    )
    ax.set_xticks(
        np.arange(4),
        ["trace identity", "trace RFF", "trace tile", "oracle identity"],
        rotation=20,
    )
    ax.set_ylabel("Final 200 trials accuracy")
    ax.set_title("Core T-maze; prior trace-only junction cue decodability ≈ 1.0")
    fig.tight_layout()
    fig.savefig(output_dir / "03_core_tmaze_trace_only.png", dpi=160)
    plt.close(fig)

    hidden = summaries[
        (summaries["stage"] == "stage-b2")
        & summaries["condition"].isin(["raw", "whitened", "matched"])
    ]
    conditions = ("raw", "whitened", "matched")
    hidden_adapters = ("identity", "residual_rff", "tile_coding")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(conditions), dtype=float)
    for offset, adapter in zip((-0.25, 0.0, 0.25), hidden_adapters, strict=True):
        values = [
            hidden[
                (hidden["condition"] == condition) & (hidden["adapter"] == adapter)
            ]["final_performance"]
            for condition in conditions
        ]
        ax.bar(
            x + offset,
            [float(value.mean()) for value in values],
            width=0.25,
            yerr=[float(value.sem()) if len(value) > 1 else 0.0 for value in values],
            label=adapter,
            capsize=2,
        )
    ax.set_xticks(x, conditions)
    ax.set_ylabel("Final-window reward")
    ax.set_title("Hidden velocity adapter comparison (mean ± SEM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_hidden_velocity_adapters.png", dpi=160)
    plt.close(fig)


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
    stage_a_root: Path | None = None,
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
        stage_a_root=stage_a_root,
    )
    if formal and (not allow_full_run or os.environ.get("RL_RUN_CONTEXT") != "remote"):
        raise RuntimeError("formal utilization run requires RL_RUN_CONTEXT=remote and --allow-full-run")
    if formal and storage_report is None:
        raise RuntimeError("formal utilization run requires --storage-report from a real pilot")
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
    pending_identities = {
        record["cell"]
        for record in report["execution_plan"]
        if record["status"] == "pending"
    }
    # Stages are deliberately sequential; process-level parallelism is confined
    # within one stage so Stage B reuse can never race Stage A production.
    for stage in selected:
        pending = [
            cell
            for cell in matrix
            if cell.stage == stage and cell.identity in pending_identities
        ]
        _run_tasks(pending, suite, workers, resume, retry_invalid)
    aggregate_suite(suite, selected, report)
    return suite


def _expand_stages(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("--stage all cannot be combined with another stage")
        return FORMAL_STAGES
    if any(value in {"smoke", "storage-pilot"} for value in values) and len(values) != 1:
        raise ValueError("smoke and storage-pilot must run separately")
    ordered = tuple(dict.fromkeys(values))
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", action="append", required=True,
        choices=["stage-a", "stage-b1", "stage-b2", "all", "smoke", "storage-pilot"],
    )
    parser.add_argument("--baseline-root")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="results/utilization_adapters")
    parser.add_argument("--storage-report")
    parser.add_argument("--stage-a-root")
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
        stage_a_root = None if args.stage_a_root is None else Path(args.stage_a_root)
        report = dry_run_report(
            stages,
            baseline_root=baseline,
            storage_report=storage,
            suite_root=Path(args.output_root) / args.run_name,
            stage_a_root=stage_a_root,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run:
        printable = dict(report)
        records = printable.pop("reuse_validation")
        plan = printable.pop("execution_plan")
        printable["reuse_validation_records"] = len(records)
        printable["execution_plan_records"] = len(plan)
        printable["incompatible_details"] = [
            record for record in records if record.get("status") in {"incompatible", "duplicate"}
        ]
        print(json.dumps(printable, indent=2, sort_keys=True))
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
        stage_a_root=stage_a_root,
    )
    print(f"RESULT_DIR={result.resolve()}")


if __name__ == "__main__":
    main()
