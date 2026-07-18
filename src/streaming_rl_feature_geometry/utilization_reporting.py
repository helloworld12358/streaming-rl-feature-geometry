"""Independent integrity audit, aggregation, and plotting for adapter results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .experiment import write_json
from .utilization_experiment import (
    MatrixCell,
    build_matrix,
    make_formal_figures,
    stage_config,
    validate_suite_run,
)


REQUIRED_SUMMARY_COLUMNS = {
    "stage",
    "environment",
    "condition",
    "adapter",
    "seed",
    "final_performance",
    "run_status",
    "nan_count",
    "inf_count",
    "divergence_flag",
    "wall_seconds",
    "source_summary",
    "source_commit",
    "source_config_hash",
    "reuse_type",
}


def _cell_key(row: Any) -> str:
    return (
        f"{row.stage}/{row.environment}/{row.condition}/"
        f"{row.adapter}/seed_{int(row.seed):03d}"
    )


def _load_stage_root(stage: str, root: Path) -> pd.DataFrame:
    path = root / "run_summaries.csv"
    if not path.is_file():
        raise ValueError(f"{stage} root is missing {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_SUMMARY_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    if set(frame["stage"]) != {stage}:
        raise ValueError(f"{path} contains wrong stages {sorted(set(frame['stage']))}")
    return frame


def audit_stage_roots(stage_roots: dict[str, Path]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit exact matrices and every referenced run before aggregation."""

    frames = [_load_stage_root(stage, root) for stage, root in stage_roots.items()]
    summaries = pd.concat(frames, ignore_index=True)
    actual_keys = [_cell_key(row) for row in summaries.itertuples()]
    expected_cells = build_matrix(stage_roots.keys())
    expected_keys = {cell.identity for cell in expected_cells}
    duplicates = sorted({key for key in actual_keys if actual_keys.count(key) > 1})
    actual_set = set(actual_keys)
    missing = sorted(expected_keys - actual_set)
    extra = sorted(actual_set - expected_keys)
    invalid_rows: list[dict[str, Any]] = []
    source_commits: set[str] = set()
    source_configs: set[str] = set()
    for row in summaries.itertuples():
        key = _cell_key(row)
        source = Path(str(row.source_summary))
        engine = "core" if row.stage == "stage-b1" else "cross"
        cell = MatrixCell(
            str(row.stage), engine, str(row.environment), str(row.condition),
            str(row.adapter), int(row.seed), "audit",
        )
        validation = validate_suite_run(
            source.parent, stage_config(cell.stage, cell.adapter), cell
        )
        reasons = list(validation.get("differences", []))
        if str(row.run_status) not in {"ok", "valid"}:
            reasons.append(f"run_status={row.run_status!r}")
        for name in ("nan_count", "inf_count", "divergence_flag"):
            if int(getattr(row, name)) != 0:
                reasons.append(f"{name}={getattr(row, name)!r}")
        if not np.isfinite(float(row.final_performance)):
            reasons.append("final_performance is non-finite")
        if reasons:
            invalid_rows.append({"cell": key, "reasons": reasons, "source": str(source)})
        source_commits.add(str(row.source_commit))
        source_configs.add(str(row.source_config_hash))

    unlisted_sources: list[str] = []
    listed = {str(Path(path).resolve()) for path in summaries["source_summary"]}
    for root in stage_roots.values():
        for path in root.rglob("summary.json"):
            if "runs" in path.parts and str(path.resolve()) not in listed:
                unlisted_sources.append(str(path))
    paired_groups = summaries.groupby(
        ["stage", "environment", "condition", "seed"]
    )["adapter"].apply(lambda values: "identity" in set(values))
    report = {
        "status": "valid",
        "stages": list(stage_roots),
        "expected": len(expected_keys),
        "completed": len(actual_keys),
        "valid": len(actual_keys) - len(invalid_rows),
        "invalid": len(invalid_rows),
        "failed": int((~summaries["run_status"].isin(["ok", "valid"])).sum()),
        "incomplete": len(missing),
        "duplicate": len(duplicates),
        "extra": len(extra) + len(unlisted_sources),
        "wrong_seed_or_axis": extra,
        "missing_cells": missing,
        "duplicate_cells": duplicates,
        "extra_cells": extra,
        "unlisted_result_files": unlisted_sources,
        "mixed_commits": sorted(source_commits) if len(source_commits) > 1 else [],
        "source_commits": sorted(source_commits),
        "source_config_hashes": sorted(source_configs),
        "runtime_invalid_rows": invalid_rows,
        "paired_baseline_coverage": int(paired_groups.sum()),
        "paired_baseline_expected": int(len(paired_groups)),
        "missing_paired_baseline_groups": int((~paired_groups).sum()),
        "nan_count": int(summaries["nan_count"].sum()),
        "inf_count": int(summaries["inf_count"].sum()),
        "divergence_count": int(summaries["divergence_flag"].sum()),
    }
    if any(
        (
            report["invalid"], report["failed"], report["incomplete"],
            report["duplicate"], report["extra"], len(report["mixed_commits"]),
            report["nan_count"], report["inf_count"], report["divergence_count"],
            report["missing_paired_baseline_groups"],
        )
    ):
        report["status"] = "invalid"
    return report, summaries


def _paired_table(summaries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in summaries.groupby(["stage", "environment", "condition"]):
        stage, environment, condition = keys
        identity = group[group["adapter"] == "identity"].set_index("seed")[
            "final_performance"
        ]
        for adapter in ("residual_rff", "tile_coding"):
            right = group[group["adapter"] == adapter].set_index("seed")[
                "final_performance"
            ]
            common = identity.index.intersection(right.index)
            difference = right.loc[common] - identity.loc[common]
            for seed in common:
                rows.append(
                    {
                        "stage": stage,
                        "environment": environment,
                        "condition": condition,
                        "left_adapter": "identity",
                        "right_adapter": adapter,
                        "seed": int(seed),
                        "left_performance": float(identity.loc[seed]),
                        "right_performance": float(right.loc[seed]),
                        "paired_difference": float(difference.loc[seed]),
                        "paired_mean": float(difference.mean()),
                        "paired_sem": float(difference.sem()) if len(difference) > 1 else np.nan,
                        "valid_seed_count": int(len(common)),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_stage_roots(stage_roots: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    report, summaries = audit_stage_roots(stage_roots)
    if report["status"] != "valid":
        raise RuntimeError("formal utilization audit failed; aggregation is blocked")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "run_manifest.json", summaries.to_dict(orient="records"))
    summaries.to_csv(output_dir / "run_summaries.csv", index=False)
    summaries.groupby(
        ["stage", "environment", "condition", "adapter"], as_index=False
    )["final_performance"].agg(
        mean="mean", sem="sem", valid_seed_count="count"
    ).to_csv(output_dir / "aggregate_performance.csv", index=False)
    _paired_table(summaries).to_csv(
        output_dir / "paired_adapter_differences.csv", index=False
    )
    summaries.groupby(["stage", "adapter"], as_index=False)[
        ["nan_count", "inf_count", "divergence_flag", "wall_seconds"]
    ].sum().to_csv(output_dir / "runtime_summary.csv", index=False)
    reuse_frames = []
    for root in stage_roots.values():
        path = root / "reuse_validation.csv"
        if path.is_file() and path.stat().st_size:
            reuse_frames.append(pd.read_csv(path))
    pd.concat(reuse_frames, ignore_index=True).to_csv(
        output_dir / "reuse_validation.csv", index=False
    ) if reuse_frames else pd.DataFrame().to_csv(
        output_dir / "reuse_validation.csv", index=False
    )
    pd.DataFrame(columns=["cell", "reason", "path"]).to_csv(
        output_dir / "failed_runs.csv", index=False
    )
    (output_dir / "experiment_summary.md").write_text(
        "# Utilization adapter experiment summary\n\n"
        f"- Logical cells: {len(summaries)}\n"
        f"- Valid cells: {report['valid']}\n"
        "- Interpretation is descriptive; no statistical-significance language is used.\n",
        encoding="utf-8",
    )
    make_formal_figures(summaries, output_dir / "figures")
    write_json(output_dir / "integrity_audit.json", report)
    return report


def _stage_roots(args: argparse.Namespace) -> dict[str, Path]:
    values = {
        "stage-a": args.stage_a_root,
        "stage-b1": args.stage_b1_root,
        "stage-b2": args.stage_b2_root,
    }
    roots = {stage: Path(value) for stage, value in values.items() if value is not None}
    if not roots:
        raise ValueError("at least one stage root is required")
    return roots


def audit_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-root")
    parser.add_argument("--stage-b1-root")
    parser.add_argument("--stage-b2-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report, _ = audit_stage_roots(_stage_roots(args))
    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "valid":
        raise SystemExit(2)


def aggregate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-root", required=True)
    parser.add_argument("--stage-b1-root", required=True)
    parser.add_argument("--stage-b2-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = aggregate_stage_roots(_stage_roots(args), Path(args.output_dir))
    print(json.dumps(report, indent=2, sort_keys=True))


def plotting_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summaries", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.run_summaries)
    make_formal_figures(frame, Path(args.output_dir))
    print(f"FIGURE_COUNT={len(list(Path(args.output_dir).glob('*.png')))}")
