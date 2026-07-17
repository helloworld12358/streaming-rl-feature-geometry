"""Full-horizon diagnostic reruns for the production failures reported in July 2026."""

from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from streaming_rl_feature_geometry.cross_experiment import (
    aggregate_cross,
    build_cross_tasks,
    load_cross_config,
    run_cross_one,
    validate_cross_results,
)
from streaming_rl_feature_geometry.experiment import write_json
from streaming_rl_feature_geometry.production_runtime import contained_path, repository_root


KNOWN_CASES = {
    "tmaze": {"condition": "gaussian_moment", "seeds": [4, 8, 13, 14]},
    "ringworld": {"condition": "gaussian_moment", "seeds": [9, 10, 15]},
    "two_loop": {"condition": "gaussian_moment", "seeds": [0, 1, 2, 7, 10]},
    "hidden_velocity": {"condition": "whitened", "seeds": [1]},
}

BASELINE_EVIDENCE = {
    ("tmaze", "gaussian_moment", 13): {
        "baseline_max_control_update_norm": 5.320254234653238e66,
        "baseline_final_parameter_norm": 1.3430143965076576e66,
    },
    ("ringworld", "gaussian_moment", 9): {
        "baseline_max_control_update_norm": 1.693528905671e95,
        "baseline_final_parameter_norm": 1.871338738720e93,
    },
    ("two_loop", "gaussian_moment", 10): {
        "baseline_max_control_update_norm": 9.138643054291e66,
        "baseline_final_parameter_norm": 1.302336021700e65,
    },
    ("hidden_velocity", "whitened", 1): {
        "baseline_max_control_update_norm": 1.038407811160e24,
        "baseline_final_parameter_norm": 1.189644846372e23,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-dir", default=".local_diagnostics/known_failure_full_horizon"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    repository = repository_root()
    root = contained_path(arguments.output_dir, repository, label="diagnostic_output_dir")
    if root.exists() and not (arguments.resume or arguments.retry_invalid):
        raise FileExistsError(f"{root} exists; use --resume or --retry-invalid")
    root.mkdir(parents=True, exist_ok=True)
    base = load_cross_config("configs/cross_extension_fixed_full.json")
    summaries: list[pd.DataFrame] = []
    for environment, case in KNOWN_CASES.items():
        config = copy.deepcopy(base)
        config["profile"] = f"known_failure_full_horizon_{environment}"
        config["remote_full"] = False
        config["seeds"] = list(case["seeds"])
        config["environments"] = {environment: config["environments"][environment]}
        config["environments"][environment]["conditions"] = [case["condition"]]
        config["workers"] = arguments.workers
        stage_root = root / environment
        stage_root.mkdir(parents=True, exist_ok=True)
        write_json(stage_root / "config.json", config)
        tasks = build_cross_tasks(
            config,
            stage_root,
            resume=arguments.resume or arguments.retry_invalid,
            retry_failed=arguments.retry_invalid,
        )
        if arguments.workers == 1:
            list(map(run_cross_one, tasks))
        else:
            with Pool(min(arguments.workers, len(tasks))) as pool:
                pool.map(run_cross_one, tasks)
        aggregate_cross(stage_root, config)
        validate_cross_results(stage_root, config)
        frame = pd.read_csv(stage_root / "aggregate_summary.csv")
        summaries.append(frame)
    observed = pd.concat(summaries, ignore_index=True, sort=False)
    rows = []
    for row in observed.itertuples(index=False):
        key = (row.environment, row.condition, int(row.seed))
        record = {
            "environment": row.environment,
            "condition": row.condition,
            "seed": int(row.seed),
            "interactions": int(row.interactions),
            "post_max_control_update_norm": float(row.max_control_update_norm),
            "post_final_parameter_norm": float(row.final_parameter_norm),
            "post_max_feature_squared_norm": float(row.max_feature_squared_norm),
            "nan_count": int(row.nan_count),
            "inf_count": int(row.inf_count),
            "divergence_flag": int(row.divergence_flag),
            "run_status": row.run_status,
            "baseline_known_failure": True,
            **BASELINE_EVIDENCE.get(key, {}),
        }
        record["passes_interpretable_range"] = bool(
            record["nan_count"] == 0
            and record["inf_count"] == 0
            and record["divergence_flag"] == 0
            and np.isfinite(record["post_max_control_update_norm"])
            and record["post_max_control_update_norm"] <= 100.0
            and np.isfinite(record["post_final_parameter_norm"])
            and record["post_final_parameter_norm"] <= 1e12
        )
        rows.append(record)
    report = pd.DataFrame(rows).sort_values(["environment", "seed"])
    report.to_csv(root / "before_after_validation.csv", index=False)
    write_json(root / "before_after_validation.json", report.to_dict(orient="records"))
    result = {
        "cases": len(report),
        "passed": int(report["passes_interpretable_range"].sum()),
        "failed": int((~report["passes_interpretable_range"]).sum()),
        "report": str((root / "before_after_validation.json").resolve()),
    }
    print(json.dumps(result, sort_keys=True))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
