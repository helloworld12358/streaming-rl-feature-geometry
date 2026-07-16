import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from streaming_rl_feature_geometry.cross_experiment import (
    aggregate_cross,
    run_cross_one,
    validate_cross_results,
)
from streaming_rl_feature_geometry.cross_production import package_suite, sha256_file


def _tiny_config(tmp_path):
    return {
        "profile": "resume_unit",
        "experiment_stage": "fixed",
        "controller_alpha_mode": "fixed",
        "seeds": [9000],
        "environments": {
            "hidden_velocity": {
                "interactions": 32,
                "conditions": ["raw"],
                "bank": "compact",
                "kwargs": {"noise_std": 0.0},
                "control_alpha": 0.001,
            }
        },
        "horizons": [0.5],
        "trace_dim": 4,
        "predictive_alpha": 0.005,
        "control_alpha": 0.005,
        "gamma": 0.95,
        "lambda": 0.8,
        "epsilon": 0.05,
        "transform_eps": 0.001,
        "transform_min_samples": 4,
        "cov_update_every": 4,
        "moment_beta": 0.005,
        "moment_learning_rate": 0.0005,
        "metrics_stride": 8,
        "analysis_burn_in": 4,
        "reservoir_size": 16,
        "moving_window": 4,
        "final_window": 8,
        "workers": 1,
        "output_dir": str(tmp_path),
    }


def test_resume_skips_success_and_retry_preserves_failed_attempt(tmp_path):
    config = _tiny_config(tmp_path)
    root = tmp_path / "suite" / "fixed-full"
    task = (config, "hidden_velocity", "raw", 9000, str(root))
    first = run_cross_one(task)
    run_dir = root / "runs" / "hidden_velocity" / "raw" / "seed_9000"
    original_summary_time = (run_dir / "summary.csv").stat().st_mtime_ns
    resumed = run_cross_one((*task, {"resume": True}))
    assert resumed["final_performance"] == pytest.approx(first["final_performance"])
    assert (run_dir / "summary.csv").stat().st_mtime_ns == original_summary_time
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["exit_status"] = "failed"
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    retried = run_cross_one((*task, {"resume": True, "retry_failed": True}))
    assert retried["run_status"] == "ok"
    assert list((root / "failed_attempts").rglob("manifest.json"))
    aggregate_cross(root, config)
    validate_cross_results(root, config)
    (run_dir / "summary.json").unlink()
    with pytest.raises(AssertionError, match="missing"):
        validate_cross_results(root, config)


def test_full_and_analysis_packages_have_verified_sha_and_required_inventory(tmp_path):
    suite = tmp_path / "suite"
    stage = suite / "fixed-full"
    figures = stage / "figures"
    figures.mkdir(parents=True)
    required = {
        "aggregate_summary.csv": "value\n1\n",
        "condition_summary.csv": "value\n1\n",
        "aggregate_representation.csv": "value\n1\n",
        "aggregate_task_information.csv": "value\n1\n",
        "robust_condition_summary.csv": "value\n1\n",
        "catastrophic_failure_summary.csv": "value\n0\n",
        "config.json": "{}\n",
        "manifest.json": "{}\n",
        "analysis_manifest.json": "{}\n",
        "RESULTS_SUMMARY_ZH.md": "# summary\n",
        "selected_learning_rates.csv": "environment,condition,selected_alpha\nhidden_velocity,raw,0.001\n",
    }
    for name, contents in required.items():
        (stage / name).write_text(contents, encoding="utf-8")
    (figures / "plot.png").write_bytes(b"png")
    run_dir = stage / "runs" / "hidden_velocity" / "raw" / "seed_9000"
    run_dir.mkdir(parents=True)
    pd.DataFrame([{"reward": -1.0}]).to_csv(run_dir / "step_metrics.csv", index=False)
    packages = package_suite(suite, "smoke", suite / "packages")
    for archive_key, checksum_key in (("full", "full_sha256"), ("analysis_core", "analysis_core_sha256")):
        archive = Path(packages[archive_key])
        checksum = Path(packages[checksum_key])
        assert checksum.read_text(encoding="ascii").split()[0] == sha256_file(archive)
    with tarfile.open(packages["analysis_core"], "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
    assert any(name.endswith("aggregate_summary.csv") for name in names)
    assert any(name.endswith("robust_condition_summary.csv") for name in names)
    assert any(name.endswith("selected_learning_rates.csv") for name in names)
    assert any(name.endswith("figures/plot.png") for name in names)
    with tarfile.open(packages["full"], "r:gz") as archive:
        full_names = {member.name for member in archive.getmembers()}
    assert not any("/packages/" in name for name in full_names)
