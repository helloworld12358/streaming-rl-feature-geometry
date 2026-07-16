import json
import tarfile

import pandas as pd
import pytest

from streaming_rl_feature_geometry.experiment import _manifest_base
from streaming_rl_feature_geometry.remote_deployment import (
    config_plan,
    expected_run_count,
    initialize_suite,
    inspect_run,
    load_run_plan,
    load_suite_plan,
    package_suite,
    sha256_file,
)


def _core_config():
    return {
        "profile": "unit",
        "seeds": [0, 1],
        "conditions": ["raw", "whitened"],
        "total_interactions": 16,
        "horizons": [0.6, 0.9],
        "gvf_bank": "mixed",
        "output_dir": "results/unit",
    }


def _write_success(run_dir):
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"exit_status": "ok"}), encoding="utf-8"
    )
    pd.DataFrame([{"run_status": "ok"}]).to_csv(run_dir / "summary.csv", index=False)


def test_config_planning_filters_seeds_without_mutating_profile(tmp_path):
    config = _core_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    plan = config_plan(path, seed_start=1, seed_end=1)
    assert plan["kind"] == "core"
    assert plan["seeds"] == [1]
    assert plan["expected_runs"] == 2
    assert expected_run_count(config) == 4
    assert json.loads(path.read_text(encoding="utf-8"))["seeds"] == [0, 1]


def test_missing_and_failed_seed_detection_is_explicit(tmp_path):
    config = _core_config()
    root = tmp_path / "batch"
    _write_success(root / "runs" / "raw" / "seed_000")
    failed = root / "runs" / "raw" / "seed_001"
    _write_success(failed)
    (failed / "manifest.json").write_text(
        json.dumps({"exit_status": "failed"}), encoding="utf-8"
    )
    report = inspect_run(root, config)
    assert report["expected_runs"] == 4
    assert report["completed_runs"] == 1
    assert len(report["failed_runs"]) == 1
    assert report["missing_runs"] == ["whitened/seed_000", "whitened/seed_001"]
    assert not report["ok"]


def test_remote_plans_reference_executable_configs():
    smoke = load_run_plan("configs/remote_smoke.json")
    suite = load_suite_plan("configs/remote_full_suite.json")
    assert {record["id"] for record in smoke} == {
        "core-smoke",
        "cross-smoke",
        "extension-smoke",
    }
    assert len(suite) == 9
    assert any(record["kind"] == "cross_extension" for record in suite)
    assert {record["stage"] for record in suite} == {
        "core",
        "cross_environment",
        "production_extension",
        "nonstationary",
    }


def test_suite_directory_is_never_reused(tmp_path):
    suite = tmp_path / "unique-suite"
    initialize_suite(suite, "unit", "configs/remote_full_suite.json", 1, {"core"})
    with pytest.raises(FileExistsError):
        initialize_suite(suite, "unit", "configs/remote_full_suite.json", 1, {"core"})


def test_analysis_package_has_sha_and_excludes_raw_step_stream(tmp_path):
    suite = tmp_path / "suite"
    batch = suite / "core" / "run"
    figures = batch / "figures"
    figures.mkdir(parents=True)
    (suite / "suite_manifest.json").write_text("{}\n", encoding="utf-8")
    (batch / "config.json").write_text("{}\n", encoding="utf-8")
    (batch / "manifest.json").write_text("{}\n", encoding="utf-8")
    (batch / "aggregate_summary.csv").write_text("value\n1\n", encoding="utf-8")
    (batch / "step_metrics.csv").write_text("raw\n1\n", encoding="utf-8")
    (figures / "plot.png").write_bytes(b"png")
    result = package_suite(suite)
    assert sha256_file(result["archive"]) == result["sha256"]
    with tarfile.open(result["archive"], "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
    assert any(name.endswith("aggregate_summary.csv") for name in names)
    assert any(name.endswith("figures/plot.png") for name in names)
    assert any(name.endswith("repository_configs/remote_full_suite.json") for name in names)
    assert not any(name.endswith("step_metrics.csv") for name in names)


def test_run_manifest_records_resources_and_reproducibility_fields():
    config = _core_config()
    manifest = _manifest_base(config, "raw", 0)
    assert manifest["predictive_bank"] == "mixed"
    assert manifest["horizons"] == [0.6, 0.9]
    assert manifest["interaction_budget"] == 16
    assert manifest["gpu_backend_used"] == "none"
    assert manifest["parallelism"] == "cpu-process"
