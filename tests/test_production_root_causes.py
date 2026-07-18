import json
import multiprocessing
import pickle
from pathlib import Path

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

import streaming_rl_feature_geometry.cross_experiment as cross_experiment_module

from streaming_rl_feature_geometry.compact_storage import (
    HiddenVelocityAccumulator,
    read_columnar_npz,
)
from streaming_rl_feature_geometry.cross_experiment import (
    _completed_invalid_tuning_run,
    _completed_run,
    _hidden_velocity_summary,
    aggregate_cross,
    build_cross_tasks,
    run_cross_one,
    run_cross_worker,
    load_cross_config,
    validate_cross_results,
)
from streaming_rl_feature_geometry.cross_reporting import _condition_boxplot
from streaming_rl_feature_geometry.production_runtime import (
    RUNTIME_PATH_VARIABLES,
    contained_path,
    repository_root,
    safe_worker_limit,
    validate_runtime_environment,
)
from streaming_rl_feature_geometry.runtime_validation import (
    RuntimeFailure,
    RuntimeValidityError,
    RuntimeValidityTracker,
)
from streaming_rl_feature_geometry.storage_budget import main as storage_budget_main
from streaming_rl_feature_geometry.storage_budget import project_storage
from streaming_rl_feature_geometry.storage_budget import validated_production_peak_bytes


def _runtime_failure(candidate_alpha=0.024):
    return RuntimeFailure(
        environment="tmaze",
        condition="standardized",
        seed=100,
        candidate_alpha=candidate_alpha,
        interaction_index=2542,
        metric="control_td_error",
        observed_value=1006849798943.882,
        threshold=1e12,
        relevant_feature_norm=12.0,
        parameter_norm=210870231429.4838,
        update_norm=295081900424.02765,
        transform_state={"kind": "standardized", "sample_count": 2543},
        error_type="RuntimeValidityError",
        failure_classification="finite_extreme_numerical_divergence",
    )


def _pool_raise_runtime_validity_error(failure):
    raise RuntimeValidityError(failure)


def _write_deterministic_runtime_failure(
    config, environment, condition, seed, run_dir, started, task_options
):
    tracker = RuntimeValidityTracker(
        environment=environment,
        condition=condition,
        seed=seed,
        candidate_alpha=task_options.get("candidate_alpha"),
        run_dir=run_dir,
        extreme_finite_limit=float(config.get("extreme_finite_limit", 1e12)),
    )
    tracker.observe(
        7,
        {"control_td_error": 1.1e12},
        feature_norm=3.0,
        parameter_norm=4.0,
        update_norm=5.0,
        transform_state={"kind": condition, "sample_count": 8},
    )
    raise AssertionError("unreachable after deterministic runtime failure")


def compact_config(tmp_path):
    return {
        "profile": "compact_unit",
        "experiment_stage": "fixed",
        "controller_alpha_mode": "fixed",
        "storage_schema": "compact_v2",
        "result_schema_version": 2,
        "seeds": [0],
        "environments": {
            "hidden_velocity": {
                "interactions": 80,
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
        "reservoir_size": 32,
        "moving_window": 4,
        "final_window": 8,
        "workers": 1,
        "output_dir": str(tmp_path),
    }


def tiny_tuning_config(tmp_path):
    config = compact_config(tmp_path)
    config.update(
        profile="tiny_lr_tune_unit",
        experiment_stage="lr_tune",
        seeds=[100],
        seed_sets={
            "pilot": [200],
            "tuning": [100],
            "evaluation": [0],
            "smoke": [9000],
        },
        alpha_tuning={
            "multipliers": [1.0, 2.0],
            "selection_criterion": "robust_score",
            "failure_penalty": 1.0,
        },
    )
    config["environments"]["hidden_velocity"]["interactions"] = 32
    return config


def test_production_smoke_covers_all_required_environments_and_conditions():
    config = load_cross_config("configs/cross_extension_smoke.json")
    assert set(config["environments"]) == {
        "tmaze",
        "ringworld",
        "two_loop",
        "hidden_velocity",
        "hidden_velocity_informative",
    }
    required = {
        "observation_only",
        "oracle",
        "raw",
        "standardized",
        "whitened",
        "gaussian_moment",
        "matched",
    }
    assert all(required <= set(specification["conditions"]) for specification in config["environments"].values())


def test_repository_containment_rejects_escape_before_writing(tmp_path):
    root = repository_root()
    inside = contained_path(root / "results" / "unit", root, label="result")
    assert inside.is_relative_to(root)
    outside = root.parent / f".{root.name}-{tmp_path.name}-escaped"
    with pytest.raises(ValueError, match="escapes repository root"):
        contained_path(outside, root, label="result")
    assert not outside.exists()


def test_all_production_cache_and_temp_paths_must_be_inside_repository(monkeypatch):
    root = repository_root()
    for name in RUNTIME_PATH_VARIABLES:
        monkeypatch.setenv(name, str(root / ".runtime" / name.lower()))
    resolved = validate_runtime_environment(root)
    assert set(resolved) == set(RUNTIME_PATH_VARIABLES)
    monkeypatch.setenv("TMPDIR", str(root.parent / "escaped-tmp"))
    with pytest.raises(ValueError, match="escapes repository root"):
        validate_runtime_environment(root)


def test_worker_limit_uses_cgroup_quota_not_host_visible_cpu_count(monkeypatch):
    monkeypatch.setattr(
        "streaming_rl_feature_geometry.production_runtime.os.cpu_count", lambda: 128
    )
    monkeypatch.setattr(
        "streaming_rl_feature_geometry.production_runtime.cgroup_cpu_limit", lambda: 20.9
    )
    safe, effective, quota = safe_worker_limit()
    assert (safe, effective, quota) == (19, 20, 20.9)


def test_runtime_tracker_detects_finite_extreme_and_writes_exact_identity(tmp_path):
    tracker = RuntimeValidityTracker(
        environment="ringworld",
        condition="gaussian_moment",
        seed=9,
        candidate_alpha=0.006,
        run_dir=tmp_path,
        extreme_finite_limit=1e12,
    )
    tracker.observe(
        10,
        {"controller_parameter_norm": 3.0},
        feature_norm=2.0,
        parameter_norm=3.0,
        update_norm=0.2,
        transform_state={"kind": "gaussian_moment"},
    )
    with pytest.raises(RuntimeValidityError, match="interaction 11"):
        tracker.observe(
            11,
            {"controller_parameter_norm": 1.1e12},
            feature_norm=7.0,
            parameter_norm=1.1e12,
            update_norm=4.0,
            transform_state={"kind": "gaussian_moment", "sample_count": 12},
        )
    report = json.loads((tmp_path / "runtime_validity.json").read_text(encoding="utf-8"))
    failure = report["first_failure"]
    assert report["status"] == "invalid" and not report["resume_eligible"]
    assert (failure["environment"], failure["condition"], failure["seed"]) == (
        "ringworld",
        "gaussian_moment",
        9,
    )
    assert failure["metric"] == "controller_parameter_norm"
    assert failure["interaction_index"] == 11
    assert failure["failure_classification"] == "finite_extreme_numerical_divergence"
    assert report["preceding_context"][0]["interaction_index"] == 10


def test_runtime_validity_error_pickle_round_trip_preserves_structured_failure():
    original = RuntimeValidityError(_runtime_failure())
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored, RuntimeValidityError)
    assert isinstance(restored.failure, RuntimeFailure)
    assert restored.failure == original.failure
    assert str(restored) == str(original)


def test_runtime_validity_error_crosses_real_multiprocessing_pool_without_hanging():
    context = multiprocessing.get_context("spawn")
    failure = _runtime_failure()
    with context.Pool(1) as pool:
        result = pool.apply_async(_pool_raise_runtime_validity_error, (failure,))
        with pytest.raises(RuntimeValidityError) as caught:
            result.get(timeout=15)
    assert caught.value.failure == failure


def test_lr_tune_preserves_invalid_attempt_continues_and_selects_only_fully_valid_alpha(
    tmp_path, monkeypatch
):
    config = tiny_tuning_config(tmp_path)
    root = tmp_path / "lr-tune"
    tasks = build_cross_tasks(config, root)
    original_execute = cross_experiment_module._execute_cross_run
    invalid_alpha = 0.002

    def controlled_execute(
        run_config, environment, condition, seed, run_dir, started, task_options
    ):
        if np.isclose(float(task_options["candidate_alpha"]), invalid_alpha):
            return _write_deterministic_runtime_failure(
                run_config,
                environment,
                condition,
                seed,
                run_dir,
                started,
                task_options,
            )
        return original_execute(
            run_config,
            environment,
            condition,
            seed,
            run_dir,
            started,
            task_options,
        )

    monkeypatch.setattr(cross_experiment_module, "_execute_cross_run", controlled_execute)
    outcomes = [run_cross_worker(task) for task in tasks]
    assert [outcome["worker_status"] for outcome in outcomes].count("valid") == 1
    assert [outcome["worker_status"] for outcome in outcomes].count(
        "invalid_tuning_candidate"
    ) == 1
    aggregate_cross(root, config)
    validation = validate_cross_results(root, config)
    assert validation == {
        "result_dir": str(root.resolve()),
        "expected_attempts": 2,
        "attempted_candidates": 2,
        "valid_runs": 1,
        "invalid_attempts": 1,
    }
    invalid = pd.read_csv(root / "invalid_tuning_candidates.csv")
    assert len(invalid) == 1
    assert invalid.iloc[0].candidate_alpha == pytest.approx(invalid_alpha)
    assert invalid.iloc[0].exit_status == "invalid"
    assert invalid.iloc[0].failure_classification == "finite_extreme_numerical_divergence"
    manifest = json.loads(
        (root / invalid.iloc[0].manifest_path).read_text(encoding="utf-8")
    )
    validity = json.loads(
        (root / invalid.iloc[0].runtime_validity_path).read_text(encoding="utf-8")
    )
    assert manifest["exit_status"] == "invalid"
    assert manifest["first_failure"] == validity["first_failure"]
    assert "RuntimeValidityError" in manifest["traceback"]
    assert not (Path(manifest["result_path"]) / "summary.csv").exists()
    selected = pd.read_csv(root / "selected_learning_rates.csv")
    assert selected.iloc[0].selected_alpha == pytest.approx(0.001)
    assert selected.iloc[0].valid_candidate_count == 1
    assert selected.iloc[0].invalid_candidate_count == 1
    assert bool(selected.iloc[0].all_candidates_attempted)

    valid_task = next(
        task for task in tasks if np.isclose(float(task[5]["candidate_alpha"]), 0.001)
    )
    valid_run_dir = cross_experiment_module._task_run_dir(valid_task)
    valid_mtime = (valid_run_dir / "summary.csv").stat().st_mtime_ns
    valid_options = {**valid_task[5], "resume": True}
    assert run_cross_worker((*valid_task[:5], valid_options))["worker_status"] == "valid"
    assert (valid_run_dir / "summary.csv").stat().st_mtime_ns == valid_mtime


def test_lr_tune_invalid_resume_retry_and_provenance_mismatch_semantics(
    tmp_path, monkeypatch
):
    config = tiny_tuning_config(tmp_path)
    root = tmp_path / "lr-tune-resume"
    task = build_cross_tasks(config, root)[1]
    original_execute = cross_experiment_module._execute_cross_run
    monkeypatch.setattr(
        cross_experiment_module,
        "_execute_cross_run",
        _write_deterministic_runtime_failure,
    )
    first = run_cross_worker(task)
    assert first["worker_status"] == "invalid_tuning_candidate"
    run_dir = cross_experiment_module._task_run_dir(task)
    manifest_path = run_dir / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert _completed_invalid_tuning_run(
        run_dir,
        config,
        (task[1], task[2], task[3]),
        task[5],
    )
    for key, bad_value in (
        ("git_commit", "0" * 40),
        ("config_hash", "bad-config"),
        ("result_schema_version", 999),
    ):
        changed = dict(original_manifest)
        changed[key] = bad_value
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        assert not _completed_invalid_tuning_run(
            run_dir,
            config,
            (task[1], task[2], task[3]),
            task[5],
        )
    manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")
    invalid_mtime = manifest_path.stat().st_mtime_ns

    def unexpected_execute(*args, **kwargs):
        raise AssertionError("ordinary resume reran a completed invalid tuning attempt")

    monkeypatch.setattr(cross_experiment_module, "_execute_cross_run", unexpected_execute)
    resume_options = {**task[5], "resume": True}
    resumed = run_cross_worker((*task[:5], resume_options))
    assert resumed["worker_status"] == "invalid_tuning_candidate"
    assert manifest_path.stat().st_mtime_ns == invalid_mtime

    monkeypatch.setattr(
        cross_experiment_module,
        "_execute_cross_run",
        original_execute,
    )
    retry_options = {**task[5], "resume": True, "retry_failed": True}
    retried = run_cross_worker((*task[:5], retry_options))
    assert retried["worker_status"] == "valid"
    archived = list((root / "failed_attempts").rglob("runtime_validity.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text(encoding="utf-8"))["status"] == "invalid"


def test_lr_tune_all_candidates_invalid_fails_without_selected_alpha(tmp_path, monkeypatch):
    config = tiny_tuning_config(tmp_path)
    root = tmp_path / "all-invalid"
    monkeypatch.setattr(
        cross_experiment_module,
        "_execute_cross_run",
        _write_deterministic_runtime_failure,
    )
    outcomes = [run_cross_worker(task) for task in build_cross_tasks(config, root)]
    assert all(outcome["worker_status"] == "invalid_tuning_candidate" for outcome in outcomes)
    with pytest.raises(AssertionError, match="no fully valid"):
        aggregate_cross(root, config)
    assert len(pd.read_csv(root / "invalid_tuning_candidates.csv")) == 2
    assert not (root / "selected_learning_rates.csv").exists()


@pytest.mark.parametrize("stage", ["fixed", "lr_eval", "norm_scaled"])
def test_non_tuning_runtime_validity_error_remains_fail_closed(
    stage, tmp_path, monkeypatch
):
    config = compact_config(tmp_path)
    config["experiment_stage"] = stage
    if stage == "norm_scaled":
        config["controller_alpha_mode"] = "norm_scaled"
    root = tmp_path / stage
    task = (
        config,
        "hidden_velocity",
        "raw",
        0,
        str(root),
        {"controller_alpha": 0.001},
    )
    monkeypatch.setattr(
        cross_experiment_module,
        "_execute_cross_run",
        _write_deterministic_runtime_failure,
    )
    with pytest.raises(RuntimeValidityError):
        run_cross_worker(task)
    manifest = json.loads(
        (cross_experiment_module._task_run_dir(task) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["exit_status"] == "invalid"


def test_lr_tune_unknown_software_exception_is_not_converted_to_invalid_candidate(
    tmp_path, monkeypatch
):
    config = tiny_tuning_config(tmp_path)
    root = tmp_path / "unknown-error"
    task = build_cross_tasks(config, root)[0]

    def software_bug(*args, **kwargs):
        raise ValueError("deterministic software bug")

    monkeypatch.setattr(cross_experiment_module, "_execute_cross_run", software_bug)
    with pytest.raises(ValueError, match="software bug"):
        run_cross_worker(task)
    manifest = json.loads(
        (cross_experiment_module._task_run_dir(task) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["exit_status"] == "failed"
    assert "ValueError: deterministic software bug" in manifest["traceback"]


def test_compact_run_has_typed_partitions_and_no_duplicate_raw_aggregates(tmp_path):
    config = compact_config(tmp_path)
    root = tmp_path / "suite"
    summary = run_cross_one((config, "hidden_velocity", "raw", 0, str(root)))
    run_dir = root / "runs" / "hidden_velocity" / "raw" / "seed_000"
    assert summary["run_status"] == "valid"
    assert (run_dir / "strided_trace.npz").is_file()
    assert (run_dir / "decision_event_trace.npz").is_file()
    assert (run_dir / "runtime_validity.json").is_file()
    assert not (run_dir / "step_metrics.csv").exists()
    assert not (run_dir / "update_metrics.csv").exists()
    assert not (run_dir / "diagnostic_samples.npz").exists()
    trace = read_columnar_npz(run_dir / "strided_trace.npz")
    assert len(trace) == 10  # interactions 0, 8, ..., 72; no per-action oversampling
    aggregate_cross(root, config)
    assert not (root / "aggregate_steps.csv").exists()
    assert not (root / "aggregate_updates.csv").exists()
    inventory = json.loads((root / "trace_partition_inventory.json").read_text(encoding="utf-8"))
    assert inventory["strided_partitions"] == 1
    assert inventory["monolithic_raw_aggregates_written"] is False


def test_resume_rejects_commit_config_and_schema_mismatch(tmp_path):
    config = compact_config(tmp_path)
    root = tmp_path / "suite"
    run_cross_one((config, "hidden_velocity", "raw", 0, str(root)))
    run_dir = root / "runs" / "hidden_velocity" / "raw" / "seed_000"
    assert _completed_run(run_dir, config, ("hidden_velocity", "raw", 0), {})
    manifest_path = run_dir / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, bad_value in (
        ("git_commit", "0" * 40),
        ("config_hash", "bad-config"),
        ("result_schema_version", 999),
    ):
        changed = dict(original)
        changed[key] = bad_value
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        assert not _completed_run(run_dir, config, ("hidden_velocity", "raw", 0), {})
    manifest_path.write_text(json.dumps(original), encoding="utf-8")


def test_online_hidden_velocity_summary_matches_offline_authoritative_metrics():
    from streaming_rl_feature_geometry.cross_env import make_environment

    environment = make_environment("hidden_velocity_informative", seed=31)
    diagnostics = []
    accumulator = HiddenVelocityAccumulator(
        environment="hidden_velocity_informative",
        final_window=40,
        settling_consecutive_steps=8,
        recovery_consecutive_steps=5,
    )
    for interaction in range(400):
        _, reward, info = environment.step(interaction % environment.n_actions)
        assert reward == pytest.approx(-info.diagnostics["total_cost"])
        diagnostics.append(dict(info.diagnostics))
        accumulator.update(interaction, info.diagnostics)
    config = {"final_window": 40, "settling_consecutive_steps": 8, "recovery_consecutive_steps": 5}
    offline = _hidden_velocity_summary(diagnostics, config, "hidden_velocity_informative")
    online = accumulator.finalize()
    for key in (
        "mean_position_cost",
        "mean_velocity_cost",
        "mean_action_cost",
        "mean_total_cost",
        "position_rmse",
        "velocity_rmse",
        "final_window_total_cost",
        "boundary_hit_rate",
        "post_disturbance_position_rmse",
        "post_disturbance_velocity_rmse",
        "post_disturbance_stabilization_rate",
    ):
        assert online[key] == pytest.approx(offline[key], nan_ok=True)


def test_matplotlib_minimum_api_does_not_pass_new_boxplot_keyword(tmp_path, monkeypatch):
    observed = {}
    original = matplotlib.axes.Axes.boxplot

    def checked(self, *args, **kwargs):
        observed.update(kwargs)
        assert "tick_labels" not in kwargs
        assert "labels" not in kwargs
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "boxplot", checked)
    frame = pd.DataFrame(
        {"condition": ["raw", "raw", "whitened", "whitened"], "metric": [1, 2, 2, 3]}
    )
    _condition_boxplot(frame, "metric", tmp_path, "box.png", "compatibility")
    assert observed.get("showfliers") is True
    assert (tmp_path / "box.png").is_file()
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"matplotlib>=3.7,<4"' in pyproject


def test_storage_projection_is_pilot_calibrated_and_accounts_for_packages(
    tmp_path, monkeypatch
):
    config = compact_config(tmp_path)
    pilot = tmp_path / "pilot"
    run_cross_one((config, "hidden_velocity", "raw", 0, str(pilot)))
    (pilot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    formal = tmp_path / "formal.json"
    formal.write_text(json.dumps(config), encoding="utf-8")
    report = project_storage(pilot, [formal], safety_factor=1.1)
    assert report["pilot_runs"] == 1 and report["formal_runs"] == 1
    assert report["projected_result_bytes"] > 0
    assert report["projected_full_package_bytes"] > 0
    assert report["projected_peak_bytes"] > report["projected_result_bytes"]
    assert report["largest_file_categories"]
    scaled_config = dict(config)
    scaled_config["environments"] = json.loads(json.dumps(config["environments"]))
    scaled_config["environments"]["hidden_velocity"]["interactions"] = 8_000
    formal.write_text(json.dumps(scaled_config), encoding="utf-8")
    scaled = project_storage(pilot, [formal], safety_factor=1.0)
    categories = {row["category"]: row for row in scaled["largest_file_categories"]}
    assert categories["compact_traces"]["projected_bytes"] > 50 * categories["compact_traces"]["pilot_bytes"]
    assert categories["manifests_and_logs"]["projected_bytes"] == categories["manifests_and_logs"]["pilot_bytes"]
    assert scaled["projected_full_package_bytes"] >= scaled["projected_result_bytes"]
    assert scaled["package_projection_method"] == "uncompressed_source_plus_tar_header_budget"
    formal.write_text(json.dumps(config), encoding="utf-8")
    production_report = dict(report)
    production_report.update(
        pilot_runs=50,
        pilot_valid_runs=50,
        pilot_git_dirty_flags=[False],
        pilot_result_schema_versions=[2],
        formal_runs=4200,
        formal_configs=[
            {
                "profile": profile,
                "storage_schema": "compact_v2",
                "result_schema_version": 2,
            }
            for profile in (
                "cross_extension_fixed_full",
                "cross_extension_lr_tune_full",
                "cross_extension_lr_eval_full",
                "cross_extension_norm_scaled_full",
            )
        ],
    )
    production_path = tmp_path / "production-storage.json"
    production_path.write_text(json.dumps(production_report), encoding="utf-8")
    expected_commit = production_report["pilot_git_commits"][0]
    assert (
        validated_production_peak_bytes(production_path, expected_commit)
        == production_report["projected_peak_bytes"]
    )
    production_report["pilot_git_dirty_flags"] = [True]
    production_path.write_text(json.dumps(production_report), encoding="utf-8")
    with pytest.raises(ValueError, match="dirty worktree"):
        validated_production_peak_bytes(production_path, expected_commit)
    output = tmp_path / "nested" / "storage_projection.json"
    inventory = tmp_path / "nested" / "storage_inventory.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "storage-budget",
            "--pilot-dir",
            str(pilot),
            "--formal-config",
            str(formal),
            "--output",
            str(output),
            "--inventory-csv",
            str(inventory),
        ],
    )
    storage_budget_main()
    assert json.loads(output.read_text(encoding="utf-8"))["formal_runs"] == 1
    assert inventory.is_file()
