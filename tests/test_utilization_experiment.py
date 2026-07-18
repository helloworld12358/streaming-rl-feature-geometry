import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import streaming_rl_feature_geometry.cross_experiment as cross_module
import streaming_rl_feature_geometry.utilization_experiment as utilization_module
import streaming_rl_feature_geometry.utilization_reporting as reporting_module
from streaming_rl_feature_geometry.cross_experiment import load_cross_config, run_cross_one
from streaming_rl_feature_geometry.experiment import config_hash, run_one, write_json
from streaming_rl_feature_geometry.utilization_experiment import (
    FORMAL_STAGES,
    MatrixCell,
    build_matrix,
    dry_run_report,
    stage_config,
    run_experiment,
    make_formal_figures,
    validate_baseline_run,
    cell_signature,
)
from streaming_rl_feature_geometry.utilization_reporting import (
    _paired_table,
    aggregate_stage_roots,
    audit_stage_roots,
)
from streaming_rl_feature_geometry.utilization_storage import (
    REQUIRED_RUN_FILES,
    build_storage_report,
)


def test_registered_formal_matrices_have_exact_preregistered_counts():
    stage_a = build_matrix(["stage-a"])
    stage_b1 = build_matrix(["stage-b1"])
    stage_b2 = build_matrix(["stage-b2"])
    assert len(stage_a) == 1120
    assert sum(cell.disposition == "baseline_candidate" for cell in stage_a) == 560
    assert sum(cell.disposition == "new" for cell in stage_a) == 560
    assert len(stage_b1) == 120
    assert sum(cell.disposition == "new" for cell in stage_b1) == 120
    assert len(stage_b2) == 150
    assert sum(cell.disposition == "stage_a_candidate" for cell in stage_b2) == 100
    assert sum(cell.disposition == "new" for cell in stage_b2) == 50


def test_smoke_covers_all_registered_axes_with_bounded_interactions():
    raw = json.loads(Path("configs/utilization_adapter_smoke.json").read_text(encoding="utf-8"))
    assert set(raw["environments"]) == {
        "tmaze", "ringworld", "two_loop", "hidden_velocity"
    }
    assert {adapter["name"] for adapter in raw["utilization_adapters"]} == {
        "identity", "residual_rff", "tile_coding"
    }
    for spec in raw["environments"].values():
        assert set(spec["conditions"]) >= {"observation_only", "raw", "matched", "oracle"}
        assert int(spec["interactions"]) <= 1000


def test_formal_profiles_exclude_informative_environment_and_preserve_fixed_alpha():
    for stage in FORMAL_STAGES:
        raw = json.loads(
            Path(f"configs/utilization_{stage.replace('-', '_')}_full.json").read_text(
                encoding="utf-8"
            )
        )
        assert raw["controller_alpha_mode"] == "fixed"
        assert raw["storage_schema"] == "adapter_summary_v1"
        if "environments" in raw:
            assert "hidden_velocity_informative" not in raw["environments"]


def test_all_dry_run_reports_missing_baseline_as_pending_without_shrinking_matrix():
    report = dry_run_report(FORMAL_STAGES)
    assert report["logical_cells"] == 1390
    assert report["missing_baseline_cells"] == 560
    assert report["missing_stage_a_cells"] == 0
    assert report["reusable_identity_cells"] == 0
    assert report["stage_a_reuse_cells"] == 100
    assert report["cross_stage_deduplicated_cells"] == 100
    assert report["new_pending_runs"] == 1290
    assert report["pending_by_adapter"] == {
        "identity": 600, "residual_rff": 600, "tile_coding": 90
    }
    assert not report["baseline_ready"]
    assert report["execution_ready"]
    assert report["duplicate_cells"] == report["incompatible_cells"] == 0
    assert report["expected_output_bytes"] > 0
    assert report["estimated_peak_disk_bytes"] >= report["expected_output_bytes"]


def test_stage_b2_alone_reports_external_and_stage_a_dependencies():
    report = dry_run_report(["stage-b2"])
    assert report["logical_cells"] == 150
    assert report["missing_baseline_cells"] == 0
    assert report["missing_stage_a_cells"] == 100
    assert report["new_pending_runs"] == 150
    assert report["execution_ready"]


def test_stage_b2_strictly_reuses_all_matching_stage_a_identity_and_rff(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        utilization_module,
        "_stage_a_source_path",
        lambda root, cell: tmp_path / cell.adapter / cell.condition / str(cell.seed),
    )
    monkeypatch.setattr(
        utilization_module,
        "validate_suite_run",
        lambda run_dir, config, cell: {
            "status": "compatible", "source": str(run_dir), "differences": []
        },
    )
    report = dry_run_report(["stage-b2"], stage_a_root=tmp_path)
    assert report["stage_a_reuse_cells"] == 100
    assert report["new_pending_runs"] == 50
    assert report["pending_by_adapter"] == {
        "identity": 0, "residual_rff": 0, "tile_coding": 50
    }


def test_formal_aggregation_generates_only_four_registered_figures(tmp_path):
    rows = []
    for environment in ("tmaze", "ringworld", "two_loop", "hidden_velocity"):
        for adapter in ("identity", "residual_rff"):
            rows.append(
                {
                    "stage": "stage-a", "environment": environment,
                    "condition": "raw", "adapter": adapter, "seed": 0,
                    "final_performance": 0.1,
                }
            )
    for adapter in ("identity", "residual_rff", "tile_coding"):
        rows.append(
            {
                "stage": "stage-b1", "environment": "tmaze", "condition": "trace_only",
                "adapter": adapter, "seed": 0, "final_performance": 0.2,
            }
        )
        for condition in ("raw", "whitened", "matched"):
            rows.append(
                {
                    "stage": "stage-b2", "environment": "hidden_velocity",
                    "condition": condition, "adapter": adapter, "seed": 0,
                    "final_performance": -0.2,
                }
            )
    rows.append(
        {
            "stage": "stage-b1", "environment": "tmaze", "condition": "oracle",
            "adapter": "identity", "seed": 0, "final_performance": 0.9,
        }
    )
    make_formal_figures(pd.DataFrame(rows), tmp_path)
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "01_stage_a_identity_vs_rff.png",
        "02_stage_a_paired_rff_minus_identity.png",
        "03_core_tmaze_trace_only.png",
        "04_hidden_velocity_adapters.png",
    }


def test_formal_guard_fails_locally_but_missing_baseline_is_a_valid_remote_plan(tmp_path):
    report_path = tmp_path / "storage.json"
    write_json(
        report_path,
        {
            "schema": "adapter_storage_pilot_v1",
            "status": "valid",
            "pilot_runs": 1,
            "adapter_summary_bytes_per_run": 1000,
        },
    )
    with pytest.raises(RuntimeError, match="RL_RUN_CONTEXT"):
        run_experiment(
            ["stage-a"], baseline_root=None, workers=1, run_name="blocked",
            output_root=tmp_path, storage_report=report_path, allow_full_run=False,
            resume=False, retry_invalid=False,
        )
    assert not (tmp_path / "blocked").exists()
    report = dry_run_report(["stage-a"], baseline_root=None, storage_report=report_path)
    assert report["new_pending_runs"] == 1120
    assert report["missing_baseline_cells"] == 560
    assert report["execution_ready"]


def test_stage_a_copies_scientific_fields_from_fixed_cross_full_source():
    source = load_cross_config("configs/cross_full.json")
    target = stage_config("stage-a", "identity")
    global_fields = {
        "seeds", "seed_sets", "horizons", "trace_dim", "predictive_alpha",
        "control_alpha", "gamma", "lambda", "epsilon", "transform_eps",
        "transform_min_samples", "cov_update_every", "moment_beta",
        "moment_learning_rate", "covariance_shrinkage", "matrix_smoothing",
        "metrics_stride", "analysis_burn_in", "reservoir_size", "moving_window",
        "final_window",
    }
    for field in global_fields:
        assert target[field] == source[field]
    assert set(target["environments"]) == {
        "tmaze", "ringworld", "two_loop", "hidden_velocity"
    }
    for environment in target["environments"]:
        target_spec = dict(target["environments"][environment])
        source_spec = dict(source["environments"][environment])
        target_spec.pop("conditions")
        source_spec.pop("conditions")
        assert target_spec == source_spec
    assert target["storage_schema"] == "adapter_summary_v1"
    assert target["controller_alpha_mode"] == "fixed"


def _write_compatible_baseline(
    run_dir: Path, expected: dict, cell: MatrixCell, commit: str | None = None
) -> None:
    run_dir.mkdir(parents=True)
    observed = dict(expected)
    commit = commit or cross_module.git_value("rev-parse", "HEAD")
    manifest = {
        "environment": cell.environment,
        "condition": cell.condition,
        "seed": cell.seed,
        "config_hash": config_hash(observed),
        "git_commit": commit,
        "source_commit": commit,
        "utilization_adapter": {"name": "identity"},
        "controller_input_definition": "adapter_of_complete_controller_input_observation_then_condition_state_then_bias_v2",
        "exit_status": "ok",
        "resume_eligible": True,
    }
    write_json(run_dir / "config.json", observed)
    write_json(run_dir / "manifest.json", manifest)
    write_json(
        run_dir / "runtime_validity.json",
        {"status": "valid", "resume_eligible": True},
    )
    write_json(
        run_dir / "summary.json",
        {
            "run_status": "valid",
            "final_performance": 0.5,
            "nan_count": 0,
            "inf_count": 0,
            "divergence_flag": 0,
        },
    )


def test_baseline_validator_is_fieldwise_strict_and_fail_closed(tmp_path):
    expected = stage_config("stage-a", "identity")
    cell = next(
        cell
        for cell in build_matrix(["stage-a"])
        if cell.environment == "tmaze"
        and cell.condition == "raw"
        and cell.adapter == "identity"
        and cell.seed == 0
    )
    run_dir = tmp_path / "run"
    _write_compatible_baseline(run_dir, expected, cell)
    assert validate_baseline_run(run_dir, expected, cell)["status"] == "compatible"
    observed = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    observed["gamma"] = 0.1
    write_json(run_dir / "config.json", observed)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["config_hash"] = config_hash(observed)
    write_json(run_dir / "manifest.json", manifest)
    record = validate_baseline_run(run_dir, expected, cell)
    assert record["status"] == "incompatible"
    assert any("gamma" in difference for difference in record["differences"])


def test_synthetic_complete_baseline_reuses_all_stage_a_identity_cells(tmp_path):
    expected = stage_config("stage-a", "identity")
    commit = cross_module.git_value("rev-parse", "HEAD")
    identity_cells = [
        cell for cell in build_matrix(["stage-a"]) if cell.adapter == "identity"
    ]
    for cell in identity_cells:
        run_dir = (
            tmp_path / "runs" / cell.environment / cell.condition
            / f"seed_{cell.seed:03d}"
        )
        _write_compatible_baseline(run_dir, expected, cell, commit)
    report = dry_run_report(FORMAL_STAGES, baseline_root=tmp_path)
    assert report["reusable_identity_cells"] == 560
    assert report["missing_baseline_cells"] == 0
    assert report["incompatible_baseline_cells"] == 0
    assert report["new_pending_runs"] == 730
    assert report["pending_by_adapter"] == {
        "identity": 40, "residual_rff": 600, "tile_coding": 90
    }


def tiny_adapter_config(tmp_path, adapter):
    value = json.loads(Path("configs/utilization_adapter_smoke.json").read_text(encoding="utf-8"))
    value["profile"] = "utilization_unit"
    value["seeds"] = [0]
    value["environments"] = {
        "tmaze": {
            "interactions": 80,
            "conditions": ["raw"],
            "bank": "mixed",
            "kwargs": {"corridor_length": 4},
            "threshold": 0.8,
        }
    }
    value["analysis_burn_in"] = 20
    value["final_window"] = 20
    value["utilization_adapter"] = adapter
    value.pop("utilization_adapters")
    value["output_dir"] = str(tmp_path)
    path = tmp_path / f"{adapter['name']}.json"
    write_json(path, value)
    return load_cross_config(path)


@pytest.mark.parametrize(
    "adapter",
    [
        {"name": "identity"},
        {
            "name": "residual_rff", "width": 64,
            "frequency_scale": 1.0, "residual": True,
        },
        {
            "name": "tile_coding", "num_tilings": 8,
            "table_size": 512, "tiles_per_unit": 4.0, "residual": True,
        },
    ],
)
def test_adapter_summary_run_is_finite_and_writes_only_lightweight_files(tmp_path, adapter):
    config = tiny_adapter_config(tmp_path, adapter)
    root = tmp_path / adapter["name"]
    summary = run_cross_one((config, "tmaze", "raw", 0, str(root)))
    run_dir = root / "runs" / "tmaze" / "raw" / "seed_000"
    assert {path.name for path in run_dir.iterdir()} == REQUIRED_RUN_FILES
    assert summary["adapter_name"] == adapter["name"]
    assert summary["controller_alpha_mode"] == "fixed"
    assert summary["nan_count"] == summary["inf_count"] == summary["divergence_flag"] == 0
    numeric = [
        summary["final_performance"], summary["cumulative_reward"],
        summary["final_parameter_norm"], summary["mean_feature_squared_norm"],
        summary["max_feature_squared_norm"],
    ]
    assert np.isfinite(numeric).all()


def test_real_lightweight_runs_produce_storage_projection(tmp_path):
    root = tmp_path / "pilot"
    adapters = [
        {"name": "identity"},
        {"name": "residual_rff", "width": 64, "frequency_scale": 1.0, "residual": True},
        {"name": "tile_coding", "num_tilings": 8, "table_size": 512, "tiles_per_unit": 4.0, "residual": True},
    ]
    for adapter in adapters:
        config = tiny_adapter_config(tmp_path, adapter)
        run_cross_one((config, "tmaze", "raw", 0, str(root / adapter["name"])))
    report, inventory = build_storage_report(root)
    assert report["status"] == "valid"
    assert report["pilot_runs"] == len(inventory) == 3
    assert report["adapter_summary_bytes_per_run"] > 0
    assert report["formal_new_runs"] == 1290
    assert set(report["bytes_per_run_by_adapter"]) == {
        "identity", "residual_rff", "tile_coding"
    }
    assert report["peak_plus_10_gib_bytes"] > report["estimated_peak_disk_bytes"]


@pytest.mark.parametrize("adapter", ["identity", "residual_rff", "tile_coding"])
def test_core_trace_only_uses_same_adapter_hook_and_lightweight_schema(tmp_path, adapter):
    config = stage_config("stage-b1", adapter)
    config["profile"] = "utilization_core_unit"
    config["seeds"] = [0]
    config["conditions"] = ["trace_only"]
    config["total_interactions"] = 100
    config["analysis_burn_in"] = 20
    config["reservoir_size"] = 32
    root = tmp_path / adapter
    summary = run_one((config, "trace_only", 0, str(root)))
    run_dir = root / "runs" / "trace_only" / "seed_000"
    assert {path.name for path in run_dir.iterdir()} == REQUIRED_RUN_FILES
    assert summary["adapter"] == adapter
    assert summary["adapter_input_dim"] == config["trace_dim"] + 8 + 1
    assert summary["controller_alpha_mode"] == "fixed"
    assert summary["run_status"] == "valid"


def test_legacy_and_explicit_identity_have_exact_training_trajectory(tmp_path, monkeypatch):
    identity = {"name": "identity"}
    explicit = tiny_adapter_config(tmp_path, identity)
    legacy = dict(explicit)
    legacy.pop("utilization_adapter")
    captures = []
    active = {"features": [], "actions": [], "updates": [], "controllers": []}

    original_features = cross_module._controller_features
    original_act = cross_module.SarsaLambda.act
    original_update = cross_module.SarsaLambda.update
    original_init = cross_module.SarsaLambda.__init__

    def recording_features(observation, state):
        value = original_features(observation, state)
        active["features"].append(value.copy())
        return value

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        active["controllers"].append(self)

    def recording_act(self, features):
        action = original_act(self, features)
        active["actions"].append(action)
        return action

    def recording_update(self, features, action, reward, next_features, next_action):
        result = original_update(self, features, action, reward, next_features, next_action)
        active["updates"].append(
            (features.copy(), action, reward, next_features.copy(), next_action, result)
        )
        return result

    monkeypatch.setattr(cross_module, "_controller_features", recording_features)
    monkeypatch.setattr(cross_module.SarsaLambda, "__init__", recording_init)
    monkeypatch.setattr(cross_module.SarsaLambda, "act", recording_act)
    monkeypatch.setattr(cross_module.SarsaLambda, "update", recording_update)

    summaries = []
    for name, config in (("legacy", legacy), ("identity", explicit)):
        active = {"features": [], "actions": [], "updates": [], "controllers": []}
        summaries.append(run_cross_one((config, "tmaze", "raw", 0, str(tmp_path / name))))
        captures.append(active)

    assert len(captures[0]["features"]) == len(captures[1]["features"])
    for left, right in zip(captures[0]["features"], captures[1]["features"], strict=True):
        np.testing.assert_array_equal(left, right)
    assert captures[0]["actions"] == captures[1]["actions"]
    assert len(captures[0]["updates"]) == len(captures[1]["updates"])
    for left, right in zip(captures[0]["updates"], captures[1]["updates"], strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:3] == right[1:3]
        np.testing.assert_array_equal(left[3], right[3])
        assert left[4:] == right[4:]
    np.testing.assert_array_equal(
        captures[0]["controllers"][0].w, captures[1]["controllers"][0].w
    )
    for summary in summaries:
        summary.pop("wall_seconds")
        summary.pop("run_dir")
    assert summaries[0] == summaries[1]


def test_observation_only_adapter_receives_observation_and_bias(tmp_path):
    adapter = {
        "name": "residual_rff", "width": 64,
        "frequency_scale": 1.0, "residual": True,
    }
    config = tiny_adapter_config(tmp_path, adapter)
    config["environments"]["tmaze"]["conditions"] = ["observation_only"]
    root = tmp_path / "complete-input"
    summary = run_cross_one((config, "tmaze", "observation_only", 0, str(root)))
    assert summary["adapter_input_dim"] == 8 + 1
    assert summary["adapter_block_dim"] == 64
    assert summary["final_controller_dim"] == 8 + 1 + 64
    manifest = json.loads(
        (root / "runs" / "tmaze" / "observation_only" / "seed_000" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["controller_input_definition"].endswith(
        "observation_then_condition_state_then_bias_v2"
    )
    for field in (
        "run_id", "source_commit", "adapter_seed", "adapter_input_dim",
        "adapter_block_dim", "final_controller_dim", "interaction_budget",
        "final_window", "controller_alpha_mode", "control_alpha",
        "predictive_alpha", "gamma", "lambda", "epsilon", "bank",
        "environment_kwargs", "config_hash", "result_schema",
    ):
        assert field in manifest


def test_stage_b2_reuse_signature_matches_stage_a_for_complete_input():
    stage_a = MatrixCell(
        "stage-a", "cross", "hidden_velocity", "observation_only",
        "residual_rff", 0, "new",
    )
    stage_b2 = MatrixCell(
        "stage-b2", "cross", "hidden_velocity", "observation_only",
        "residual_rff", 0, "stage_a_candidate",
    )
    assert cell_signature(
        stage_config("stage-a", "residual_rff"), stage_a
    ) == cell_signature(stage_config("stage-b2", "residual_rff"), stage_b2)


def test_synthetic_paired_aggregation_has_required_fields_and_values():
    frame = pd.DataFrame(
        [
            {"stage": "stage-a", "environment": "tmaze", "condition": "raw", "adapter": "identity", "seed": 0, "final_performance": 0.4},
            {"stage": "stage-a", "environment": "tmaze", "condition": "raw", "adapter": "residual_rff", "seed": 0, "final_performance": 0.7},
            {"stage": "stage-a", "environment": "tmaze", "condition": "raw", "adapter": "identity", "seed": 1, "final_performance": 0.5},
            {"stage": "stage-a", "environment": "tmaze", "condition": "raw", "adapter": "residual_rff", "seed": 1, "final_performance": 0.6},
        ]
    )
    paired = _paired_table(frame)
    assert set(
        [
            "environment", "condition", "left_adapter", "right_adapter", "seed",
            "left_performance", "right_performance", "paired_difference",
            "paired_mean", "paired_sem", "valid_seed_count",
        ]
    ) <= set(paired.columns)
    assert paired["paired_difference"].tolist() == pytest.approx([0.3, 0.1])
    assert paired["paired_mean"].tolist() == pytest.approx([0.2, 0.2])
    assert paired["valid_seed_count"].tolist() == [2, 2]


def test_synthetic_integrity_audit_accepts_exact_cells_and_rejects_duplicates(
    tmp_path, monkeypatch
):
    root = tmp_path / "stage-a"
    root.mkdir()
    cell = MatrixCell("stage-a", "cross", "tmaze", "raw", "identity", 0, "new")
    row = {
        "stage": "stage-a", "environment": "tmaze", "condition": "raw",
        "adapter": "identity", "seed": 0, "final_performance": 0.5,
        "run_status": "valid", "nan_count": 0, "inf_count": 0,
        "divergence_flag": 0, "wall_seconds": 1.0,
        "source_summary": str(root / "source" / "summary.json"),
        "source_commit": "same", "source_config_hash": "cfg",
        "reuse_type": "pending",
    }
    pd.DataFrame([row]).to_csv(root / "run_summaries.csv", index=False)
    monkeypatch.setattr(reporting_module, "build_matrix", lambda stages: [cell])
    monkeypatch.setattr(
        reporting_module, "validate_suite_run",
        lambda run_dir, config, expected: {"status": "compatible", "differences": []},
    )
    monkeypatch.setattr(reporting_module, "stage_config", lambda stage, adapter: {})
    report, _ = audit_stage_roots({"stage-a": root})
    assert report["status"] == "valid"
    pd.DataFrame([row, row]).to_csv(root / "run_summaries.csv", index=False)
    report, _ = audit_stage_roots({"stage-a": root})
    assert report["status"] == "invalid"
    assert report["duplicate"] == 1


def test_synthetic_end_to_end_aggregation_writes_tables_and_only_four_figures(
    tmp_path, monkeypatch
):
    rows = []
    for environment in ("tmaze", "ringworld", "two_loop", "hidden_velocity"):
        for adapter in ("identity", "residual_rff"):
            rows.append(
                {
                    "stage": "stage-a", "environment": environment,
                    "condition": "raw", "adapter": adapter, "seed": 0,
                    "final_performance": 0.5, "nan_count": 0, "inf_count": 0,
                    "divergence_flag": 0, "wall_seconds": 1.0,
                    "run_status": "valid", "source_summary": "synthetic",
                    "source_commit": "same", "source_config_hash": "cfg",
                    "reuse_type": "pending",
                }
            )
    for adapter in ("identity", "residual_rff", "tile_coding"):
        rows.append(
            {
                "stage": "stage-b1", "environment": "tmaze",
                "condition": "trace_only", "adapter": adapter, "seed": 0,
                "final_performance": 0.5, "nan_count": 0, "inf_count": 0,
                "divergence_flag": 0, "wall_seconds": 1.0,
                "run_status": "valid", "source_summary": "synthetic",
                "source_commit": "same", "source_config_hash": "cfg",
                "reuse_type": "pending",
            }
        )
        for condition in ("raw", "whitened", "matched"):
            rows.append(
                {
                    "stage": "stage-b2", "environment": "hidden_velocity",
                    "condition": condition, "adapter": adapter, "seed": 0,
                    "final_performance": -0.5, "nan_count": 0, "inf_count": 0,
                    "divergence_flag": 0, "wall_seconds": 1.0,
                    "run_status": "valid", "source_summary": "synthetic",
                    "source_commit": "same", "source_config_hash": "cfg",
                    "reuse_type": "pending",
                }
            )
    rows.append(
        {
            "stage": "stage-b1", "environment": "tmaze", "condition": "oracle",
            "adapter": "identity", "seed": 0, "final_performance": 0.9,
            "nan_count": 0, "inf_count": 0, "divergence_flag": 0,
            "wall_seconds": 1.0, "run_status": "valid",
            "source_summary": "synthetic", "source_commit": "same",
            "source_config_hash": "cfg", "reuse_type": "pending",
        }
    )
    frame = pd.DataFrame(rows)
    audit = {"status": "valid", "valid": len(frame)}
    monkeypatch.setattr(
        reporting_module, "audit_stage_roots", lambda roots: (audit, frame)
    )
    output = tmp_path / "aggregate"
    aggregate_stage_roots({}, output)
    required = {
        "run_manifest.json", "run_summaries.csv", "aggregate_performance.csv",
        "paired_adapter_differences.csv", "runtime_summary.csv",
        "reuse_validation.csv", "failed_runs.csv", "experiment_summary.md",
        "integrity_audit.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert len(list((output / "figures").glob("*.png"))) == 4
