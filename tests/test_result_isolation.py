import json

import pandas as pd
import pytest

from streaming_rl_feature_geometry.experiment import aggregate, run_one, validate_results


def tiny_config(output_dir):
    return {
        "profile": "unit",
        "seeds": [0],
        "conditions": ["raw"],
        "total_interactions": 160,
        "corridor_length": 2,
        "horizons": [0.6, 0.9],
        "gvf_bank": "mixed",
        "trace_dim": 12,
        "gvf_alpha": 0.005,
        "control_alpha": 0.005,
        "gamma": 0.98,
        "lambda": 0.8,
        "epsilon": 0.1,
        "transform_eps": 0.001,
        "transform_min_samples": 32,
        "cov_update_every": 20,
        "moment_beta": 0.005,
        "moment_learning_rate": 0.0005,
        "metrics_stride": 20,
        "analysis_burn_in": 20,
        "reservoir_size": 200,
        "moving_window_trials": 10,
        "final_window_trials": 10,
        "accuracy_threshold": 0.8,
        "workers": 1,
        "nonstationary": False,
        "output_dir": str(output_dir),
    }


def test_run_outputs_are_complete_and_existing_path_is_not_overwritten(tmp_path):
    root = tmp_path / "one-run"
    root.mkdir()
    config = tiny_config(tmp_path)
    task = (config, "raw", 0, str(root))
    run_one(task)
    with pytest.raises(FileExistsError):
        run_one(task)
    aggregate(root)
    report = validate_results(root, config)
    assert report["runs"] == 1
    run_dir = root / "runs" / "raw" / "seed_000"
    recorded = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert recorded["total_interactions"] == 160
    assert manifest["config_hash"] and manifest["exit_status"] == "ok"
    assert len(pd.read_csv(root / "aggregate_summary.csv")) == 1


def test_validation_allows_not_applicable_transform_metrics_for_baseline(tmp_path):
    root = tmp_path / "baseline-and-raw"
    root.mkdir()
    config = tiny_config(tmp_path)
    config["conditions"] = ["observation_only", "raw", "gaussian_moment"]
    for condition in config["conditions"]:
        run_one((config, condition, 0, str(root)))
    aggregate(root)

    summaries = pd.read_csv(root / "aggregate_summary.csv")
    baseline = summaries.query("condition == 'observation_only'").iloc[0]
    raw = summaries.query("condition == 'raw'").iloc[0]
    assert pd.isna(baseline["transform_condition_number"])
    assert pd.isna(raw["gaussian_parameter_change_mean"])
    assert validate_results(root, config)["runs"] == 3

    summaries.drop(columns=["active_fraction", "bounded_max_abs"]).to_csv(
        root / "aggregate_summary.csv", index=False
    )
    assert validate_results(root, config)["runs"] == 3
