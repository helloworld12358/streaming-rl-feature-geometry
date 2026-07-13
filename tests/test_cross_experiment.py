import json
import subprocess
import sys

import pandas as pd
import pytest

from streaming_rl_feature_geometry.cross_experiment import (
    _read_many,
    load_cross_config,
    run_cross_one,
)


def tiny_cross_config(tmp_path):
    return {
        "profile": "cross_unit",
        "seeds": [0],
        "environments": {
            "tmaze": {
                "interactions": 160,
                "conditions": ["raw"],
                "bank": "mixed",
                "kwargs": {"corridor_length": 2},
            },
            "ringworld": {
                "interactions": 160,
                "conditions": ["raw"],
                "bank": "mixed",
                "kwargs": {"size": 8},
            },
            "two_loop": {
                "interactions": 160,
                "conditions": ["raw"],
                "bank": "mixed",
                "kwargs": {"loop_lengths": [6, 8]},
            },
            "hidden_velocity": {
                "interactions": 160,
                "conditions": ["raw"],
                "bank": "mixed",
                "kwargs": {"noise_std": 0.0},
                "control_alpha": 0.001,
            },
        },
        "horizons": [0.65, 0.9],
        "trace_dim": 16,
        "predictive_alpha": 0.005,
        "control_alpha": 0.005,
        "gamma": 0.95,
        "lambda": 0.8,
        "epsilon": 0.05,
        "transform_eps": 0.001,
        "transform_min_samples": 20,
        "cov_update_every": 20,
        "moment_beta": 0.005,
        "moment_learning_rate": 0.0005,
        "metrics_stride": 20,
        "analysis_burn_in": 20,
        "reservoir_size": 200,
        "moving_window": 20,
        "final_window": 20,
        "workers": 1,
        "output_dir": str(tmp_path),
    }


@pytest.mark.parametrize("environment", ("tmaze", "ringworld", "two_loop", "hidden_velocity"))
def test_common_runner_writes_isolated_complete_environment_runs(tmp_path, environment):
    config = tiny_cross_config(tmp_path)
    root = tmp_path / "runs"
    summary = run_cross_one((config, environment, "raw", 0, str(root)))
    run_dir = root / "runs" / environment / "raw" / "seed_000"
    required = {
        "config.json",
        "manifest.json",
        "summary.csv",
        "summary.json",
        "step_metrics.csv",
        "decision_metrics.csv",
        "prediction_metrics.csv",
        "representation_metrics.csv",
        "task_information_metrics.csv",
        "update_metrics.csv",
        "diagnostic_samples.npz",
        "predictive_definitions.json",
        "predictive_weights.npy",
        "stdout.log",
        "figures",
    }
    assert required <= {path.name for path in run_dir.iterdir()}
    assert summary["environment"] == environment and summary["run_status"] == "ok"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["environment"] == environment and manifest["exit_status"] == "ok"
    with pytest.raises(FileExistsError):
        run_cross_one((config, environment, "raw", 0, str(root)))


def test_cross_same_seed_reproduces_scientific_summary(tmp_path):
    config = tiny_cross_config(tmp_path)
    first = run_cross_one((config, "ringworld", "matched", 0, str(tmp_path / "a")))
    second = run_cross_one((config, "ringworld", "matched", 0, str(tmp_path / "b")))
    ignored = {"wall_seconds", "run_dir"}
    assert {k: v for k, v in first.items() if k not in ignored} == {
        k: v for k, v in second.items() if k not in ignored
    }


def test_cross_full_guard_requires_both_safeguards(tmp_path, monkeypatch):
    config = tiny_cross_config(tmp_path)
    config["profile"] = "cross_full"
    config["remote_full"] = True
    path = tmp_path / "full.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.delenv("RL_RUN_CONTEXT", raising=False)
    command = [
        sys.executable,
        "scripts/run_cross_experiment.py",
        "--config",
        str(path),
        "--run-name",
        "blocked",
        "--allow-full-run",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "both required" in result.stderr
    assert not (tmp_path / "blocked").exists()


def test_repository_cross_configs_load_and_preserve_three_pilot_seeds():
    smoke = load_cross_config("configs/cross_smoke.json")
    pilot = load_cross_config("configs/cross_pilot.json")
    assert set(smoke["environments"]) == set(pilot["environments"])
    assert pilot["seeds"] == [0, 1, 2]
    assert all("matched" in spec["conditions"] for spec in pilot["environments"].values())


def test_cross_aggregation_reader_skips_structurally_empty_tables(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("\n", encoding="utf-8")
    populated = tmp_path / "populated.csv"
    pd.DataFrame([{"value": 3.0}]).to_csv(populated, index=False)
    frame = _read_many([empty, populated])
    assert frame.to_dict(orient="records") == [{"value": 3.0}]
