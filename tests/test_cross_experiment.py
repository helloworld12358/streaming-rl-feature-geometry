import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from streaming_rl_feature_geometry.cross_experiment import (
    _controller_state,
    _read_many,
    load_cross_config,
    run_cross_one,
)
from streaming_rl_feature_geometry.cross_metrics import task_information_metrics


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
    decisions = pd.read_csv(run_dir / "decision_metrics.csv")
    assert set(decisions.columns) == {
        "t",
        "environment",
        "condition",
        "seed",
        "group",
        "position",
        "correct",
        "moving_accuracy",
        "reward",
    }
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
    compact = load_cross_config("configs/cross_pilot_compact.json")
    short = load_cross_config("configs/cross_pilot_short_horizon.json")
    remote_paths = [
        "configs/cross_full.json",
        "configs/cross_full_compact.json",
        "configs/cross_full_short_horizon.json",
        "configs/cross_full_nonstationary.json",
    ]
    remote = [load_cross_config(path) for path in remote_paths]
    assert set(smoke["environments"]) == set(pilot["environments"])
    assert pilot["seeds"] == [0, 1, 2]
    assert all("matched" in spec["conditions"] for spec in pilot["environments"].values())
    assert compact["seeds"] == short["seeds"] == [0, 1, 2]
    assert all(spec["bank"] == "compact" for spec in compact["environments"].values())
    assert set(short["environments"]) == {"tmaze", "ringworld"}
    assert short["horizons"] == [0.2, 0.4]
    assert all(config["remote_full"] and len(config["seeds"]) == 20 for config in remote)


def test_cross_nonstationary_change_is_e1_only_and_recorded(tmp_path):
    config = tiny_cross_config(tmp_path)
    config["environments"] = {
        "tmaze": {
            "interactions": 80,
            "conditions": ["raw"],
            "bank": "compact",
            "kwargs": {"corridor_length": 2},
            "nonstationary": True,
            "change_point": 40,
            "corridor_length_after": 4,
        }
    }
    config["analysis_burn_in"] = 4
    config["final_window"] = 2
    path = tmp_path / "nonstationary.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_cross_config(path)
    summary = run_cross_one((loaded, "tmaze", "raw", 0, str(tmp_path / "run")))
    assert summary["change_point"] == 40
    assert np.isfinite(summary["pre_change_accuracy"])
    assert np.isfinite(summary["final_post_change_accuracy"])

    config["environments"] = {
        "ringworld": {
            "interactions": 80,
            "conditions": ["raw"],
            "nonstationary": True,
            "change_point": 40,
            "corridor_length_after": 4,
        }
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="E1/T-maze only"):
        load_cross_config(path)


def test_non_oracle_controller_state_cannot_read_oracle_features():
    class DiagnosticOnlyOracle:
        @property
        def oracle_features(self):
            raise AssertionError("non-oracle agent attempted to read diagnostic oracle state")

    transformed = np.asarray([1.0, 2.0])
    assert _controller_state("observation_only", DiagnosticOnlyOracle(), transformed).size == 0
    assert (_controller_state("matched", DiagnosticOnlyOracle(), transformed) == transformed).all()


@pytest.mark.parametrize(
    ("key", "value"),
    (("seeds", [0, 0]), ("horizons", [0.9, 0.9]), ("horizons", [1.0])),
)
def test_cross_config_rejects_duplicate_or_noncausal_axes(tmp_path, key, value):
    config = tiny_cross_config(tmp_path)
    config[key] = value
    path = tmp_path / f"bad-{key}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cross_config(path)


def test_cross_aggregation_reader_skips_structurally_empty_tables(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("\n", encoding="utf-8")
    populated = tmp_path / "populated.csv"
    pd.DataFrame([{"value": 3.0}]).to_csv(populated, index=False)
    frame = _read_many([empty, populated])
    assert frame.to_dict(orient="records") == [{"value": 3.0}]


@pytest.mark.parametrize(
    ("environment", "latent_dim"),
    (("ringworld", 2), ("two_loop", 3), ("hidden_velocity", 2)),
)
def test_grouped_task_probes_return_finite_fallback_when_groups_are_insufficient(
    environment, latent_dim
):
    features = np.ones((20, 4))
    latents = np.zeros((20, latent_dim))
    groups = np.zeros(20, dtype=int)
    metrics, payload = task_information_metrics(
        environment, features, latents, groups, seed=3
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert payload == {}


def test_ringworld_neighborhood_metric_rewards_ordered_circular_features():
    angles = np.linspace(-np.pi, np.pi, 80, endpoint=False)
    features = np.column_stack((np.cos(angles), np.sin(angles)))
    latents = np.column_stack((np.arange(len(angles)), angles))
    groups = np.arange(len(angles))
    ordered, _ = task_information_metrics(
        "ringworld", features, latents, groups, seed=7
    )
    permutation = np.random.default_rng(9).permutation(len(features))
    shuffled, _ = task_information_metrics(
        "ringworld", features[permutation], latents, groups, seed=7
    )
    assert ordered["neighborhood_preservation"] > 0.8
    assert ordered["neighborhood_preservation"] > shuffled["neighborhood_preservation"]
