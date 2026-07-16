"""Config-driven cross-environment streaming experiment runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import sys
import time
from collections import deque
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .controller import SarsaLambda
from .alpha_tuning import (
    DEFAULT_SEED_SETS,
    candidate_alphas,
    load_selected_learning_rates,
    select_learning_rates,
    validate_seed_sets,
)
from .cross_env import ENVIRONMENT_IDS, make_environment
from .cross_metrics import (
    DiagnosticReservoir,
    decision_conditioned_metrics,
    task_information_metrics,
)
from .cross_reporting import make_cross_figures
from .experiment import config_hash, dependency_versions, git_value, write_json
from .predictive import CausalPredictiveBank
from .priors import MATCHED_PRIOR_FOR_ENV, TaskMatchedTransform
from .robust_stats import build_robust_summaries
from .transforms import FeatureTransform, rep_metrics


BASELINE_CONDITIONS = {"observation_only", "oracle"}
GENERIC_CONDITIONS = {
    "raw",
    "rms_raw",
    "standardized",
    "decorrelated",
    "whitened",
    "gaussian_moment",
    "unit_sphere",
    "sparse",
    "bounded",
}
CROSS_CONDITIONS = BASELINE_CONDITIONS | GENERIC_CONDITIONS | {"matched"}
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
HIDDEN_SUMMARY_DEFAULTS: dict[str, Any] = {
    "mean_position_cost": np.nan,
    "mean_velocity_cost": np.nan,
    "mean_action_cost": np.nan,
    "mean_control_effort": np.nan,
    "mean_total_cost": np.nan,
    "position_rmse": np.nan,
    "velocity_rmse": np.nan,
    "mean_abs_position": np.nan,
    "mean_abs_velocity": np.nan,
    "boundary_hit_count": np.nan,
    "boundary_hit_rate": np.nan,
    "final_window_position_cost": np.nan,
    "final_window_velocity_cost": np.nan,
    "final_window_action_cost": np.nan,
    "final_window_control_effort": np.nan,
    "final_window_total_cost": np.nan,
    "final_window_position_rmse": np.nan,
    "final_window_velocity_rmse": np.nan,
    "final_window_mean_abs_position": np.nan,
    "final_window_mean_abs_velocity": np.nan,
    "final_window_stabilization_rate": np.nan,
    "final_window_boundary_hit_rate": np.nan,
    "maximum_abs_position": np.nan,
    "maximum_abs_velocity": np.nan,
    "time_outside_stable_region": np.nan,
    "settling_time": np.nan,
    "settling_time_status": "not_applicable",
    "settling_time_applicable": False,
    "recovery_time_status": "not_applicable",
    "recovery_time_applicable": False,
    "recovery_success": np.nan,
    "disturbance_recovery_time": np.nan,
    "recovery_success_rate": np.nan,
    "post_disturbance_position_rmse": np.nan,
    "post_disturbance_velocity_rmse": np.nan,
    "post_disturbance_stabilization_rate": np.nan,
}
RUN_REQUIRED_ENTRIES = {
    "config.json",
    "manifest.json",
    "predictive_definitions.json",
    "step_metrics.csv",
    "decision_metrics.csv",
    "prediction_metrics.csv",
    "update_metrics.csv",
    "representation_metrics.csv",
    "task_information_metrics.csv",
    "decision_probe_by_position.csv",
    "summary.csv",
    "summary.json",
    "diagnostic_samples.npz",
    "predictive_weights.npy",
    "stdout.log",
    "figures",
}


def load_cross_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "profile",
        "seeds",
        "environments",
        "horizons",
        "trace_dim",
        "predictive_alpha",
        "control_alpha",
        "gamma",
        "lambda",
        "epsilon",
        "output_dir",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"cross config is missing {missing}")
    unknown_envs = set(config["environments"]) - set(ENVIRONMENT_IDS)
    if unknown_envs:
        raise ValueError(f"unknown environments: {sorted(unknown_envs)}")
    if not config["seeds"] or len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("seeds must be a non-empty unique list")
    if not config["horizons"] or any(
        not 0.0 <= float(value) < 1.0 for value in config["horizons"]
    ):
        raise ValueError("horizons must be a non-empty list in [0, 1)")
    if len(config["horizons"]) != len(set(map(float, config["horizons"]))):
        raise ValueError("horizons must be unique")
    if int(config["trace_dim"]) < 1:
        raise ValueError("trace_dim must be positive")
    for key in ("predictive_alpha", "control_alpha"):
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= float(config["gamma"]) < 1:
        raise ValueError("gamma must be in [0, 1)")
    if not 0 <= float(config["lambda"]) <= 1 or not 0 <= float(config["epsilon"]) <= 1:
        raise ValueError("lambda and epsilon must be in [0, 1]")
    for env_id, spec in config["environments"].items():
        if int(spec["interactions"]) < 1:
            raise ValueError(f"{env_id} interactions must be positive")
        if not spec["conditions"] or len(spec["conditions"]) != len(set(spec["conditions"])):
            raise ValueError(f"{env_id} conditions must be a non-empty unique list")
        unknown_conditions = set(spec["conditions"]) - CROSS_CONDITIONS
        if unknown_conditions:
            raise ValueError(f"unknown {env_id} conditions: {sorted(unknown_conditions)}")
        if spec.get("bank", "mixed") not in {"compact", "mixed"}:
            raise ValueError("bank must be compact or mixed")
        if spec.get("nonstationary", False):
            if env_id != "tmaze":
                raise ValueError("the preregistered non-stationary change is E1/T-maze only")
            for key in ("change_point", "corridor_length_after"):
                if key not in spec:
                    raise ValueError(f"non-stationary tmaze requires {key}")
            if not 0 < int(spec["change_point"]) < int(spec["interactions"]):
                raise ValueError("change_point must be inside the E1 run")
            if int(spec["corridor_length_after"]) < 1:
                raise ValueError("corridor_length_after must be positive")
    alpha_mode = str(config.get("controller_alpha_mode", "fixed"))
    if alpha_mode not in {"fixed", "norm_scaled"}:
        raise ValueError("controller_alpha_mode must be fixed or norm_scaled")
    stage = str(config.get("experiment_stage", "fixed"))
    if stage not in {"fixed", "lr_tune", "lr_eval", "norm_scaled"}:
        raise ValueError("experiment_stage must be fixed, lr_tune, lr_eval, or norm_scaled")
    if stage == "norm_scaled" and alpha_mode != "norm_scaled":
        raise ValueError("norm_scaled stage requires controller_alpha_mode=norm_scaled")
    if stage != "norm_scaled" and alpha_mode == "norm_scaled":
        raise ValueError("norm_scaled controller mode requires experiment_stage=norm_scaled")
    seed_sets = config.setdefault("seed_sets", {key: list(value) for key, value in DEFAULT_SEED_SETS.items()})
    validate_seed_sets(seed_sets)
    expected_seed_key = {"lr_tune": "tuning", "lr_eval": "evaluation"}.get(stage)
    if expected_seed_key is not None and set(map(int, config["seeds"])) != set(
        map(int, seed_sets[expected_seed_key])
    ):
        raise ValueError(f"{stage} seeds must exactly match seed_sets.{expected_seed_key}")
    if stage == "lr_tune":
        tuning = config.setdefault("alpha_tuning", {})
        multipliers = tuning.setdefault("multipliers", [0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
        candidate_alphas(float(config["control_alpha"]), list(map(float, multipliers)))
    defaults = {
        "transform_eps": 1e-3,
        "transform_min_samples": 64,
        "cov_update_every": 50,
        "moment_beta": 0.005,
        "moment_learning_rate": 0.0005,
        "metrics_stride": 25,
        "analysis_burn_in": 500,
        "reservoir_size": 3000,
        "moving_window": 100,
        "final_window": 300,
        "workers": 1,
        "experiment_stage": "fixed",
        "controller_alpha_mode": "fixed",
        "controller_alpha_norm": {
            "ema_beta": 0.01,
            "epsilon": 1e-6,
            "alpha_min": 1e-6,
            "alpha_max_multiplier": 1.0,
        },
        "settling_consecutive_steps": 20,
        "recovery_consecutive_steps": 8,
        "robust_bootstrap_samples": 2000,
        "robust_bootstrap_seed": 1729,
        "catastrophic_failure": {
            "global": {"max_control_update_norm_above": 100.0},
            "environments": {
                "hidden_velocity": {
                    "final_window_reward_below": -6.0,
                    "final_window_stabilization_rate_below": 0.005,
                    "final_window_position_rmse_above": 2.4,
                },
                "hidden_velocity_informative": {
                    "final_window_reward_below": -7.0,
                    "final_window_stabilization_rate_below": 0.005,
                    "final_window_position_rmse_above": 2.4,
                },
            },
        },
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    return config


def _manifest_base(
    config: dict[str, Any], environment: str | None = None, condition: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "exact_command": [Path(sys.executable).name, *sys.argv],
        "profile": config["profile"],
        "environment": environment,
        "representation": condition,
        "condition": condition,
        "seed": seed,
        "python_version": platform.python_version(),
        "dependency_versions": dependency_versions(),
        "operating_system": platform.platform(),
        "hostname": socket.gethostname(),
        "config_hash": config_hash(config),
        "experiment_stage": config.get("experiment_stage", "fixed"),
        "controller_alpha_mode": config.get("controller_alpha_mode", "fixed"),
        "catastrophic_failure": config.get("catastrophic_failure", {}),
        "seed_sets": config.get("seed_sets", {}),
        "selected_learning_rates_sha256": config.get("selected_learning_rates_sha256"),
        "cpu_count": os.cpu_count(),
        "cpu_model": os.environ.get("RL_REMOTE_CPU_MODEL", platform.processor() or "unknown"),
        "gpu_detection": os.environ.get("RL_REMOTE_GPU_DETECTION", "not-recorded"),
        "gpu_backend_used": "none",
        "parallelism": "cpu-process",
    }


def _controller_features(observation: np.ndarray, state: np.ndarray) -> np.ndarray:
    return np.concatenate((observation, state, np.ones(1, dtype=np.float64)))


def _controller_state(condition: str, environment: Any, transformed: np.ndarray) -> np.ndarray:
    """Build agent state; only the explicit oracle branch may read oracle features."""

    if condition == "observation_only":
        return np.empty(0, dtype=np.float64)
    if condition == "oracle":
        return environment.oracle_features.copy()
    return transformed.copy()


def _diagnostic_features(
    condition: str,
    observation: np.ndarray,
    oracle: np.ndarray,
    predictive: np.ndarray,
) -> np.ndarray:
    if condition == "observation_only":
        return observation.copy()
    if condition == "oracle":
        return oracle.copy()
    return predictive.copy()


def _make_transform(condition: str, environment: str, d: int, config: dict[str, Any]):
    if condition in BASELINE_CONDITIONS:
        return None
    if condition == "matched":
        return TaskMatchedTransform(
            MATCHED_PRIOR_FOR_ENV[environment],
            d,
            eps=config["transform_eps"],
        )
    return FeatureTransform(
        condition,
        d,
        eps=config["transform_eps"],
        update_every=config["cov_update_every"],
        min_samples=config["transform_min_samples"],
        moment_beta=config["moment_beta"],
        moment_learning_rate=config["moment_learning_rate"],
    )


def _transform(transform: Any, values: np.ndarray) -> np.ndarray:
    return values.copy() if transform is None else transform.transform(values)


def _first_stable_run(stabilized: np.ndarray, consecutive: int, start: int = 0) -> float:
    run = 0
    for index in range(max(0, start), len(stabilized)):
        run = run + 1 if bool(stabilized[index]) else 0
        if run >= consecutive:
            return float(index - consecutive + 1 - start)
    return float("nan")


def _hidden_velocity_summary(
    diagnostics: list[dict[str, Any]], config: dict[str, Any], environment: str
) -> dict[str, Any]:
    """Summarize authoritative environment diagnostics without reward reimplementation."""

    frame = pd.DataFrame(diagnostics)
    if frame.empty:
        return {}
    final = frame.tail(int(config["final_window"]))
    result: dict[str, Any] = {
        "mean_position_cost": float(frame["position_cost"].mean()),
        "mean_velocity_cost": float(frame["velocity_cost"].mean()),
        "mean_action_cost": float(frame["action_cost"].mean()),
        "mean_control_effort": float(frame["control_effort"].mean()),
        "mean_total_cost": float(frame["total_cost"].mean()),
        "position_rmse": float(np.sqrt(frame["position_squared"].mean())),
        "velocity_rmse": float(np.sqrt(frame["velocity_squared"].mean())),
        "mean_abs_position": float(frame["abs_position"].mean()),
        "mean_abs_velocity": float(frame["abs_velocity"].mean()),
        "stabilization_rate": float(frame["stabilized"].mean()),
        "boundary_hit_count": int(frame["boundary_hit"].sum()),
        "boundary_hit_rate": float(frame["boundary_hit"].mean()),
        "final_window_position_cost": float(final["position_cost"].mean()),
        "final_window_velocity_cost": float(final["velocity_cost"].mean()),
        "final_window_action_cost": float(final["action_cost"].mean()),
        "final_window_control_effort": float(final["control_effort"].mean()),
        "final_window_total_cost": float(final["total_cost"].mean()),
        "final_window_position_rmse": float(np.sqrt(final["position_squared"].mean())),
        "final_window_velocity_rmse": float(np.sqrt(final["velocity_squared"].mean())),
        "final_window_mean_abs_position": float(final["abs_position"].mean()),
        "final_window_mean_abs_velocity": float(final["abs_velocity"].mean()),
        "final_window_stabilization_rate": float(final["stabilized"].mean()),
        "final_window_boundary_hit_rate": float(final["boundary_hit"].mean()),
        "maximum_abs_position": float(frame["abs_position"].max()),
        "maximum_abs_velocity": float(frame["abs_velocity"].max()),
        "time_outside_stable_region": int((~frame["stabilized"].astype(bool)).sum()),
    }
    stable = frame["stabilized"].to_numpy(dtype=bool)
    settling = _first_stable_run(stable, int(config.get("settling_consecutive_steps", 20)))
    result.update(
        settling_time=settling,
        settling_time_status="ok" if np.isfinite(settling) else "not_reached",
        settling_time_applicable=True,
    )
    disturbance_indices = np.flatnonzero(frame["disturbance_active"].to_numpy(dtype=bool))
    if environment != "hidden_velocity_informative":
        result.update(
            recovery_time=np.nan,
            recovery_success=np.nan,
            recovery_time_status="not_applicable",
            recovery_time_applicable=False,
            disturbance_recovery_time=np.nan,
            recovery_success_rate=np.nan,
            post_disturbance_position_rmse=np.nan,
            post_disturbance_velocity_rmse=np.nan,
            post_disturbance_stabilization_rate=np.nan,
        )
        return result
    recovery_times: list[float] = []
    consecutive = int(config.get("recovery_consecutive_steps", 8))
    for disturbance_index in disturbance_indices:
        next_indices = disturbance_indices[disturbance_indices > disturbance_index]
        end = int(next_indices[0]) if len(next_indices) else len(frame)
        recovery = _first_stable_run(stable[:end], consecutive, int(disturbance_index))
        recovery_times.append(recovery)
    successful = np.asarray([value for value in recovery_times if np.isfinite(value)], dtype=float)
    post = frame[frame["post_disturbance"].astype(bool)]
    success_rate = float(len(successful) / len(recovery_times)) if recovery_times else np.nan
    mean_recovery = float(np.mean(successful)) if len(successful) else np.nan
    result.update(
        recovery_time=mean_recovery,
        recovery_success=success_rate,
        recovery_time_status=("ok" if len(successful) else "not_reached") if recovery_times else "no_disturbance",
        recovery_time_applicable=True,
        disturbance_recovery_time=mean_recovery,
        recovery_success_rate=success_rate,
        post_disturbance_position_rmse=(
            float(np.sqrt(post["position_squared"].mean())) if len(post) else np.nan
        ),
        post_disturbance_velocity_rmse=(
            float(np.sqrt(post["velocity_squared"].mean())) if len(post) else np.nan
        ),
        post_disturbance_stabilization_rate=(
            float(post["stabilized"].mean()) if len(post) else np.nan
        ),
    )
    return result


def _execute_cross_run(
    config: dict[str, Any], environment: str, condition: str, seed: int, run_dir: Path,
    started: float, task_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_options = task_options or {}
    env_spec = config["environments"][environment]
    env = make_environment(environment, seed=seed, **env_spec.get("kwargs", {}))
    bank = CausalPredictiveBank(
        environment,
        env.obs_dim,
        env.event_names,
        seed=seed + 17,
        alpha=env_spec.get("predictive_alpha", config["predictive_alpha"]),
        continuations=tuple(config["horizons"]),
        trace_dim=config["trace_dim"],
        bank=env_spec.get("bank", "mixed"),
    )
    write_json(run_dir / "predictive_definitions.json", bank.definition_records())
    observation = env.observation
    raw = bank.features(observation)
    transform = _make_transform(condition, environment, bank.d, config)
    transformed = _transform(transform, raw) if condition not in BASELINE_CONDITIONS else np.empty(0)
    state = _controller_state(condition, env, transformed)
    features = _controller_features(observation, state)
    base_alpha = float(env_spec.get("control_alpha", config["control_alpha"]))
    controller_alpha = float(task_options.get("controller_alpha", base_alpha))
    alpha_mode = str(config.get("controller_alpha_mode", "fixed"))
    norm = config.get("controller_alpha_norm", {})
    controller = SarsaLambda(
        env.n_actions,
        len(features),
        seed=seed + 31,
        alpha=controller_alpha,
        gamma=env_spec.get("gamma", config["gamma"]),
        lam=env_spec.get("lambda", config["lambda"]),
        epsilon=env_spec.get("epsilon", config["epsilon"]),
        alpha_mode=alpha_mode,
        norm_ema_beta=float(norm.get("ema_beta", 0.01)),
        norm_epsilon=float(norm.get("epsilon", 1e-6)),
        alpha_min=float(norm.get("alpha_min", 1e-6)),
        alpha_max=controller_alpha * float(norm.get("alpha_max_multiplier", 1.0)),
    )
    action = controller.act(features)
    diagnostic_dim = len(_diagnostic_features(condition, observation, env.oracle_features, transformed))
    latent_dim = len(env.latent_state)
    reservoir = DiagnosticReservoir(config["reservoir_size"], seed + 101)
    step_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    recent_rewards: deque[float] = deque(maxlen=config["moving_window"])
    recent_correct: deque[float] = deque(maxlen=config["moving_window"])
    all_rewards: list[float] = []
    all_correct: list[float] = []
    correct_times: list[int] = []
    all_stabilized: list[float] = []
    hidden_diagnostics: list[dict[str, Any]] = []
    control_deltas: list[float] = []
    control_updates: list[float] = []
    predictive_updates: list[float] = []
    predictive_mse: list[float] = []
    cumulative_reward = 0.0
    threshold_time = -1
    actual_change_point: int | None = None
    hidden_environment = environment in {"hidden_velocity", "hidden_velocity_informative"}
    target_threshold = float(env_spec.get("threshold", -0.3 if hidden_environment else 0.8))

    for interaction in range(int(env_spec["interactions"])):
        if env_spec.get("nonstationary", False) and interaction == int(
            env_spec["change_point"]
        ):
            env.set_corridor_length(int(env_spec["corridor_length_after"]))
            actual_change_point = interaction
        current_oracle = env.oracle_features.copy()
        analysis_vector = _diagnostic_features(
            condition, observation, current_oracle, transformed
        )
        next_observation, reward, info = env.step(action)
        prediction_update = bank.update(next_observation, info.events)
        next_raw = bank.features(next_observation)
        next_transformed = (
            _transform(transform, next_raw)
            if condition not in BASELINE_CONDITIONS
            else np.empty(0)
        )
        next_state = _controller_state(condition, env, next_transformed)
        next_features = _controller_features(next_observation, next_state)
        next_action = controller.act(next_features)
        control_delta, update_norm, parameter_norm = controller.update(
            features, action, reward, next_features, next_action
        )
        if not np.isfinite(next_features).all():
            raise FloatingPointError("non-finite cross-environment controller input")

        if interaction >= config["analysis_burn_in"]:
            reservoir.add(
                analysis_vector,
                np.asarray(info.latent),
                info.group,
                position=info.position,
                decision=info.decision,
                phase=info.phase,
            )
        cumulative_reward += reward
        all_rewards.append(float(reward))
        recent_rewards.append(float(reward))
        control_deltas.append(control_delta)
        control_updates.append(update_norm)
        predictive_updates.append(prediction_update.update_norm)
        predictive_mse.append(prediction_update.mse)
        if info.correct is not None:
            value = float(info.correct)
            all_correct.append(value)
            correct_times.append(interaction)
            recent_correct.append(value)
            decision_row: dict[str, Any] = {
                    "t": interaction,
                    "environment": environment,
                    "condition": condition,
                    "seed": seed,
                    "group": info.group,
                    "position": info.position,
                    "correct": value,
                    "moving_accuracy": float(np.mean(recent_correct)),
                    "reward": reward,
                }
            if "candidate_alpha" in task_options:
                decision_row["candidate_alpha"] = task_options["candidate_alpha"]
            decision_rows.append(decision_row)
        if info.stabilized is not None:
            all_stabilized.append(float(info.stabilized))
        if hidden_environment:
            diagnostic = dict(info.diagnostics)
            if not diagnostic:
                raise AssertionError("hidden-velocity step omitted authoritative diagnostics")
            if not np.isclose(float(diagnostic["reward"]), -float(diagnostic["total_cost"])):
                raise AssertionError("hidden-velocity reward must equal negative total cost")
            if not np.isclose(reward, float(diagnostic["reward"])):
                raise AssertionError("logged hidden-velocity reward differs from environment reward")
            hidden_diagnostics.append(diagnostic)

        moving_reward = float(np.mean(recent_rewards))
        moving_performance = (
            float(np.mean(recent_correct)) if recent_correct else moving_reward
        )
        if threshold_time < 0 and len(recent_rewards) == recent_rewards.maxlen:
            if moving_performance >= target_threshold:
                threshold_time = interaction
        record_diagnostics = interaction % config["metrics_stride"] == 0 or info.decision
        if hidden_environment or record_diagnostics:
            step_row: dict[str, Any] = {
                "t": interaction,
                "environment": environment,
                "condition": condition,
                "seed": seed,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "moving_reward": moving_reward,
                "moving_performance": moving_performance,
                "stabilized": float(info.stabilized) if info.stabilized is not None else np.nan,
                "post_change": float(
                    actual_change_point is not None and interaction >= actual_change_point
                ),
            }
            if "candidate_alpha" in task_options:
                step_row["candidate_alpha"] = task_options["candidate_alpha"]
            if hidden_environment:
                step_row.update(info.diagnostics)
            step_rows.append(step_row)
        if record_diagnostics:
            prediction_row: dict[str, Any] = {
                "t": interaction,
                "environment": environment,
                "condition": condition,
                "seed": seed,
                "predictive_td_mse": prediction_update.mse,
                "predictive_update_norm": prediction_update.update_norm,
                "predictive_parameter_norm": prediction_update.parameter_norm,
            }
            if "candidate_alpha" in task_options:
                prediction_row["candidate_alpha"] = task_options["candidate_alpha"]
            for index, name in enumerate(bank.feature_names):
                prediction_row[f"td_squared__{name}"] = prediction_update.deltas[index] ** 2
                prediction_row[f"cumulant__{name}"] = prediction_update.cumulants[index]
            prediction_rows.append(prediction_row)
            update_rows.append(
                {
                    "t": interaction,
                    "environment": environment,
                    "condition": condition,
                    "seed": seed,
                    "control_td_error": control_delta,
                    "control_update_norm": update_norm,
                    "controller_parameter_norm": parameter_norm,
                    "predictive_update_norm": prediction_update.update_norm,
                    "effective_controller_alpha": controller.last_effective_alpha,
                    "feature_squared_norm": float(features @ features),
                }
            )
            if "candidate_alpha" in task_options:
                update_rows[-1]["candidate_alpha"] = task_options["candidate_alpha"]

        observation, raw, transformed = next_observation, next_raw, next_transformed
        features, action = next_features, next_action

    sample_features, sample_latents, sample_groups = reservoir.arrays(diagnostic_dim, latent_dim)
    sample_positions, sample_decisions, sample_phases = reservoir.metadata_arrays()
    representation = rep_metrics(sample_features)
    task_metrics, probe_payload = task_information_metrics(
        environment, sample_features, sample_latents, sample_groups, seed + 700
    )
    decision_metrics, by_position_rows = decision_conditioned_metrics(
        environment,
        sample_features,
        sample_latents,
        sample_groups,
        sample_positions,
        sample_decisions,
        sample_phases,
        seed + 1700,
    )
    representation.update(
        environment=environment,
        condition=condition,
        seed=seed,
        experiment_stage=config.get("experiment_stage", "fixed"),
    )
    if "candidate_alpha" in task_options:
        representation["candidate_alpha"] = task_options["candidate_alpha"]
    if transform is not None:
        representation.update(transform.state_metrics())
    task_row = dict(
        task_metrics,
        **decision_metrics,
        environment=environment,
        condition=condition,
        seed=seed,
        experiment_stage=config.get("experiment_stage", "fixed"),
    )
    if "candidate_alpha" in task_options:
        task_row["candidate_alpha"] = task_options["candidate_alpha"]
    final_rewards = all_rewards[-config["final_window"] :]
    final_correct = all_correct[-config["final_window"] :]
    final_reward = float(np.mean(final_rewards))
    final_accuracy = float(np.mean(final_correct)) if final_correct else 0.0
    final_performance = final_reward if hidden_environment else final_accuracy
    pre_change_accuracy = -1.0
    final_post_change_accuracy = -1.0
    recovery_time = np.nan
    if env_spec.get("nonstationary", False):
        if actual_change_point is None:
            raise RuntimeError("configured non-stationary E1 change was not applied")
        timed = np.asarray(correct_times, dtype=int)
        correct = np.asarray(all_correct, dtype=float)
        pre = correct[timed < actual_change_point]
        post_mask = timed >= actual_change_point
        post = correct[post_mask]
        post_times = timed[post_mask]
        window = int(config["final_window"])
        pre_change_accuracy = float(np.mean(pre[-window:])) if len(pre) else 0.0
        final_post_change_accuracy = float(np.mean(post[-window:])) if len(post) else 0.0
        target = pre_change_accuracy * 0.95
        for index in range(len(post)):
            current = post[index : index + window]
            if len(current) == window and float(np.mean(current)) >= target:
                recovery_time = int(post_times[index] - actual_change_point)
                break
    summary: dict[str, Any] = {
        "environment": environment,
        "condition": condition,
        "representation": condition,
        "matched_prior": MATCHED_PRIOR_FOR_ENV[environment] if condition == "matched" else "none",
        "bank": env_spec.get("bank", "mixed"),
        "horizon": ",".join(map(str, config["horizons"])),
        "seed": seed,
        "interactions": int(env_spec["interactions"]),
        "n_decisions": len(all_correct),
        "overall_accuracy": float(np.mean(all_correct)) if all_correct else 0.0,
        "final_window_accuracy": final_accuracy,
        "cumulative_reward": cumulative_reward,
        "mean_reward": float(np.mean(all_rewards)),
        "final_window_reward": final_reward,
        "final_performance": final_performance,
        "time_to_threshold": threshold_time,
        "change_point": actual_change_point if actual_change_point is not None else -1,
        "pre_change_accuracy": pre_change_accuracy,
        "final_post_change_accuracy": final_post_change_accuracy,
        "recovery_time": recovery_time,
        "stabilization_rate": float(np.mean(all_stabilized)) if all_stabilized else 0.0,
        "final_window_stabilization": float(
            np.mean(all_stabilized[-config["final_window"] :])
        ) if all_stabilized else 0.0,
        "control_cost": float(-np.mean(all_rewards)) if environment == "hidden_velocity" else 0.0,
        "predictive_td_mse": float(np.mean(predictive_mse)),
        "control_td_variance": float(np.var(control_deltas)),
        "mean_control_update_norm": float(np.mean(control_updates)),
        "max_control_update_norm": float(np.max(control_updates)),
        "mean_predictive_update_norm": float(np.mean(predictive_updates)),
        "final_parameter_norm": float(np.linalg.norm(controller.w)),
        "nan_count": 0,
        "inf_count": 0,
        "divergence_flag": 0,
        "wall_seconds": time.time() - started,
        "run_dir": str(run_dir),
        "run_status": "ok",
        "experiment_stage": config.get("experiment_stage", "fixed"),
        "candidate_alpha": task_options.get("candidate_alpha", np.nan),
        "base_alpha_multiplier": task_options.get("base_alpha_multiplier", np.nan),
        **task_metrics,
        **decision_metrics,
        **controller.alpha_metrics(),
        **HIDDEN_SUMMARY_DEFAULTS,
    }
    if hidden_environment:
        summary.update(_hidden_velocity_summary(hidden_diagnostics, config, environment))
        summary["control_cost"] = summary["mean_total_cost"]

    pd.DataFrame(step_rows).to_csv(run_dir / "step_metrics.csv", index=False)
    decision_columns = [
        "t",
        "environment",
        "condition",
        "seed",
        "group",
        "position",
        "correct",
        "moving_accuracy",
        "reward",
    ]
    if "candidate_alpha" in task_options:
        decision_columns.append("candidate_alpha")
    pd.DataFrame(decision_rows, columns=decision_columns).to_csv(
        run_dir / "decision_metrics.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(run_dir / "prediction_metrics.csv", index=False)
    pd.DataFrame(update_rows).to_csv(run_dir / "update_metrics.csv", index=False)
    pd.DataFrame([representation]).to_csv(run_dir / "representation_metrics.csv", index=False)
    pd.DataFrame([task_row]).to_csv(run_dir / "task_information_metrics.csv", index=False)
    by_position = pd.DataFrame(by_position_rows)
    if len(by_position):
        by_position.insert(0, "seed", seed)
        if "candidate_alpha" in task_options:
            by_position.insert(0, "candidate_alpha", task_options["candidate_alpha"])
        by_position.insert(0, "condition", condition)
        by_position.insert(0, "environment", environment)
    by_position.to_csv(run_dir / "decision_probe_by_position.csv", index=False)
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    write_json(run_dir / "summary.json", summary)
    np.savez_compressed(
        run_dir / "diagnostic_samples.npz",
        features=sample_features,
        latents=sample_latents,
        groups=sample_groups,
        positions=sample_positions,
        decisions=sample_decisions,
        phases=sample_phases,
        **probe_payload,
    )
    np.save(run_dir / "predictive_weights.npy", bank.w)
    (run_dir / "stdout.log").write_text(
        f"completed environment={environment} condition={condition} seed={seed}\n",
        encoding="utf-8",
    )
    return summary


def _task_parts(task: tuple[Any, ...]) -> tuple[dict[str, Any], str, str, int, str, dict[str, Any]]:
    if len(task) not in {5, 6}:
        raise ValueError("cross task must have five fields plus optional task options")
    config, environment, condition, seed, root_string = task[:5]
    options = dict(task[5]) if len(task) == 6 else {}
    return config, str(environment), str(condition), int(seed), str(root_string), options


def _alpha_path(alpha: float) -> str:
    return f"alpha_{alpha:.12g}".replace("+", "").replace(".", "p")


def _task_run_dir(task: tuple[Any, ...]) -> Path:
    _, environment, condition, seed, root_string, options = _task_parts(task)
    root = Path(root_string)
    base = root / "runs" / environment / condition
    if "candidate_alpha" in options:
        base /= _alpha_path(float(options["candidate_alpha"]))
    return base / f"seed_{seed:03d}"


def _completed_run(run_dir: Path) -> bool:
    if not run_dir.is_dir() or not RUN_REQUIRED_ENTRIES <= {path.name for path in run_dir.iterdir()}:
        return False
    if not (run_dir / "figures").is_dir() or any(
        not (run_dir / name).is_file() for name in RUN_REQUIRED_ENTRIES - {"figures"}
    ):
        return False
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = pd.read_csv(run_dir / "summary.csv")
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    return (
        manifest.get("exit_status") == "ok"
        and len(summary) == 1
        and summary.iloc[0].get("run_status") == "ok"
    )


def _archive_attempt(run_dir: Path, root: Path) -> Path:
    relative = run_dir.relative_to(root / "runs")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = root / "failed_attempts" / relative / timestamp
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(destination))
    return destination


def run_cross_one(task: tuple[Any, ...]) -> dict[str, Any]:
    config, environment, condition, seed, root_string, options = _task_parts(task)
    root = Path(root_string)
    run_dir = _task_run_dir(task)
    if run_dir.exists():
        if _completed_run(run_dir) and (options.get("resume") or options.get("retry_failed")):
            return pd.read_csv(run_dir / "summary.csv").iloc[0].to_dict()
        manifest_path = run_dir / "manifest.json"
        failed = False
        if manifest_path.exists():
            try:
                failed = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "exit_status"
                ) == "failed"
            except (OSError, ValueError):
                failed = False
        permitted = bool(options.get("retry_failed")) if failed else bool(options.get("resume"))
        if not permitted:
            action = "--retry-failed" if failed else "--resume"
            raise FileExistsError(f"{run_dir} already exists; use {action} to preserve and rerun it")
        _archive_attempt(run_dir, root)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    started = time.time()
    manifest = _manifest_base(config, environment, condition, seed)
    env_spec = config["environments"][environment]
    manifest.update(
        start_time=datetime.now(timezone.utc).isoformat(),
        exit_status="running",
        result_path=str(run_dir.resolve()),
        bank=env_spec.get("bank", "mixed"),
        horizon=config["horizons"],
        interaction_budget=env_spec["interactions"],
        candidate_alpha=options.get("candidate_alpha"),
        base_alpha_multiplier=options.get("base_alpha_multiplier"),
        controller_alpha=options.get(
            "controller_alpha", env_spec.get("control_alpha", config["control_alpha"])
        ),
        environment_kwargs=env_spec.get("kwargs", {}),
        final_window=config["final_window"],
        settling_consecutive_steps=config.get("settling_consecutive_steps", 20),
        recovery_consecutive_steps=config.get("recovery_consecutive_steps", 8),
        controller_alpha_norm=config.get("controller_alpha_norm", {}),
        alpha_tuning=config.get("alpha_tuning", {}),
    )
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "config.json", config)
    try:
        summary = _execute_cross_run(
            config, environment, condition, seed, run_dir, started, options
        )
    except Exception as error:
        manifest.update(
            end_time=datetime.now(timezone.utc).isoformat(),
            exit_status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "stdout.log").write_text(
            f"failed environment={environment} condition={condition} seed={seed}\n"
            f"{type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
        raise
    manifest.update(
        end_time=datetime.now(timezone.utc).isoformat(),
        exit_status="ok",
        wall_seconds=summary["wall_seconds"],
    )
    write_json(run_dir / "manifest.json", manifest)
    return summary


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size <= 1:
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_cross(
    root: str | Path, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    if config is None:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    summaries = _read_many(sorted((root / "runs").rglob("summary.csv")))
    representation = _read_many(
        sorted((root / "runs").rglob("representation_metrics.csv"))
    )
    task = _read_many(sorted((root / "runs").rglob("task_information_metrics.csv")))
    steps = _read_many(sorted((root / "runs").rglob("step_metrics.csv")))
    decisions = _read_many(sorted((root / "runs").rglob("decision_metrics.csv")))
    updates = _read_many(sorted((root / "runs").rglob("update_metrics.csv")))
    decision_by_position = _read_many(
        sorted((root / "runs").rglob("decision_probe_by_position.csv"))
    )
    robust, failures, summaries = build_robust_summaries(summaries, config)
    summaries.to_csv(root / "aggregate_summary.csv", index=False)
    representation.to_csv(root / "aggregate_representation.csv", index=False)
    task.to_csv(root / "aggregate_task_information.csv", index=False)
    steps.to_csv(root / "aggregate_steps.csv", index=False)
    decisions.to_csv(root / "aggregate_decisions.csv", index=False)
    updates.to_csv(root / "aggregate_updates.csv", index=False)
    decision_by_position.to_csv(root / "aggregate_decision_probe_by_position.csv", index=False)
    robust.to_csv(root / "robust_condition_summary.csv", index=False)
    failures.to_csv(root / "catastrophic_failure_summary.csv", index=False)
    numeric = [
        column for column in summaries.select_dtypes(include=[np.number]).columns if column != "seed"
    ]
    group_columns = ["environment", "condition"]
    if "candidate_alpha" in summaries and not summaries["candidate_alpha"].isna().all():
        group_columns.append("candidate_alpha")
    grouped = summaries.groupby(group_columns, sort=False, dropna=False)[numeric].agg(["mean", "sem"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    grouped.copy().reset_index().to_csv(root / "condition_summary.csv", index=False)
    if config.get("experiment_stage") == "lr_tune":
        selected = select_learning_rates(summaries, config)
        selected.to_csv(root / "selected_learning_rates.csv", index=False)
    make_cross_figures(root, summaries, representation, task, steps, decisions, updates)
    return summaries, representation


def expected_cross_run_count(config: dict[str, Any]) -> int:
    multiplier = len(config.get("alpha_tuning", {}).get("multipliers", []))
    if config.get("experiment_stage") != "lr_tune":
        multiplier = 1
    return sum(
        len(spec["conditions"]) * len(config["seeds"]) * multiplier
        for spec in config["environments"].values()
    )


def validate_cross_results(root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    summaries = pd.read_csv(root / "aggregate_summary.csv")
    expected = expected_cross_run_count(config)
    if len(summaries) != expected:
        raise AssertionError(f"expected {expected} cross runs, found {len(summaries)}")
    identity = ["environment", "condition", "seed"]
    if config.get("experiment_stage") == "lr_tune":
        identity.append("candidate_alpha")
    if summaries.duplicated(identity).any():
        raise AssertionError(f"duplicate {'/'.join(identity)} summaries")
    required_numeric = [
        "interactions",
        "cumulative_reward",
        "final_performance",
        "predictive_td_mse",
        "control_td_variance",
        "mean_control_update_norm",
        "final_parameter_norm",
        "nan_count",
        "inf_count",
        "divergence_flag",
    ]
    if not np.isfinite(summaries[required_numeric].to_numpy()).all():
        raise AssertionError("non-finite required cross-environment summary metrics")
    if not (summaries["run_status"] == "ok").all():
        raise AssertionError("at least one cross-environment run failed")
    for run_dir in sorted(path for path in (root / "runs").rglob("seed_*") if path.is_dir()):
        missing = sorted(RUN_REQUIRED_ENTRIES - {path.name for path in run_dir.iterdir()})
        if missing:
            raise AssertionError(f"{run_dir} is missing {missing}")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["exit_status"] != "ok":
            raise AssertionError(f"failed manifest {run_dir}")
    return {"result_dir": str(root.resolve()), "runs": len(summaries), "expected_runs": expected}


def smoke_cross_assertions(root: str | Path, config: dict[str, Any]) -> None:
    validate_cross_results(root, config)
    summaries = pd.read_csv(Path(root) / "aggregate_summary.csv")
    for environment in config["environments"]:
        frame = summaries[summaries["environment"] == environment]
        spec = config["environments"][environment]
        oracle_metric = str(spec.get("oracle_metric", "final_performance"))
        if oracle_metric not in frame.columns:
            raise AssertionError(f"unknown oracle sanity metric {oracle_metric!r}")
        oracle = frame.query("condition == 'oracle'")[oracle_metric].mean()
        observation = frame.query("condition == 'observation_only'")[oracle_metric].mean()
        minimum = float(spec.get("oracle_minimum", -np.inf))
        gap = float(spec.get("oracle_gap", 0.0))
        if oracle < minimum:
            raise AssertionError(
                f"{environment} oracle {oracle_metric} {oracle:.4f} is below {minimum:.4f}"
            )
        if oracle <= observation + gap:
            raise AssertionError(f"{environment} oracle did not outperform observation-only")
    if not list((Path(root) / "figures").glob("*.png")):
        raise AssertionError("cross smoke generated no real figures")


def _split_filters(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result = {item.strip() for value in values for item in value.split(",") if item.strip()}
    return result or None


def _filtered_config(
    config: dict[str, Any], environments: set[str] | None, conditions: set[str] | None,
    seeds: set[int] | None,
) -> dict[str, Any]:
    filtered = copy.deepcopy(config)
    if environments is not None:
        unknown = environments - set(filtered["environments"])
        if unknown:
            raise ValueError(f"environment filter contains unknown values: {sorted(unknown)}")
        filtered["environments"] = {
            key: value for key, value in filtered["environments"].items() if key in environments
        }
    if conditions is not None:
        observed: set[str] = set()
        for specification in filtered["environments"].values():
            observed.update(specification["conditions"])
            specification["conditions"] = [
                value for value in specification["conditions"] if value in conditions
            ]
            if not specification["conditions"]:
                raise ValueError("condition filter removed every condition from an environment")
        unknown = conditions - observed
        if unknown:
            raise ValueError(f"condition filter contains unknown values: {sorted(unknown)}")
    if seeds is not None:
        unknown_seeds = seeds - set(map(int, filtered["seeds"]))
        if unknown_seeds:
            raise ValueError(f"seed filter contains values outside the profile: {sorted(unknown_seeds)}")
        filtered["seeds"] = [seed for seed in filtered["seeds"] if int(seed) in seeds]
    return filtered


def build_cross_tasks(
    config: dict[str, Any],
    root: Path,
    *,
    selected_learning_rates: dict[tuple[str, str], float] | None = None,
    resume: bool = False,
    retry_failed: bool = False,
) -> list[tuple[Any, ...]]:
    """Expand one validated profile into isolated run tasks."""

    stage = config.get("experiment_stage", "fixed")
    tasks: list[tuple[Any, ...]] = []
    for environment, spec in config["environments"].items():
        base_alpha = float(spec.get("control_alpha", config["control_alpha"]))
        for condition in spec["conditions"]:
            alpha_options: list[dict[str, Any]]
            if stage == "lr_tune":
                alpha_options = [
                    {
                        "controller_alpha": alpha,
                        "candidate_alpha": alpha,
                        "base_alpha_multiplier": multiplier,
                    }
                    for alpha, multiplier in candidate_alphas(
                        base_alpha, list(map(float, config["alpha_tuning"]["multipliers"]))
                    )
                ]
            elif stage == "lr_eval":
                if selected_learning_rates is None:
                    raise ValueError("lr_eval requires selected_learning_rates.csv")
                key = (environment, condition)
                if key not in selected_learning_rates:
                    raise ValueError(f"selected learning rate missing for {environment}/{condition}")
                alpha_options = [{"controller_alpha": selected_learning_rates[key]}]
            else:
                alpha_options = [{"controller_alpha": base_alpha}]
            for options in alpha_options:
                options.update(resume=resume, retry_failed=retry_failed)
                for seed in config["seeds"]:
                    tasks.append((config, environment, condition, int(seed), str(root), options.copy()))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--environment", action="append", help="environment filter; repeat or comma-separate")
    parser.add_argument("--condition", action="append", help="condition filter; repeat or comma-separate")
    parser.add_argument("--seed", action="append", help="seed filter; repeat or comma-separate")
    parser.add_argument("--selected-learning-rates")
    args = parser.parse_args()
    config = load_cross_config(args.config)
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        parser.error("--run-name may contain only letters, digits, dot, underscore, and hyphen")
    is_full = bool(config.get("remote_full", False) or "full" in config["profile"])
    if is_full and not (args.allow_full_run and os.environ.get("RL_RUN_CONTEXT") == "remote"):
        raise SystemExit(
            "Cross-environment full profile blocked: RL_RUN_CONTEXT=remote and --allow-full-run are both required."
        )
    environments = _split_filters(args.environment)
    conditions = _split_filters(args.condition)
    seed_strings = _split_filters(args.seed)
    try:
        seeds = {int(value) for value in seed_strings} if seed_strings else None
        config = _filtered_config(config, environments, conditions, seeds)
    except ValueError as error:
        parser.error(str(error))
    root = Path(args.output_dir or config["output_dir"]) / args.run_name
    if args.aggregate_only:
        aggregate_cross(root, config)
        print(json.dumps(validate_cross_results(root, config), sort_keys=True))
        return
    selected_rates: dict[tuple[str, str], float] | None = None
    selected_source: Path | None = None
    if config.get("experiment_stage") == "lr_eval":
        selected_path_value = args.selected_learning_rates or config.get("selected_learning_rates")
        if not selected_path_value:
            parser.error("lr_eval requires --selected-learning-rates")
        selected_source = Path(selected_path_value)
        expected_pairs = {
            (environment, condition)
            for environment, specification in config["environments"].items()
            for condition in specification["conditions"]
        }
        try:
            selected_rates = load_selected_learning_rates(selected_source, expected_pairs)
            config["selected_learning_rates_sha256"] = _file_sha256(selected_source)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    tasks = build_cross_tasks(
        config,
        root,
        selected_learning_rates=selected_rates,
        resume=args.resume or args.retry_failed,
        retry_failed=args.retry_failed,
    )
    if args.dry_run:
        states = {
            "expected_runs": len(tasks),
            "completed_runs": sum(_completed_run(_task_run_dir(task)) for task in tasks),
            "pending_runs": sum(not _task_run_dir(task).exists() for task in tasks),
            "existing_incomplete_or_failed_runs": sum(
                _task_run_dir(task).exists() and not _completed_run(_task_run_dir(task))
                for task in tasks
            ),
        }
        print(json.dumps(states, sort_keys=True))
        return
    if root.exists() and not (args.resume or args.retry_failed):
        raise FileExistsError(f"{root} already exists; use --resume or --retry-failed")
    root.mkdir(parents=True, exist_ok=args.resume or args.retry_failed)
    if (root / "config.json").exists():
        existing_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if config_hash(existing_config) != config_hash(config):
            raise ValueError("resume/retry config does not match the existing result root")
    else:
        write_json(root / "config.json", config)
    selected_sha256: str | None = None
    if selected_source is not None:
        selected_sha256 = str(config["selected_learning_rates_sha256"])
        selected_target = root / "selected_learning_rates.csv"
        if selected_target.exists() and _file_sha256(selected_target) != selected_sha256:
            raise ValueError("resume selected_learning_rates.csv differs from the existing result root")
        if not selected_target.exists():
            shutil.copy2(selected_source, selected_target)
    manifest = _manifest_base(config)
    manifest.update(
        run_name=args.run_name,
        start_time=datetime.now(timezone.utc).isoformat(),
        exit_status="running",
        selected_learning_rates_sha256=selected_sha256,
    )
    write_json(root / "manifest.json", manifest)
    workers = max(1, min(args.workers or config["workers"], len(tasks), cpu_count()))
    try:
        if workers == 1:
            list(map(run_cross_one, tasks))
        else:
            with Pool(workers) as pool:
                pool.map(run_cross_one, tasks)
        aggregate_cross(root, config)
        if config["profile"] == "cross_smoke" and not any((environments, conditions, seeds)):
            smoke_cross_assertions(root, config)
        validation = validate_cross_results(root, config)
        manifest.update(
            end_time=datetime.now(timezone.utc).isoformat(),
            exit_status="ok",
            workers=workers,
            validation=validation,
        )
    except Exception as error:
        manifest.update(
            end_time=datetime.now(timezone.utc).isoformat(),
            exit_status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        write_json(root / "manifest.json", manifest)
        raise
    write_json(root / "manifest.json", manifest)
    print(f"RESULT_DIR={root.resolve()}")


if __name__ == "__main__":
    main()
