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
import traceback
from collections import deque
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .controller import SarsaLambda
from .compact_storage import (
    COMPACT_SCHEMA_VERSION,
    COMPACT_TRACE_FILES,
    HiddenVelocityAccumulator,
    ScalarMoments,
    read_columnar_npz,
    write_columnar_npz,
)
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
from .experiment import config_hash, dependency_versions, git_value, write_json
from .predictive import CausalPredictiveBank
from .priors import MATCHED_PRIOR_FOR_ENV, TaskMatchedTransform
from .production_runtime import (
    contained_path,
    repository_root,
    safe_worker_limit,
    validate_runtime_environment,
)
from .robust_stats import build_robust_summaries
from .runtime_validation import RuntimeValidityError, RuntimeValidityTracker
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
LEGACY_RUN_REQUIRED_ENTRIES = {
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
COMPACT_RUN_REQUIRED_ENTRIES = {
    "config.json",
    "manifest.json",
    "predictive_definitions.json",
    "representation_metrics.csv",
    "task_information_metrics.csv",
    "decision_probe_by_position.csv",
    "summary.csv",
    "summary.json",
    "runtime_validity.json",
    "stdout.log",
    "figures",
    *COMPACT_TRACE_FILES,
}
INVALID_TUNING_COLUMNS = [
    "environment",
    "condition",
    "seed",
    "candidate_alpha",
    "base_alpha_multiplier",
    "exit_status",
    "failure_classification",
    "metric",
    "observed_value",
    "threshold",
    "interaction_index",
    "manifest_path",
    "runtime_validity_path",
    "config_hash",
    "git_commit",
    "result_schema_version",
]
RUNTIME_FAILURE_FIELDS = {
    "environment",
    "condition",
    "seed",
    "candidate_alpha",
    "interaction_index",
    "metric",
    "observed_value",
    "threshold",
    "relevant_feature_norm",
    "parameter_norm",
    "update_norm",
    "transform_state",
    "error_type",
    "failure_classification",
}


def required_run_entries(config: dict[str, Any]) -> set[str]:
    if config.get("storage_schema", "legacy_csv") == "compact_v2":
        return set(COMPACT_RUN_REQUIRED_ENTRIES)
    return set(LEGACY_RUN_REQUIRED_ENTRIES) | {"runtime_validity.json"}


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
    storage_schema = str(config.get("storage_schema", "legacy_csv"))
    if storage_schema not in {"legacy_csv", "compact_v2"}:
        raise ValueError("storage_schema must be legacy_csv or compact_v2")
    defaults = {
        "transform_eps": 1e-3,
        "transform_min_samples": 64,
        "cov_update_every": 50,
        "moment_beta": 0.005,
        "moment_learning_rate": 0.0005,
        "covariance_shrinkage": 0.05,
        "matrix_smoothing": 0.1,
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
        "storage_schema": "legacy_csv",
        "result_schema_version": COMPACT_SCHEMA_VERSION,
        "extreme_finite_limit": 1e12,
        "diagnostic_full_trace_runs": [],
        "enforce_repository_containment": False,
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
        "result_schema_version": int(config.get("result_schema_version", COMPACT_SCHEMA_VERSION)),
        "storage_schema": config.get("storage_schema", "legacy_csv"),
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
        covariance_shrinkage=config.get("covariance_shrinkage", 0.05),
        matrix_smoothing=config.get("matrix_smoothing", 0.1),
    )


def _transform(transform: Any, values: np.ndarray) -> np.ndarray:
    return values.copy() if transform is None else transform.transform(values)


def _transform_snapshot(transform: Any) -> dict[str, Any]:
    if transform is None:
        return {"kind": "none"}
    result: dict[str, Any] = {
        "kind": str(getattr(transform, "kind", type(transform).__name__)),
        "sample_count": int(getattr(getattr(transform, "stats", None), "n", 0)),
        "last_output_rms": float(getattr(transform, "last_output_rms", 0.0)),
        "last_output_max_abs": float(getattr(transform, "last_output_max_abs", 0.0)),
        "last_matrix_change": float(getattr(transform, "last_matrix_change", 0.0)),
        "whitening_gain": float(getattr(transform, "last_whitening_gain", 1.0)),
    }
    if hasattr(transform, "gaussian_skew_parameter"):
        result.update(
            gaussian_skew_parameter=np.asarray(transform.gaussian_skew_parameter).tolist(),
            gaussian_tail_parameter=np.asarray(transform.gaussian_tail_parameter).tolist(),
        )
    return result


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
    compact_storage = config.get("storage_schema", "legacy_csv") == "compact_v2"
    step_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    compact_trace_rows: list[dict[str, Any]] = []
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
    reward_stats = ScalarMoments()
    stabilization_stats = ScalarMoments()
    control_delta_stats = ScalarMoments()
    control_update_stats = ScalarMoments()
    predictive_update_stats = ScalarMoments()
    predictive_mse_stats = ScalarMoments()
    final_reward_window: deque[float] = deque(maxlen=int(config["final_window"]))
    final_stabilized_window: deque[float] = deque(maxlen=int(config["final_window"]))
    prediction_td_squared_sum = np.zeros(bank.d, dtype=np.float64)
    prediction_cumulant_sum = np.zeros(bank.d, dtype=np.float64)
    hidden_accumulator = (
        HiddenVelocityAccumulator(
            environment=environment,
            final_window=int(config["final_window"]),
            settling_consecutive_steps=int(config.get("settling_consecutive_steps", 20)),
            recovery_consecutive_steps=int(config.get("recovery_consecutive_steps", 8)),
        )
        if environment in {"hidden_velocity", "hidden_velocity_informative"}
        else None
    )
    tracker = RuntimeValidityTracker(
        environment=environment,
        condition=condition,
        seed=seed,
        candidate_alpha=task_options.get("candidate_alpha"),
        run_dir=run_dir,
        extreme_finite_limit=float(config.get("extreme_finite_limit", 1e12)),
    )
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
        try:
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
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            tracker.record_exception(
                interaction,
                error,
                feature_norm=float(np.linalg.norm(features)),
                parameter_norm=float(np.linalg.norm(controller.w)),
                transform_state=_transform_snapshot(transform),
            )
            raise AssertionError("unreachable after fail-closed runtime exception")
        transform_state = _transform_snapshot(transform)
        transform_metrics: dict[str, Any] = {}
        if transform is not None:
            if hasattr(transform, "W"):
                transform_metrics["whitening_state"] = transform.W
            if hasattr(transform, "stats"):
                denominator = max(int(transform.stats.n) - 1, 1)
                transform_metrics["covariance_state"] = (
                    transform.stats.M2 / denominator
                )
            if hasattr(transform, "gaussian_raw_moments"):
                transform_metrics["moment_estimator_state"] = transform.gaussian_raw_moments
        tracker.observe(
            interaction,
            {
                "controller_features": features,
                "next_controller_features": next_features,
                "predictive_parameters": bank.w,
                "controller_parameters": controller.w,
                "controller_eligibility_trace": controller.e,
                "predictive_stream_state": bank.current_x,
                "predictive_current_predictions": bank.current_predictions,
                "predictive_trace_state": bank.m,
                "control_td_error": control_delta,
                "control_update_norm": update_norm,
                "predictive_td_error": prediction_update.deltas,
                "predictive_update_norm": prediction_update.update_norm,
                "reward": reward,
                "cost": info.diagnostics.get("total_cost") if info.diagnostics else None,
                "effective_controller_alpha": controller.last_effective_alpha,
                **transform_metrics,
            },
            feature_norm=float(np.linalg.norm(features)),
            parameter_norm=parameter_norm,
            update_norm=update_norm,
            transform_state=transform_state,
        )

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
        reward_stats.update(float(reward))
        final_reward_window.append(float(reward))
        if not compact_storage:
            all_rewards.append(float(reward))
        recent_rewards.append(float(reward))
        if not compact_storage:
            control_deltas.append(control_delta)
            control_updates.append(update_norm)
            predictive_updates.append(prediction_update.update_norm)
            predictive_mse.append(prediction_update.mse)
        control_delta_stats.update(control_delta)
        control_update_stats.update(update_norm)
        predictive_update_stats.update(prediction_update.update_norm)
        predictive_mse_stats.update(prediction_update.mse)
        prediction_td_squared_sum += np.square(prediction_update.deltas)
        prediction_cumulant_sum += prediction_update.cumulants
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
            if not compact_storage:
                all_stabilized.append(float(info.stabilized))
            stabilization_stats.update(float(info.stabilized))
            final_stabilized_window.append(float(info.stabilized))
        if hidden_environment:
            diagnostic = dict(info.diagnostics)
            if not diagnostic:
                raise AssertionError("hidden-velocity step omitted authoritative diagnostics")
            if not np.isclose(float(diagnostic["reward"]), -float(diagnostic["total_cost"])):
                raise AssertionError("hidden-velocity reward must equal negative total cost")
            if not np.isclose(reward, float(diagnostic["reward"])):
                raise AssertionError("logged hidden-velocity reward differs from environment reward")
            if compact_storage:
                assert hidden_accumulator is not None
                hidden_accumulator.update(interaction, diagnostic)
            else:
                hidden_diagnostics.append(diagnostic)

        moving_reward = float(np.mean(recent_rewards))
        moving_performance = (
            float(np.mean(recent_correct)) if recent_correct else moving_reward
        )
        if threshold_time < 0 and len(recent_rewards) == recent_rewards.maxlen:
            if moving_performance >= target_threshold:
                threshold_time = interaction
        # Some continuing-control environments make an action decision on every
        # transition.  Only externally scored decisions need an event row;
        # otherwise ``info.decision`` would silently defeat the configured
        # stride and restore full-resolution production traces.
        record_diagnostics = (
            interaction % config["metrics_stride"] == 0 or info.correct is not None
        )
        if record_diagnostics or (
            compact_storage and hidden_environment and bool(info.diagnostics.get("disturbance_active", False))
        ):
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
            if compact_storage:
                compact_trace_rows.append(
                    {
                        **step_rows[-1],
                        "predictive_td_mse": prediction_update.mse,
                        "predictive_update_norm": prediction_update.update_norm,
                        "predictive_parameter_norm": prediction_update.parameter_norm,
                        "control_td_error": control_delta,
                        "control_update_norm": update_norm,
                        "controller_parameter_norm": parameter_norm,
                        "effective_controller_alpha": controller.last_effective_alpha,
                        "feature_squared_norm": float(features @ features),
                    }
                )

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
    final_correct = all_correct[-config["final_window"] :]
    final_reward = (
        float(np.mean(final_reward_window))
        if compact_storage
        else float(np.mean(all_rewards[-config["final_window"] :]))
    )
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
    validity = tracker.finalize()
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
        "mean_reward": reward_stats.mean if compact_storage else float(np.mean(all_rewards)),
        "final_window_reward": final_reward,
        "final_performance": final_performance,
        "time_to_threshold": threshold_time,
        "change_point": actual_change_point if actual_change_point is not None else -1,
        "pre_change_accuracy": pre_change_accuracy,
        "final_post_change_accuracy": final_post_change_accuracy,
        "recovery_time": recovery_time,
        "stabilization_rate": (
            stabilization_stats.mean
            if compact_storage and stabilization_stats.n
            else (float(np.mean(all_stabilized)) if all_stabilized else 0.0)
        ),
        "final_window_stabilization": (
            float(np.mean(final_stabilized_window))
            if compact_storage and final_stabilized_window
            else (
                float(np.mean(all_stabilized[-config["final_window"] :]))
                if all_stabilized
                else 0.0
            )
        ),
        "control_cost": (
            -reward_stats.mean
            if compact_storage and environment == "hidden_velocity"
            else (
                float(-np.mean(all_rewards)) if environment == "hidden_velocity" else 0.0
            )
        ),
        "predictive_td_mse": (
            predictive_mse_stats.mean if compact_storage else float(np.mean(predictive_mse))
        ),
        "control_td_variance": (
            control_delta_stats.variance if compact_storage else float(np.var(control_deltas))
        ),
        "mean_control_update_norm": (
            control_update_stats.mean if compact_storage else float(np.mean(control_updates))
        ),
        "max_control_update_norm": (
            control_update_stats.maximum if compact_storage else float(np.max(control_updates))
        ),
        "mean_predictive_update_norm": (
            predictive_update_stats.mean
            if compact_storage
            else float(np.mean(predictive_updates))
        ),
        "final_parameter_norm": float(np.linalg.norm(controller.w)),
        "nan_count": int(validity["nan_count"]),
        "inf_count": int(validity["inf_count"]),
        "divergence_flag": int(validity["divergence_flag"]),
        "wall_seconds": time.time() - started,
        "run_dir": str(run_dir),
        "run_status": "valid" if compact_storage else "ok",
        "result_schema_version": int(config.get("result_schema_version", COMPACT_SCHEMA_VERSION)),
        "storage_schema": config.get("storage_schema", "legacy_csv"),
        "experiment_stage": config.get("experiment_stage", "fixed"),
        "candidate_alpha": task_options.get("candidate_alpha", np.nan),
        "base_alpha_multiplier": task_options.get("base_alpha_multiplier", np.nan),
        **task_metrics,
        **decision_metrics,
        **controller.alpha_metrics(),
        **HIDDEN_SUMMARY_DEFAULTS,
    }
    if hidden_environment:
        if compact_storage:
            assert hidden_accumulator is not None
            summary.update(hidden_accumulator.finalize())
        else:
            summary.update(_hidden_velocity_summary(hidden_diagnostics, config, environment))
        summary["control_cost"] = summary["mean_total_cost"]

    if not compact_storage:
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
    if compact_storage:
        write_columnar_npz(run_dir / "strided_trace.npz", compact_trace_rows)
        write_columnar_npz(run_dir / "decision_event_trace.npz", decision_rows)
        write_columnar_npz(
            run_dir / "disturbance_event_trace.npz",
            hidden_accumulator.event_rows if hidden_accumulator is not None else [],
        )
        count = max(int(env_spec["interactions"]), 1)
        np.savez_compressed(
            run_dir / "prediction_feature_summary.npz",
            feature_names=np.asarray(bank.feature_names, dtype="U"),
            mean_td_squared=prediction_td_squared_sum / count,
            mean_cumulant=prediction_cumulant_sum / count,
            sample_count=np.asarray([count], dtype=np.int64),
        )
        state_payload: dict[str, np.ndarray] = {
            "predictive_weights": bank.w,
            "controller_weights": controller.w,
            "controller_trace": controller.e,
        }
        if transform is not None and hasattr(transform, "W"):
            state_payload["transform_matrix"] = transform.W
        if transform is not None and hasattr(transform, "gaussian_raw_moments"):
            state_payload["gaussian_raw_moments"] = transform.gaussian_raw_moments
            state_payload["gaussian_skew_parameter"] = transform.gaussian_skew_parameter
            state_payload["gaussian_tail_parameter"] = transform.gaussian_tail_parameter
        np.savez_compressed(run_dir / "model_state.npz", **state_payload)
    else:
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
    diagnostic_identity = f"{environment}/{condition}/seed_{seed:03d}"
    if not compact_storage or diagnostic_identity in set(config.get("diagnostic_full_trace_runs", [])):
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
    if not compact_storage:
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


def _tuning_identity(
    environment: str, condition: str, seed: int, candidate_alpha: float
) -> tuple[str, str, int, str]:
    return str(environment), str(condition), int(seed), _alpha_path(float(candidate_alpha))


def _expected_tuning_identities(
    config: dict[str, Any],
) -> dict[tuple[str, str, int, str], dict[str, float]]:
    expected: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for environment, specification in config["environments"].items():
        base_alpha = float(specification.get("control_alpha", config["control_alpha"]))
        alphas = candidate_alphas(
            base_alpha, list(map(float, config["alpha_tuning"]["multipliers"]))
        )
        for condition in specification["conditions"]:
            for alpha, multiplier in alphas:
                for seed in config["seeds"]:
                    identity = _tuning_identity(environment, condition, int(seed), alpha)
                    if identity in expected:
                        raise ValueError(f"duplicate configured tuning identity {identity}")
                    expected[identity] = {
                        "candidate_alpha": float(alpha),
                        "base_alpha_multiplier": float(multiplier),
                    }
    return expected


def _invalid_tuning_record(run_dir: Path, root: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    validity_path = run_dir / "runtime_validity.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("exit_status") != "invalid":
        return None
    try:
        validity = json.loads(validity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        validity = {}
    failure = validity.get("first_failure")
    if not isinstance(failure, dict):
        failure = {}
    return {
        "environment": manifest.get("environment", failure.get("environment")),
        "condition": manifest.get("condition", failure.get("condition")),
        "seed": manifest.get("seed", failure.get("seed")),
        "candidate_alpha": manifest.get(
            "candidate_alpha", failure.get("candidate_alpha")
        ),
        "base_alpha_multiplier": manifest.get("base_alpha_multiplier"),
        "exit_status": manifest.get("exit_status"),
        "failure_classification": failure.get("failure_classification"),
        "metric": failure.get("metric"),
        "observed_value": failure.get("observed_value"),
        "threshold": failure.get("threshold"),
        "interaction_index": failure.get("interaction_index"),
        "manifest_path": str(manifest_path.relative_to(root)),
        "runtime_validity_path": str(validity_path.relative_to(root)),
        "config_hash": manifest.get("config_hash"),
        "git_commit": manifest.get("git_commit"),
        "result_schema_version": manifest.get("result_schema_version"),
    }


def _read_invalid_tuning_attempts(root: Path) -> pd.DataFrame:
    records = []
    runs_root = root / "runs"
    if runs_root.is_dir():
        for manifest_path in sorted(runs_root.rglob("manifest.json")):
            record = _invalid_tuning_record(manifest_path.parent, root)
            if record is not None:
                records.append(record)
    return pd.DataFrame(records, columns=INVALID_TUNING_COLUMNS)


def _completed_run(
    run_dir: Path,
    config: dict[str, Any] | None = None,
    expected_identity: tuple[str, str, int] | None = None,
    options: dict[str, Any] | None = None,
    expected_commit: str | None = None,
) -> bool:
    if config is None:
        try:
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
    required = required_run_entries(config)
    if not run_dir.is_dir() or not required <= {path.name for path in run_dir.iterdir()}:
        return False
    if not (run_dir / "figures").is_dir() or any(
        not (run_dir / name).is_file() for name in required - {"figures"}
    ):
        return False
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = pd.read_csv(run_dir / "summary.csv")
        validity = json.loads((run_dir / "runtime_validity.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    if expected_identity is not None:
        environment, condition, seed = expected_identity
        if (
            manifest.get("environment") != environment
            or manifest.get("condition") != condition
            or int(manifest.get("seed", -1)) != int(seed)
        ):
            return False
    expected_alpha = (options or {}).get("candidate_alpha")
    observed_alpha = manifest.get("candidate_alpha")
    if expected_alpha is not None and (
        observed_alpha is None or not np.isclose(float(observed_alpha), float(expected_alpha))
    ):
        return False
    required_numeric = [
        "final_performance",
        "control_td_variance",
        "max_control_update_norm",
        "final_parameter_norm",
        "nan_count",
        "inf_count",
        "divergence_flag",
    ]
    if len(summary) != 1 or any(column not in summary for column in required_numeric):
        return False
    values = summary[required_numeric].to_numpy(dtype=float)
    extreme_limit = float(config.get("extreme_finite_limit", 1e12))
    return bool(
        manifest.get("exit_status") == "ok"
        and manifest.get("resume_eligible") is True
        and manifest.get("config_hash") == config_hash(config)
        and int(manifest.get("result_schema_version", -1))
        == int(config.get("result_schema_version", COMPACT_SCHEMA_VERSION))
        and manifest.get("git_commit")
        == (expected_commit or git_value("rev-parse", "HEAD"))
        and summary.iloc[0].get("run_status") in {"ok", "valid"}
        and validity.get("status") == "valid"
        and validity.get("resume_eligible") is True
        and np.isfinite(values).all()
        and int(summary.iloc[0]["nan_count"]) == 0
        and int(summary.iloc[0]["inf_count"]) == 0
        and int(summary.iloc[0]["divergence_flag"]) == 0
        and not (np.abs(values[:, :4]) > extreme_limit).any()
    )


def _completed_invalid_tuning_run(
    run_dir: Path,
    config: dict[str, Any],
    expected_identity: tuple[str, str, int],
    options: dict[str, Any],
    expected_commit: str | None = None,
) -> bool:
    if config.get("experiment_stage") != "lr_tune":
        return False
    required = {
        "config.json",
        "manifest.json",
        "runtime_validity.json",
        "stdout.log",
        "figures",
    }
    if not run_dir.is_dir() or not required <= {path.name for path in run_dir.iterdir()}:
        return False
    if (run_dir / "summary.csv").exists() or (run_dir / "summary.json").exists():
        return False
    try:
        run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        validity = json.loads((run_dir / "runtime_validity.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    environment, condition, seed = expected_identity
    expected_alpha = options.get("candidate_alpha")
    expected_multiplier = options.get("base_alpha_multiplier")
    failure = validity.get("first_failure")
    manifest_failure = manifest.get("first_failure")
    if not isinstance(failure, dict) or not RUNTIME_FAILURE_FIELDS <= set(failure):
        return False
    if not isinstance(manifest_failure, dict) or json.dumps(
        manifest_failure, sort_keys=True
    ) != json.dumps(failure, sort_keys=True):
        return False
    try:
        identity_matches = (
            manifest.get("environment") == environment
            and manifest.get("condition") == condition
            and int(manifest.get("seed", -1)) == int(seed)
            and failure.get("environment") == environment
            and failure.get("condition") == condition
            and int(failure.get("seed", -1)) == int(seed)
            and expected_alpha is not None
            and manifest.get("candidate_alpha") is not None
            and failure.get("candidate_alpha") is not None
            and np.isclose(float(manifest["candidate_alpha"]), float(expected_alpha))
            and np.isclose(float(failure["candidate_alpha"]), float(expected_alpha))
            and expected_multiplier is not None
            and manifest.get("base_alpha_multiplier") is not None
            and np.isclose(
                float(manifest["base_alpha_multiplier"]), float(expected_multiplier)
            )
        )
    except (TypeError, ValueError):
        return False
    return bool(
        identity_matches
        and manifest.get("exit_status") == "invalid"
        and manifest.get("run_status") == "invalid"
        and manifest.get("resume_eligible") is False
        and manifest.get("error_type") == "RuntimeValidityError"
        and validity.get("status") == "invalid"
        and validity.get("resume_eligible") is False
        and int(validity.get("divergence_flag", 0)) == 1
        and config_hash(run_config) == config_hash(config)
        and manifest.get("config_hash") == config_hash(config)
        and int(manifest.get("result_schema_version", -1))
        == int(config.get("result_schema_version", COMPACT_SCHEMA_VERSION))
        and manifest.get("git_commit")
        == (expected_commit or git_value("rev-parse", "HEAD"))
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
        if _completed_run(
            run_dir, config, (environment, condition, seed), options
        ) and (options.get("resume") or options.get("retry_failed")):
            return pd.read_csv(run_dir / "summary.csv").iloc[0].to_dict()
        if (
            _completed_invalid_tuning_run(
                run_dir, config, (environment, condition, seed), options
            )
            and options.get("resume")
            and not options.get("retry_failed")
        ):
            record = _invalid_tuning_record(run_dir, root)
            if record is None:
                raise RuntimeError(f"completed invalid tuning evidence disappeared: {run_dir}")
            return {"worker_status": "invalid_tuning_candidate", **record}
        manifest_path = run_dir / "manifest.json"
        failed = False
        if manifest_path.exists():
            try:
                failed = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "exit_status"
                ) in {"failed", "invalid", "interrupted"}
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
        invalid = isinstance(error, RuntimeValidityError)
        error_traceback = traceback.format_exc()
        manifest.update(
            end_time=datetime.now(timezone.utc).isoformat(),
            exit_status="invalid" if invalid else "failed",
            run_status="invalid" if invalid else "failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=error_traceback,
            resume_eligible=False,
        )
        if invalid:
            manifest["first_failure"] = vars(error.failure)
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "stdout.log").write_text(
            f"failed environment={environment} condition={condition} seed={seed}\n"
            f"{type(error).__name__}: {error}\n{error_traceback}",
            encoding="utf-8",
        )
        raise
    manifest.update(
        end_time=datetime.now(timezone.utc).isoformat(),
        exit_status="ok",
        run_status=summary["run_status"],
        resume_eligible=True,
        wall_seconds=summary["wall_seconds"],
    )
    write_json(run_dir / "manifest.json", manifest)
    return summary


def run_cross_worker(task: tuple[Any, ...]) -> dict[str, Any]:
    """Return structured tuning-invalid outcomes while propagating all other failures."""

    config, environment, condition, seed, root_string, _ = _task_parts(task)
    try:
        result = run_cross_one(task)
    except RuntimeValidityError:
        if config.get("experiment_stage") != "lr_tune":
            raise
        root = Path(root_string)
        run_dir = _task_run_dir(task)
        record = _invalid_tuning_record(run_dir, root)
        if record is None:
            raise RuntimeError(
                "lr-tune RuntimeValidityError did not leave structured invalid evidence "
                f"for {environment}/{condition}/seed={seed}"
            )
        return {"worker_status": "invalid_tuning_candidate", **record}
    if result.get("worker_status") == "invalid_tuning_candidate":
        return result
    return {
        "worker_status": "valid",
        "environment": environment,
        "condition": condition,
        "seed": seed,
        "candidate_alpha": result.get("candidate_alpha"),
    }


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


def _read_compact_many(paths: list[Path]) -> pd.DataFrame:
    """Read run partitions one at a time; no monolithic aggregate copy is written."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.is_file():
            frame = read_columnar_npz(path)
            if len(frame):
                frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _validate_tuning_attempt_coverage(
    root: Path,
    config: dict[str, Any],
    summaries: pd.DataFrame,
    invalid: pd.DataFrame,
) -> dict[str, Any]:
    expected = _expected_tuning_identities(config)
    expected_run_dirs = {
        _task_run_dir(
            (
                config,
                identity[0],
                identity[1],
                identity[2],
                str(root),
                values,
            )
        ).resolve()
        for identity, values in expected.items()
    }
    observed_run_dirs = {
        path.resolve()
        for path in (root / "runs").rglob("seed_*")
        if path.is_dir()
    }
    if observed_run_dirs != expected_run_dirs:
        raise AssertionError(
            "tuning run-directory coverage mismatch; "
            f"missing={len(expected_run_dirs - observed_run_dirs)} "
            f"extra={len(observed_run_dirs - expected_run_dirs)}"
        )
    summary_required = {
        "environment",
        "condition",
        "seed",
        "candidate_alpha",
        "base_alpha_multiplier",
        "run_status",
        "run_dir",
    }
    missing_summary = summary_required - set(summaries)
    if missing_summary:
        raise AssertionError(f"tuning summaries are missing {sorted(missing_summary)}")
    missing_invalid = set(INVALID_TUNING_COLUMNS) - set(invalid)
    if missing_invalid:
        raise AssertionError(
            f"invalid tuning candidate inventory is missing {sorted(missing_invalid)}"
        )
    current_commit = git_value("rev-parse", "HEAD")
    valid_identities: set[tuple[str, str, int, str]] = set()
    invalid_identities: set[tuple[str, str, int, str]] = set()
    for row in summaries.itertuples(index=False):
        identity = _tuning_identity(
            row.environment, row.condition, int(row.seed), float(row.candidate_alpha)
        )
        if identity in valid_identities:
            raise AssertionError(f"duplicate valid tuning identity {identity}")
        if identity not in expected:
            raise AssertionError(f"unexpected valid tuning identity {identity}")
        expected_values = expected[identity]
        if not np.isclose(
            float(row.base_alpha_multiplier), expected_values["base_alpha_multiplier"]
        ):
            raise AssertionError(f"tuning multiplier mismatch for {identity}")
        options = {
            "candidate_alpha": expected_values["candidate_alpha"],
            "base_alpha_multiplier": expected_values["base_alpha_multiplier"],
        }
        run_dir = _task_run_dir(
            (
                config,
                identity[0],
                identity[1],
                identity[2],
                str(root),
                options,
            )
        )
        if not _completed_run(
            run_dir,
            config,
            (identity[0], identity[1], identity[2]),
            options,
            expected_commit=current_commit,
        ):
            raise AssertionError(f"invalid or incomplete valid tuning run {run_dir}")
        valid_identities.add(identity)
    for row in invalid.itertuples(index=False):
        try:
            identity = _tuning_identity(
                row.environment, row.condition, int(row.seed), float(row.candidate_alpha)
            )
        except (TypeError, ValueError) as error:
            raise AssertionError("invalid tuning inventory has an incomplete identity") from error
        if identity in invalid_identities:
            raise AssertionError(f"duplicate invalid tuning identity {identity}")
        if identity not in expected:
            raise AssertionError(f"unexpected invalid tuning identity {identity}")
        expected_values = expected[identity]
        if row.exit_status != "invalid":
            raise AssertionError(f"invalid tuning inventory status mismatch for {identity}")
        if not row.failure_classification or not row.metric:
            raise AssertionError(f"invalid tuning failure evidence is incomplete for {identity}")
        if not np.isfinite(float(row.threshold)) or int(row.interaction_index) < 0:
            raise AssertionError(f"invalid tuning threshold/interaction is malformed for {identity}")
        if not np.isclose(
            float(row.base_alpha_multiplier), expected_values["base_alpha_multiplier"]
        ):
            raise AssertionError(f"invalid tuning multiplier mismatch for {identity}")
        options = {
            "candidate_alpha": expected_values["candidate_alpha"],
            "base_alpha_multiplier": expected_values["base_alpha_multiplier"],
        }
        run_dir = _task_run_dir(
            (
                config,
                identity[0],
                identity[1],
                identity[2],
                str(root),
                options,
            )
        )
        expected_manifest = str((run_dir / "manifest.json").relative_to(root))
        if str(row.manifest_path) != expected_manifest:
            raise AssertionError(f"invalid tuning manifest path mismatch for {identity}")
        if not _completed_invalid_tuning_run(
            run_dir,
            config,
            (identity[0], identity[1], identity[2]),
            options,
            expected_commit=current_commit,
        ):
            raise AssertionError(f"incomplete invalid tuning evidence {run_dir}")
        invalid_identities.add(identity)
    overlap = valid_identities & invalid_identities
    if overlap:
        raise AssertionError(f"tuning identities are both valid and invalid: {sorted(overlap)}")
    attempted = valid_identities | invalid_identities
    missing = set(expected) - attempted
    extra = attempted - set(expected)
    if missing or extra:
        raise AssertionError(
            "tuning attempted identity coverage mismatch; "
            f"missing={sorted(missing)[:10]} extra={sorted(extra)[:10]}"
        )
    eligible_alphas: dict[tuple[str, str], set[float]] = {}
    pair_counts: dict[tuple[str, str], dict[str, int | bool]] = {}
    for environment, specification in config["environments"].items():
        base_alpha = float(specification.get("control_alpha", config["control_alpha"]))
        alphas = candidate_alphas(
            base_alpha, list(map(float, config["alpha_tuning"]["multipliers"]))
        )
        for condition in specification["conditions"]:
            pair = (str(environment), str(condition))
            eligible = {
                float(alpha)
                for alpha, _ in alphas
                if all(
                    _tuning_identity(environment, condition, int(seed), alpha)
                    in valid_identities
                    for seed in config["seeds"]
                )
            }
            if not eligible:
                raise AssertionError(
                    f"no fully valid learning-rate candidate for {environment}/{condition}"
                )
            eligible_alphas[pair] = eligible
            pair_valid = sum(
                identity[0] == pair[0] and identity[1] == pair[1]
                for identity in valid_identities
            )
            pair_invalid = sum(
                identity[0] == pair[0] and identity[1] == pair[1]
                for identity in invalid_identities
            )
            pair_expected = len(config["seeds"]) * len(alphas)
            invalid_alpha_count = sum(
                any(
                    _tuning_identity(environment, condition, int(seed), alpha)
                    in invalid_identities
                    for seed in config["seeds"]
                )
                for alpha, _ in alphas
            )
            pair_counts[pair] = {
                "valid_candidate_count": len(eligible),
                "invalid_candidate_count": invalid_alpha_count,
                "attempted_candidate_count": len(eligible) + invalid_alpha_count,
                "expected_candidate_count": len(alphas),
                "valid_attempt_count": pair_valid,
                "invalid_attempt_count": pair_invalid,
                "attempted_identity_count": pair_valid + pair_invalid,
                "expected_identity_count": pair_expected,
                "all_candidates_attempted": pair_valid + pair_invalid == pair_expected,
            }
    return {
        "expected_attempts": len(expected),
        "attempted_candidates": len(attempted),
        "valid_runs": len(valid_identities),
        "invalid_attempts": len(invalid_identities),
        "eligible_alphas": eligible_alphas,
        "pair_counts": pair_counts,
    }


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
    from .cross_reporting import make_cross_figures
    if config is None:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    summaries = _read_many(sorted((root / "runs").rglob("summary.csv")))
    invalid_tuning = pd.DataFrame(columns=INVALID_TUNING_COLUMNS)
    if config.get("experiment_stage") == "lr_tune":
        invalid_tuning = _read_invalid_tuning_attempts(root)
        invalid_tuning.to_csv(root / "invalid_tuning_candidates.csv", index=False)
        write_json(
            root / "invalid_tuning_candidates.json",
            json.loads(invalid_tuning.to_json(orient="records")),
        )
        if summaries.empty:
            raise AssertionError("lr-tune has no fully valid learning-rate candidate")
        _validate_tuning_attempt_coverage(root, config, summaries, invalid_tuning)
    representation = _read_many(
        sorted((root / "runs").rglob("representation_metrics.csv"))
    )
    task = _read_many(sorted((root / "runs").rglob("task_information_metrics.csv")))
    compact_storage = config.get("storage_schema", "legacy_csv") == "compact_v2"
    if compact_storage:
        trace_paths = sorted((root / "runs").rglob("strided_trace.npz"))
        decision_paths = sorted((root / "runs").rglob("decision_event_trace.npz"))
        steps = _read_compact_many(trace_paths)
        decisions = _read_compact_many(decision_paths)
        updates = steps
        write_json(
            root / "trace_partition_inventory.json",
            {
                "storage_schema": "compact_v2",
                "strided_partitions": len(trace_paths),
                "decision_partitions": len(decision_paths),
                "strided_bytes": sum(path.stat().st_size for path in trace_paths),
                "decision_bytes": sum(path.stat().st_size for path in decision_paths),
                "monolithic_raw_aggregates_written": False,
            },
        )
    else:
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
    if not compact_storage:
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
        selected = select_learning_rates(summaries, config, invalid_tuning)
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
    stage = config.get("experiment_stage", "fixed")
    if stage != "lr_tune" and len(summaries) != expected:
        raise AssertionError(f"expected {expected} cross runs, found {len(summaries)}")
    identity = ["environment", "condition", "seed"]
    if stage == "lr_tune":
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
    if not summaries["run_status"].isin(["ok", "valid"]).all():
        raise AssertionError("at least one cross-environment run failed")
    if stage == "lr_tune":
        invalid_csv_path = root / "invalid_tuning_candidates.csv"
        invalid_json_path = root / "invalid_tuning_candidates.json"
        if not invalid_csv_path.is_file() or not invalid_json_path.is_file():
            raise AssertionError("lr-tune is missing the invalid candidate inventory")
        invalid = _read_invalid_tuning_attempts(root)
        persisted_invalid = pd.read_csv(invalid_csv_path)
        persisted_json = json.loads(invalid_json_path.read_text(encoding="utf-8"))
        if len(persisted_invalid) != len(invalid) or len(persisted_json) != len(invalid):
            raise AssertionError("persisted invalid tuning inventory count mismatch")
        try:
            pd.testing.assert_frame_equal(
                persisted_invalid.reset_index(drop=True),
                invalid.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
            )
        except AssertionError as error:
            raise AssertionError("persisted invalid tuning CSV content mismatch") from error
        expected_json = json.loads(invalid.to_json(orient="records"))
        if persisted_json != expected_json:
            raise AssertionError("persisted invalid tuning JSON content mismatch")
        coverage = _validate_tuning_attempt_coverage(root, config, summaries, invalid)
        selected_path = root / "selected_learning_rates.csv"
        if not selected_path.is_file():
            raise AssertionError("lr-tune did not produce selected_learning_rates.csv")
        selected = pd.read_csv(selected_path)
        selected_required = {
            "environment",
            "condition",
            "selected_alpha",
            "valid_candidate_count",
            "invalid_candidate_count",
            "attempted_candidate_count",
            "expected_candidate_count",
            "valid_attempt_count",
            "invalid_attempt_count",
            "attempted_identity_count",
            "expected_identity_count",
            "all_candidates_attempted",
        }
        missing_selected = selected_required - set(selected)
        if missing_selected:
            raise AssertionError(
                f"selected learning rates are missing {sorted(missing_selected)}"
            )
        expected_pairs = set(coverage["eligible_alphas"])
        observed_pairs = {
            (str(row.environment), str(row.condition))
            for row in selected.itertuples(index=False)
        }
        if observed_pairs != expected_pairs or selected.duplicated(
            ["environment", "condition"]
        ).any():
            raise AssertionError("selected learning-rate pair coverage mismatch")
        for row in selected.itertuples(index=False):
            pair = (str(row.environment), str(row.condition))
            if not any(
                np.isclose(float(row.selected_alpha), alpha)
                for alpha in coverage["eligible_alphas"][pair]
            ):
                raise AssertionError(
                    f"selected alpha is not a fully valid configured candidate for {pair}"
                )
            counts = coverage["pair_counts"][pair]
            for column in (
                "valid_candidate_count",
                "invalid_candidate_count",
                "attempted_candidate_count",
                "expected_candidate_count",
                "valid_attempt_count",
                "invalid_attempt_count",
                "attempted_identity_count",
                "expected_identity_count",
            ):
                if int(getattr(row, column)) != int(counts[column]):
                    raise AssertionError(f"selected learning-rate {column} mismatch for {pair}")
            if bool(row.all_candidates_attempted) is not True:
                raise AssertionError(f"selected learning-rate attempts are incomplete for {pair}")
        return {
            "result_dir": str(root.resolve()),
            "expected_attempts": int(coverage["expected_attempts"]),
            "attempted_candidates": int(coverage["attempted_candidates"]),
            "valid_runs": int(coverage["valid_runs"]),
            "invalid_attempts": int(coverage["invalid_attempts"]),
        }
    for run_dir in sorted(path for path in (root / "runs").rglob("seed_*") if path.is_dir()):
        run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        missing = sorted(required_run_entries(run_config) - {path.name for path in run_dir.iterdir()})
        if missing:
            raise AssertionError(f"{run_dir} is missing {missing}")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["exit_status"] != "ok":
            raise AssertionError(f"failed manifest {run_dir}")
        validity = json.loads((run_dir / "runtime_validity.json").read_text(encoding="utf-8"))
        if validity.get("status") != "valid" or validity.get("resume_eligible") is not True:
            raise AssertionError(f"invalid runtime validity record {run_dir}")
        if manifest.get("config_hash") != config_hash(run_config):
            raise AssertionError(f"run config hash mismatch {run_dir}")
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


def _completed_task_status(task: tuple[Any, ...]) -> str | None:
    config, environment, condition, seed, _, options = _task_parts(task)
    run_dir = _task_run_dir(task)
    if _completed_run(run_dir, config, (environment, condition, seed), options):
        return "valid"
    if _completed_invalid_tuning_run(
        run_dir, config, (environment, condition, seed), options
    ):
        return "invalid"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failed", "--retry-invalid", dest="retry_failed", action="store_true"
    )
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
    if config.get("enforce_repository_containment", False):
        repository = repository_root()
        validate_runtime_environment(repository)
        root = contained_path(root, repository, label="result_root")
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
        task_states = [_completed_task_status(task) for task in tasks]
        states = {
            "expected_attempts" if config.get("experiment_stage") == "lr_tune" else "expected_runs": len(tasks),
            "completed_valid_runs": sum(state == "valid" for state in task_states),
            "completed_invalid_attempts": sum(state == "invalid" for state in task_states),
            "pending_runs": sum(not _task_run_dir(task).exists() for task in tasks),
            "existing_incomplete_or_failed_runs": sum(
                _task_run_dir(task).exists() and state is None
                for task, state in zip(tasks, task_states, strict=True)
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
    requested_workers = int(args.workers or config["workers"])
    if requested_workers < 1:
        parser.error("--workers must be positive")
    if is_full:
        safe_max, effective_cpus, quota = safe_worker_limit()
        if requested_workers > safe_max:
            raise RuntimeError(
                f"requested workers={requested_workers} exceeds safe maximum={safe_max}; "
                f"effective_cpus={effective_cpus}, cgroup_quota={quota}"
            )
    workers = min(requested_workers, len(tasks), cpu_count())
    try:
        if workers == 1:
            list(map(run_cross_worker, tasks))
        else:
            with Pool(workers) as pool:
                pool.map(run_cross_worker, tasks)
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
