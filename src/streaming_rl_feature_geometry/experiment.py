"""Reproducible orchestration for the streaming feature-property study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
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
from .env import ContinuingTMaze
from .gvf import GVFUpdate, TraceGVFBank
from .metrics import ReservoirSampler, cue_separation_metrics
from .reporting import make_figures
from .transforms import FeatureTransform, rep_metrics

CONDITIONS = (
    "observation_only",
    "oracle",
    "trace_only",
    "raw",
    "rms_raw",
    "standardized",
    "decorrelated",
    "whitened",
    "gaussian_moment",
    "unit_sphere",
)
PREDICTIVE_CONDITIONS = set(CONDITIONS) - {"observation_only", "oracle", "trace_only"}
BASELINE_CONDITIONS = set(CONDITIONS) - PREDICTIVE_CONDITIONS
FULL_PROFILES = {"full_stationary", "full_nonstationary"}
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def git_value(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dependency_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "matplotlib", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "profile",
        "seeds",
        "conditions",
        "total_interactions",
        "corridor_length",
        "horizons",
        "gvf_alpha",
        "control_alpha",
        "gamma",
        "lambda",
        "epsilon",
        "transform_eps",
        "cov_update_every",
        "output_dir",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"config is missing required keys: {', '.join(missing)}")
    unknown = set(config["conditions"]) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    if len(config["conditions"]) != len(set(config["conditions"])):
        raise ValueError("conditions must be unique")
    if not config["seeds"] or len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("seeds must be a non-empty unique list")
    if int(config["total_interactions"]) < 1:
        raise ValueError("total_interactions must be positive")

    config.setdefault("gvf_bank", "mixed")
    config.setdefault("trace_dim", 12)
    config.setdefault("transform_min_samples", 64)
    config.setdefault("moment_beta", 0.005)
    config.setdefault("moment_learning_rate", 0.0005)
    config.setdefault("metrics_stride", 25)
    config.setdefault("analysis_burn_in", 500)
    config.setdefault("reservoir_size", 4000)
    config.setdefault("moving_window_trials", 100)
    config.setdefault("final_window_trials", 200)
    config.setdefault("accuracy_threshold", 0.8)
    config.setdefault("workers", 1)
    config.setdefault("nonstationary", False)
    if config["nonstationary"]:
        for key in ("change_point", "corridor_length_after"):
            if key not in config:
                raise ValueError(f"non-stationary config requires {key}")
        if not 0 < int(config["change_point"]) < int(config["total_interactions"]):
            raise ValueError("change_point must be inside the run")
    return config


def controller_features(
    observation: np.ndarray, state: np.ndarray, condition: str, cue: int
) -> np.ndarray:
    pieces = [observation]
    if condition == "oracle":
        pieces.append(np.asarray([float(cue)]))
    elif len(state):
        pieces.append(state)
    pieces.append(np.ones(1, dtype=np.float64))
    return np.concatenate(pieces)


def diagnostic_state(
    observation: np.ndarray,
    trace: np.ndarray,
    transformed: np.ndarray,
    condition: str,
    cue: int,
) -> np.ndarray:
    if condition == "observation_only":
        return observation.copy()
    if condition == "oracle":
        return np.asarray([float(cue)])
    if condition == "trace_only":
        return trace.copy()
    return transformed.copy()


def _manifest_base(
    config: dict[str, Any], condition: str | None = None, seed: int | None = None
) -> dict[str, Any]:
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "exact_command": [Path(sys.executable).name, *sys.argv],
        "profile": config["profile"],
        "condition": condition,
        "seed": seed,
        "python_version": platform.python_version(),
        "dependency_versions": dependency_versions(),
        "operating_system": platform.platform(),
        "hostname": socket.gethostname(),
        "config_hash": config_hash(config),
        "predictive_bank": config.get("gvf_bank", "mixed"),
        "horizons": list(config.get("horizons", [])),
        "interaction_budget": int(config["total_interactions"]),
        "cpu_count": os.cpu_count(),
        "cpu_model": os.environ.get("RL_REMOTE_CPU_MODEL", platform.processor() or "unknown"),
        "gpu_detection": os.environ.get("RL_REMOTE_GPU_DETECTION", "not-recorded"),
        "gpu_backend_used": "none",
        "parallelism": "cpu-process",
    }


def _position_name(env: ContinuingTMaze) -> str | None:
    if env.phase == "junction":
        return "junction"
    if env.phase == "corridor":
        return f"corridor_{env.phase_i:02d}"
    return None


def run_one(task: tuple[dict[str, Any], str, int, str]) -> dict[str, Any]:
    config, condition, seed, root_string = task
    root = Path(root_string)
    run_dir = root / "runs" / condition / f"seed_{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    started = time.time()
    manifest = _manifest_base(config, condition, seed)
    manifest.update(
        {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "exit_status": "running",
            "result_path": str(run_dir.resolve()),
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "config.json", config)
    try:
        summary = _execute_run(config, condition, seed, run_dir, started)
    except Exception as error:
        manifest.update(
            {
                "end_time": datetime.now(timezone.utc).isoformat(),
                "exit_status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(run_dir / "manifest.json", manifest)
        raise
    manifest.update(
        {
            "end_time": datetime.now(timezone.utc).isoformat(),
            "exit_status": "ok",
            "wall_seconds": summary["wall_seconds"],
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "stdout.log").write_text("completed\n", encoding="utf-8")
    return summary


def _execute_run(
    config: dict[str, Any], condition: str, seed: int, run_dir: Path, started: float
) -> dict[str, Any]:
    env = ContinuingTMaze(config["corridor_length"], seed)
    observation = env.observation
    gvf = TraceGVFBank(
        env.obs_dim,
        seed=seed + 17,
        alpha=config["gvf_alpha"],
        gammas=tuple(config["horizons"]),
        trace_dim=config["trace_dim"],
        bank=config["gvf_bank"],
    )
    write_json(run_dir / "gvf_definitions.json", gvf.definition_records())
    transform = None
    if condition in PREDICTIVE_CONDITIONS:
        transform = FeatureTransform(
            condition,
            gvf.d,
            eps=config["transform_eps"],
            update_every=config["cov_update_every"],
            min_samples=config["transform_min_samples"],
            moment_beta=config["moment_beta"],
            moment_learning_rate=config["moment_learning_rate"],
        )

    predictions = gvf.features(observation)
    trace = gvf.trace_state
    transformed = (
        transform.transform(predictions)
        if transform is not None
        else np.empty(0, dtype=np.float64)
    )
    state = trace if condition == "trace_only" else transformed
    features = controller_features(observation, state, condition, env.cue)
    controller = SarsaLambda(
        2,
        len(features),
        seed=seed + 31,
        alpha=config["control_alpha"],
        gamma=config["gamma"],
        lam=config["lambda"],
        epsilon=config["epsilon"],
    )
    action = controller.act(features)

    state_dim = len(diagnostic_state(observation, trace, transformed, condition, env.cue))
    representation_sample = ReservoirSampler(config["reservoir_size"], seed + 101)
    position_samples: dict[str, ReservoirSampler] = {}
    prediction_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    recent_correct: deque[float] = deque(maxlen=config["moving_window_trials"])
    gvf_error_sum = np.zeros(gvf.d, dtype=np.float64)
    gvf_error_count = 0
    controller_errors: list[float] = []
    predictor_update_norms: list[float] = []
    controller_update_norms: list[float] = []
    gvf_norms: list[float] = []
    cumulative_reward = 0.0
    actual_change_point: int | None = None

    for interaction in range(config["total_interactions"]):
        if config["nonstationary"] and interaction == config["change_point"]:
            env.set_corridor_length(config["corridor_length_after"])
            actual_change_point = interaction

        analysis_vector = diagnostic_state(
            observation, trace, transformed, condition, env.cue
        )
        if interaction >= config["analysis_burn_in"]:
            gvf_norms.append(float(np.linalg.norm(predictions)))
            representation_sample.add(analysis_vector, env.cue, env.trial)
            position = _position_name(env)
            if position is not None:
                if position not in position_samples:
                    position_samples[position] = ReservoirSampler(
                        config["reservoir_size"], seed + 500 + len(position_samples)
                    )
                position_samples[position].add(analysis_vector, env.cue, env.trial)

        next_observation, reward, _, info = env.step(action)
        gvf_update = gvf.update(next_observation, reward)
        next_predictions = gvf.features(next_observation)
        next_trace = gvf.trace_state
        next_transformed = (
            transform.transform(next_predictions)
            if transform is not None
            else np.empty(0, dtype=np.float64)
        )
        next_state = next_trace if condition == "trace_only" else next_transformed
        next_features = controller_features(next_observation, next_state, condition, env.cue)
        if not np.isfinite(next_features).all():
            raise FloatingPointError("non-finite controller features")
        next_action = controller.act(next_features)
        control_error, update_norm, parameter_norm = controller.update(
            features, action, reward, next_features, next_action
        )

        cumulative_reward += reward
        gvf_error_sum += np.square(gvf_update.deltas)
        gvf_error_count += 1
        controller_errors.append(control_error)
        predictor_update_norms.append(gvf_update.update_norm)
        controller_update_norms.append(update_norm)

        record = interaction % config["metrics_stride"] == 0 or info.junction
        if record:
            prediction_row: dict[str, Any] = {
                "t": interaction,
                "condition": condition,
                "seed": seed,
                "gvf_td_mse": gvf_update.mse,
                "predictor_update_norm": gvf_update.update_norm,
                "predictor_parameter_norm": gvf_update.parameter_norm,
            }
            for index, name in enumerate(gvf.feature_names):
                prediction_row[f"td_error__{name}"] = gvf_update.deltas[index]
                prediction_row[f"td_squared__{name}"] = gvf_update.deltas[index] ** 2
                prediction_row[f"cumulant__{name}"] = gvf_update.cumulants[index]
            prediction_rows.append(prediction_row)
            update_row = {
                "t": interaction,
                "condition": condition,
                "seed": seed,
                "phase": info.phase,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "control_td_error": control_error,
                "controller_update_norm": update_norm,
                "controller_parameter_norm": parameter_norm,
                "controller_feature_norm": float(np.linalg.norm(features)),
                "nan_count": 0,
                "inf_count": 0,
                "divergence_flag": 0,
            }
            if transform is not None:
                update_row.update(transform.state_metrics())
            update_rows.append(update_row)
        if info.junction:
            correct = float(bool(info.correct))
            recent_correct.append(correct)
            trial_rows.append(
                {
                    "trial": info.trial,
                    "t": interaction,
                    "condition": condition,
                    "seed": seed,
                    "cue": info.cue,
                    "corridor_length": info.corridor_length,
                    "action": action,
                    "correct": correct,
                    "reward": reward,
                    "cumulative_reward": cumulative_reward,
                    "moving_accuracy": float(np.mean(recent_correct)),
                }
            )

        observation = next_observation
        predictions = next_predictions
        trace = next_trace
        transformed = next_transformed
        features = next_features
        action = next_action

    trials = pd.DataFrame(trial_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    updates_frame = pd.DataFrame(update_rows)
    trials.to_csv(run_dir / "per_trial_metrics.csv", index=False)
    predictions_frame.to_csv(run_dir / "prediction_metrics.csv", index=False)
    updates_frame.to_csv(run_dir / "update_metrics.csv", index=False)

    samples, sample_cues, sample_groups = representation_sample.arrays(state_dim)
    representation = rep_metrics(samples)
    representation.update({"condition": condition, "seed": seed})
    if transform is not None:
        representation.update(transform.state_metrics())
    pd.DataFrame([representation]).to_csv(run_dir / "representation_metrics.csv", index=False)

    hidden_rows = []
    junction_features = np.empty((0, state_dim), dtype=np.float64)
    junction_cues = np.empty(0, dtype=int)
    for position, sampler in sorted(position_samples.items()):
        position_features, position_cues, position_groups = sampler.arrays(state_dim)
        row = cue_separation_metrics(
            position_features, position_cues, position_groups, seed=seed + 307
        )
        row.update(
            {
                "condition": condition,
                "seed": seed,
                "position": position,
                "n_samples": len(position_features),
            }
        )
        hidden_rows.append(row)
        if position == "junction":
            junction_features, junction_cues = position_features, position_cues
    pd.DataFrame(hidden_rows).to_csv(run_dir / "hidden_state_metrics.csv", index=False)
    np.savez_compressed(
        run_dir / "junction_features.npz",
        features=junction_features,
        cues=junction_cues,
    )
    np.savez_compressed(run_dir / "gvf_weights.npz", weights=gvf.w)

    final_window = min(config["final_window_trials"], len(trials))
    threshold_rows = trials[
        (trials["trial"] + 1 >= config["moving_window_trials"])
        & (trials["moving_accuracy"] >= config["accuracy_threshold"])
    ]
    time_to_threshold = int(threshold_rows.iloc[0]["t"]) if len(threshold_rows) else -1
    per_gvf_mse = gvf_error_sum / max(gvf_error_count, 1)
    cue_indices = [
        index for index, definition in enumerate(gvf.defs) if "echo" in definition.name
    ]
    outcome_indices = [
        index for index, definition in enumerate(gvf.defs) if "outcome" in definition.name
    ]
    summary: dict[str, Any] = {
        "condition": condition,
        "seed": seed,
        "interactions": int(config["total_interactions"]),
        "n_trials": int(len(trials)),
        "overall_accuracy": float(trials["correct"].mean()),
        "final_window_accuracy": float(trials.tail(final_window)["correct"].mean()),
        "time_to_accuracy_threshold": time_to_threshold,
        "cumulative_reward": float(cumulative_reward),
        "reward_per_1000_steps": float(1000.0 * cumulative_reward / config["total_interactions"]),
        "gvf_td_mse": float(per_gvf_mse.mean()),
        "cue_echo_td_mse": float(per_gvf_mse[cue_indices].mean()),
        "outcome_td_mse": float(per_gvf_mse[outcome_indices].mean()),
        "gvf_feature_norm_variance": float(np.var(gvf_norms)) if gvf_norms else 0.0,
        "control_td_rms": float(np.sqrt(np.mean(np.square(controller_errors)))),
        "control_td_variance": float(np.var(controller_errors)),
        "predictor_update_norm_mean": float(np.mean(predictor_update_norms)),
        "predictor_update_norm_max": float(np.max(predictor_update_norms)),
        "predictor_parameter_norm": float(np.linalg.norm(gvf.w)),
        "mean_update_norm": float(np.mean(controller_update_norms)),
        "p95_update_norm": float(np.quantile(controller_update_norms, 0.95)),
        "max_update_norm": float(np.max(controller_update_norms)),
        "final_parameter_norm": float(np.linalg.norm(controller.w)),
        "nan_count": 0,
        "inf_count": 0,
        "divergence_flag": 0,
        "wall_seconds": float(time.time() - started),
        "run_dir": str(run_dir),
        "run_status": "ok",
    }
    if transform is not None:
        summary.update(transform.state_metrics())
    if config["nonstationary"]:
        summary.update(_adaptation_metrics(trials, actual_change_point, config))
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    write_json(run_dir / "summary.json", summary)
    return summary


def _adaptation_metrics(
    trials: pd.DataFrame, actual_change_point: int | None, config: dict[str, Any]
) -> dict[str, Any]:
    if actual_change_point is None:
        raise RuntimeError("configured non-stationary change was not applied")
    before = trials[trials["t"] < actual_change_point].tail(config["final_window_trials"])
    after = trials[trials["t"] >= actual_change_point].copy()
    pre_accuracy = float(before["correct"].mean())
    post_minimum = float(after["moving_accuracy"].min())
    target = 0.95 * pre_accuracy
    recovered = after[after["moving_accuracy"] >= target]
    recovery_time = int(recovered.iloc[0]["t"] - actual_change_point) if len(recovered) else -1
    return {
        "change_point": actual_change_point,
        "pre_change_accuracy": pre_accuracy,
        "post_change_minimum_accuracy": post_minimum,
        "recovery_time_interactions": recovery_time,
        "final_post_change_accuracy": float(
            after.tail(config["final_window_trials"])["correct"].mean()
        ),
    }


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    summary_files = sorted((root / "runs").glob("*/seed_*/summary.csv"))
    if not summary_files:
        raise ValueError(f"no completed runs found below {root}")
    summaries = _read_many(summary_files)
    representation = _read_many(
        sorted((root / "runs").glob("*/seed_*/representation_metrics.csv"))
    )
    hidden = _read_many(sorted((root / "runs").glob("*/seed_*/hidden_state_metrics.csv")))
    trials = _read_many(sorted((root / "runs").glob("*/seed_*/per_trial_metrics.csv")))
    predictions = _read_many(sorted((root / "runs").glob("*/seed_*/prediction_metrics.csv")))
    updates = _read_many(sorted((root / "runs").glob("*/seed_*/update_metrics.csv")))

    summaries.to_csv(root / "aggregate_summary.csv", index=False)
    representation.to_csv(root / "aggregate_representation.csv", index=False)
    hidden.to_csv(root / "aggregate_hidden_state.csv", index=False)
    numeric_columns = [
        column
        for column in summaries.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    grouped = summaries.groupby("condition", sort=False)[numeric_columns].agg(["mean", "sem"])
    grouped.columns = [f"{column}_{statistic}" for column, statistic in grouped.columns]
    grouped = grouped.reset_index().fillna(0.0)
    grouped.to_csv(root / "condition_summary.csv", index=False)
    write_json(
        root / "summary.json",
        {
            "successful_runs": len(summaries),
            "conditions": grouped.to_dict(orient="records"),
        },
    )
    make_figures(root, summaries, representation, hidden, trials, predictions, updates)
    return summaries, representation


def validate_results(
    root: str | Path, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    root = Path(root)
    summaries = pd.read_csv(root / "aggregate_summary.csv")
    if summaries.empty or not (summaries["run_status"] == "ok").all():
        raise AssertionError("not every run completed successfully")
    base_transform_metrics = {
        "raw_rms",
        "scale_factor",
        "transformed_rms",
        "transform_condition_number",
        "transform_matrix_change_last",
        "transform_matrix_change_mean",
        "transform_matrix_change_max",
        "transform_refreshes",
    }
    extension_transform_metrics = {"active_fraction", "bounded_max_abs"}
    gaussian_transform_metrics = {
        "gaussian_parameter_change_mean",
        "gaussian_parameter_change_max",
        "gaussian_skew_parameter_mean",
        "gaussian_tail_parameter_mean",
        "gaussian_ew_skewness_error",
        "gaussian_ew_kurtosis_error",
    }
    optional_transform_metrics = (
        base_transform_metrics | gaussian_transform_metrics | extension_transform_metrics
    )
    required_summary_metrics = [
        column
        for column in summaries.select_dtypes(include=[np.number]).columns
        if column not in optional_transform_metrics
    ]
    if not np.isfinite(summaries[required_summary_metrics].to_numpy()).all():
        raise AssertionError("non-finite required summary metrics")
    transformed = summaries[~summaries["condition"].isin(BASELINE_CONDITIONS)]
    if not transformed.empty and not np.isfinite(
        transformed[sorted(base_transform_metrics)].to_numpy()
    ).all():
        raise AssertionError("non-finite transform metrics for transformed condition")
    available_extension_metrics = sorted(extension_transform_metrics & set(summaries))
    if available_extension_metrics and not transformed.empty and not np.isfinite(
        transformed[available_extension_metrics].to_numpy()
    ).all():
        raise AssertionError("non-finite extension transform metrics")
    gaussian = summaries[summaries["condition"] == "gaussian_moment"]
    if not gaussian.empty and not np.isfinite(
        gaussian[sorted(gaussian_transform_metrics)].to_numpy()
    ).all():
        raise AssertionError("non-finite Gaussian-inspired transform metrics")
    expected = None
    if config is not None:
        expected = len(config["conditions"]) * len(config["seeds"])
        if len(summaries) != expected:
            raise AssertionError(f"expected {expected} runs, found {len(summaries)}")
    required = {
        "config.json",
        "manifest.json",
        "per_trial_metrics.csv",
        "prediction_metrics.csv",
        "representation_metrics.csv",
        "hidden_state_metrics.csv",
        "update_metrics.csv",
        "summary.json",
        "stdout.log",
        "figures",
    }
    run_dirs = sorted((root / "runs").glob("*/seed_*"))
    for run_dir in run_dirs:
        missing = sorted(required - {path.name for path in run_dir.iterdir()})
        if missing:
            raise AssertionError(f"{run_dir} is missing {missing}")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["exit_status"] != "ok":
            raise AssertionError(f"failed run manifest: {run_dir}")
    if (summaries["n_trials"] <= 0).any():
        raise AssertionError("at least one run has no completed junction trials")
    for path in (root / "aggregate_representation.csv", root / "aggregate_hidden_state.csv"):
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number])
        optional = optional_transform_metrics | {
            column for column in numeric.columns if column.startswith("eigenvalue_")
        }
        required = [column for column in numeric.columns if column not in optional]
        if not np.isfinite(frame[required].to_numpy()).all():
            raise AssertionError(f"non-finite metrics in {path.name}")
        if path.name == "aggregate_representation.csv":
            for _, row in frame.iterrows():
                dimension = int(row["feature_dim"])
                eigenvalues = [row.get(f"eigenvalue_{index:02d}") for index in range(dimension)]
                if not np.isfinite(np.asarray(eigenvalues, dtype=float)).all():
                    raise AssertionError("missing or non-finite in-range covariance eigenvalue")
            transformed = frame[~frame["condition"].isin(BASELINE_CONDITIONS)]
            if not transformed.empty and not np.isfinite(
                transformed[sorted(base_transform_metrics)].to_numpy()
            ).all():
                raise AssertionError("non-finite representation transform metrics")
            gaussian = frame[frame["condition"] == "gaussian_moment"]
            if not gaussian.empty and not np.isfinite(
                gaussian[sorted(gaussian_transform_metrics)].to_numpy()
            ).all():
                raise AssertionError("non-finite Gaussian-inspired representation metrics")
    return {
        "result_dir": str(root.resolve()),
        "runs": len(summaries),
        "expected_runs": expected,
    }


def smoke_assertions(root: str | Path, config: dict[str, Any]) -> None:
    validate_results(root, config)
    root = Path(root)
    summaries = pd.read_csv(root / "aggregate_summary.csv")
    representation = pd.read_csv(root / "aggregate_representation.csv")
    oracle = summaries.query("condition == 'oracle'")["final_window_accuracy"].mean()
    observation = summaries.query("condition == 'observation_only'")[
        "final_window_accuracy"
    ].mean()
    if oracle <= observation:
        raise AssertionError("oracle did not outperform observation-only")
    raw = summaries.query("condition == 'raw'")
    if not (raw["gvf_feature_norm_variance"] > 1e-10).all():
        raise AssertionError("raw GVF features are constant")
    raw_isotropy = representation.query("condition == 'raw'")["isotropy_error"].mean()
    whitened_isotropy = representation.query("condition == 'whitened'")[
        "isotropy_error"
    ].mean()
    if whitened_isotropy >= raw_isotropy:
        raise AssertionError("whitening did not improve the registered isotropy metric")
    required_figures = {
        "trial_accuracy_learning_curve.png",
        "final_accuracy.png",
        "gvf_prediction_error.png",
        "covariance_eigenvalue_spectrum.png",
    }
    existing = {path.name for path in (root / "figures").glob("*.png")}
    if not required_figures <= existing:
        raise AssertionError("smoke run is missing required real figures")


def _new_run_name(profile: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{profile}-{timestamp}-{git_value('rev-parse', '--short', 'HEAD')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--seeds", help="comma-separated seed override")
    parser.add_argument("--interactions", type=int, help="interaction-budget override")
    parser.add_argument("--aggregate-only", action="store_true")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    if arguments.seeds:
        config["seeds"] = [int(item.strip()) for item in arguments.seeds.split(",") if item.strip()]
    if arguments.interactions is not None:
        if arguments.interactions < 1:
            parser.error("--interactions must be positive")
        config["total_interactions"] = arguments.interactions
    if config["profile"] in FULL_PROFILES and not (
        arguments.allow_full_run and os.environ.get("RL_RUN_CONTEXT") == "remote"
    ):
        raise SystemExit(
            "Full profile blocked: both RL_RUN_CONTEXT=remote and --allow-full-run are required."
        )

    if arguments.aggregate_only:
        if not arguments.output_dir:
            parser.error("--aggregate-only requires --output-dir pointing to one run directory")
        aggregate(arguments.output_dir)
        print(json.dumps(validate_results(arguments.output_dir), sort_keys=True))
        return

    run_name = arguments.run_name or _new_run_name(config["profile"])
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        parser.error("--run-name may contain only letters, digits, dot, underscore, and hyphen")
    root = Path(arguments.output_dir or config["output_dir"]) / run_name
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    batch_manifest = _manifest_base(config)
    batch_manifest.update(
        {
            "run_name": run_name,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "exit_status": "running",
        }
    )
    write_json(root / "manifest.json", batch_manifest)

    tasks = [
        (config, condition, int(seed), str(root))
        for condition in config["conditions"]
        for seed in config["seeds"]
    ]
    workers = arguments.workers or config.get("workers") or 1
    workers = max(1, min(int(workers), len(tasks), cpu_count()))
    try:
        if workers == 1:
            list(map(run_one, tasks))
        else:
            with Pool(workers) as pool:
                pool.map(run_one, tasks)
        aggregate(root)
        if config["profile"] == "smoke":
            smoke_assertions(root, config)
        validation = validate_results(root, config)
        batch_manifest.update(
            {
                "exit_status": "ok",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "workers": workers,
                "validation": validation,
            }
        )
        write_json(root / "manifest.json", batch_manifest)
    except Exception as error:
        batch_manifest.update(
            {
                "exit_status": "failed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(root / "manifest.json", batch_manifest)
        raise
    print(f"RESULT_DIR={root.resolve()}")


if __name__ == "__main__":
    main()
