"""Config-driven cross-environment streaming experiment runner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
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
from .cross_env import ENVIRONMENT_IDS, make_environment
from .cross_metrics import DiagnosticReservoir, task_information_metrics
from .cross_reporting import make_cross_figures
from .experiment import config_hash, dependency_versions, git_value, write_json
from .predictive import CausalPredictiveBank
from .priors import MATCHED_PRIOR_FOR_ENV, TaskMatchedTransform
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


def _execute_cross_run(
    config: dict[str, Any], environment: str, condition: str, seed: int, run_dir: Path,
    started: float,
) -> dict[str, Any]:
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
    controller = SarsaLambda(
        env.n_actions,
        len(features),
        seed=seed + 31,
        alpha=env_spec.get("control_alpha", config["control_alpha"]),
        gamma=env_spec.get("gamma", config["gamma"]),
        lam=env_spec.get("lambda", config["lambda"]),
        epsilon=env_spec.get("epsilon", config["epsilon"]),
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
    all_stabilized: list[float] = []
    control_deltas: list[float] = []
    control_updates: list[float] = []
    predictive_updates: list[float] = []
    predictive_mse: list[float] = []
    cumulative_reward = 0.0
    threshold_time = -1
    target_threshold = float(env_spec.get("threshold", 0.8 if environment != "hidden_velocity" else -0.3))

    for interaction in range(int(env_spec["interactions"])):
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
            reservoir.add(analysis_vector, np.asarray(info.latent), info.group)
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
            recent_correct.append(value)
            decision_rows.append(
                {
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
            )
        if info.stabilized is not None:
            all_stabilized.append(float(info.stabilized))

        moving_reward = float(np.mean(recent_rewards))
        moving_performance = (
            float(np.mean(recent_correct)) if recent_correct else moving_reward
        )
        if threshold_time < 0 and len(recent_rewards) == recent_rewards.maxlen:
            if moving_performance >= target_threshold:
                threshold_time = interaction
        record = interaction % config["metrics_stride"] == 0 or info.decision
        if record:
            step_rows.append(
                {
                    "t": interaction,
                    "environment": environment,
                    "condition": condition,
                    "seed": seed,
                    "reward": reward,
                    "cumulative_reward": cumulative_reward,
                    "moving_reward": moving_reward,
                    "moving_performance": moving_performance,
                    "stabilized": float(info.stabilized) if info.stabilized is not None else np.nan,
                }
            )
            prediction_row: dict[str, Any] = {
                "t": interaction,
                "environment": environment,
                "condition": condition,
                "seed": seed,
                "predictive_td_mse": prediction_update.mse,
                "predictive_update_norm": prediction_update.update_norm,
                "predictive_parameter_norm": prediction_update.parameter_norm,
            }
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
                }
            )

        observation, raw, transformed = next_observation, next_raw, next_transformed
        features, action = next_features, next_action

    sample_features, sample_latents, sample_groups = reservoir.arrays(diagnostic_dim, latent_dim)
    representation = rep_metrics(sample_features)
    task_metrics, probe_payload = task_information_metrics(
        environment, sample_features, sample_latents, sample_groups, seed + 700
    )
    representation.update(environment=environment, condition=condition, seed=seed)
    if transform is not None:
        representation.update(transform.state_metrics())
    task_row = dict(task_metrics, environment=environment, condition=condition, seed=seed)
    final_rewards = all_rewards[-config["final_window"] :]
    final_correct = all_correct[-config["final_window"] :]
    final_reward = float(np.mean(final_rewards))
    final_accuracy = float(np.mean(final_correct)) if final_correct else 0.0
    final_performance = final_reward if environment == "hidden_velocity" else final_accuracy
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
        **task_metrics,
    }

    pd.DataFrame(step_rows).to_csv(run_dir / "step_metrics.csv", index=False)
    pd.DataFrame(
        decision_rows,
        columns=(
            "t",
            "environment",
            "condition",
            "seed",
            "group",
            "position",
            "correct",
            "moving_accuracy",
            "reward",
        ),
    ).to_csv(run_dir / "decision_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "prediction_metrics.csv", index=False)
    pd.DataFrame(update_rows).to_csv(run_dir / "update_metrics.csv", index=False)
    pd.DataFrame([representation]).to_csv(run_dir / "representation_metrics.csv", index=False)
    pd.DataFrame([task_row]).to_csv(run_dir / "task_information_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    write_json(run_dir / "summary.json", summary)
    np.savez_compressed(
        run_dir / "diagnostic_samples.npz",
        features=sample_features,
        latents=sample_latents,
        groups=sample_groups,
        **probe_payload,
    )
    np.save(run_dir / "predictive_weights.npy", bank.w)
    (run_dir / "stdout.log").write_text(
        f"completed environment={environment} condition={condition} seed={seed}\n",
        encoding="utf-8",
    )
    return summary


def run_cross_one(task: tuple[dict[str, Any], str, str, int, str]) -> dict[str, Any]:
    config, environment, condition, seed, root_string = task
    root = Path(root_string)
    run_dir = root / "runs" / environment / condition / f"seed_{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    started = time.time()
    manifest = _manifest_base(config, environment, condition, seed)
    env_spec = config["environments"][environment]
    manifest.update(
        start_time=datetime.now(timezone.utc).isoformat(),
        exit_status="running",
        bank=env_spec.get("bank", "mixed"),
        horizon=config["horizons"],
        interaction_budget=env_spec["interactions"],
    )
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "config.json", config)
    try:
        summary = _execute_cross_run(config, environment, condition, seed, run_dir, started)
    except Exception as error:
        manifest.update(
            end_time=datetime.now(timezone.utc).isoformat(),
            exit_status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        write_json(run_dir / "manifest.json", manifest)
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


def aggregate_cross(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    summaries = _read_many(sorted((root / "runs").glob("*/*/seed_*/summary.csv")))
    representation = _read_many(
        sorted((root / "runs").glob("*/*/seed_*/representation_metrics.csv"))
    )
    task = _read_many(sorted((root / "runs").glob("*/*/seed_*/task_information_metrics.csv")))
    steps = _read_many(sorted((root / "runs").glob("*/*/seed_*/step_metrics.csv")))
    decisions = _read_many(sorted((root / "runs").glob("*/*/seed_*/decision_metrics.csv")))
    updates = _read_many(sorted((root / "runs").glob("*/*/seed_*/update_metrics.csv")))
    summaries.to_csv(root / "aggregate_summary.csv", index=False)
    representation.to_csv(root / "aggregate_representation.csv", index=False)
    task.to_csv(root / "aggregate_task_information.csv", index=False)
    steps.to_csv(root / "aggregate_steps.csv", index=False)
    decisions.to_csv(root / "aggregate_decisions.csv", index=False)
    updates.to_csv(root / "aggregate_updates.csv", index=False)
    numeric = [
        column for column in summaries.select_dtypes(include=[np.number]).columns if column != "seed"
    ]
    grouped = summaries.groupby(["environment", "condition"], sort=False)[numeric].agg(["mean", "sem"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    grouped.reset_index().fillna(0.0).to_csv(root / "condition_summary.csv", index=False)
    make_cross_figures(root, summaries, representation, task, steps, decisions, updates)
    return summaries, representation


def validate_cross_results(root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    summaries = pd.read_csv(root / "aggregate_summary.csv")
    expected = sum(
        len(spec["conditions"]) * len(config["seeds"])
        for spec in config["environments"].values()
    )
    if len(summaries) != expected:
        raise AssertionError(f"expected {expected} cross runs, found {len(summaries)}")
    if summaries.duplicated(["environment", "condition", "seed"]).any():
        raise AssertionError("duplicate environment/condition/seed summaries")
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
    for run_dir in sorted((root / "runs").glob("*/*/seed_*")):
        required = {
            "config.json",
            "manifest.json",
            "predictive_definitions.json",
            "step_metrics.csv",
            "decision_metrics.csv",
            "prediction_metrics.csv",
            "update_metrics.csv",
            "representation_metrics.csv",
            "task_information_metrics.csv",
            "summary.csv",
            "summary.json",
            "diagnostic_samples.npz",
            "predictive_weights.npy",
            "stdout.log",
            "figures",
        }
        missing = sorted(required - {path.name for path in run_dir.iterdir()})
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    config = load_cross_config(args.config)
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        parser.error("--run-name may contain only letters, digits, dot, underscore, and hyphen")
    is_full = bool(config.get("remote_full", False) or "full" in config["profile"])
    if is_full and not (args.allow_full_run and os.environ.get("RL_RUN_CONTEXT") == "remote"):
        raise SystemExit(
            "Cross-environment full profile blocked: RL_RUN_CONTEXT=remote and --allow-full-run are both required."
        )
    root = Path(args.output_dir or config["output_dir"]) / args.run_name
    if args.aggregate_only:
        aggregate_cross(root)
        print(json.dumps(validate_cross_results(root, config), sort_keys=True))
        return
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    manifest = _manifest_base(config)
    manifest.update(
        run_name=args.run_name,
        start_time=datetime.now(timezone.utc).isoformat(),
        exit_status="running",
    )
    write_json(root / "manifest.json", manifest)
    tasks = [
        (config, environment, condition, int(seed), str(root))
        for environment, spec in config["environments"].items()
        for condition in spec["conditions"]
        for seed in config["seeds"]
    ]
    workers = max(1, min(args.workers or config["workers"], len(tasks), cpu_count()))
    try:
        if workers == 1:
            list(map(run_cross_one, tasks))
        else:
            with Pool(workers) as pool:
                pool.map(run_cross_one, tasks)
        aggregate_cross(root)
        if config["profile"] == "cross_smoke":
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
