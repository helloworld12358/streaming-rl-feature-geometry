"""Robust, seed-level condition summaries and preregistered failure rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


ROBUST_METRICS = (
    "final_performance",
    "final_window_reward",
    "final_window_stabilization_rate",
    "final_window_position_rmse",
    "final_window_velocity_rmse",
    "final_window_control_effort",
    "settling_time",
    "recovery_time",
)


def interquartile_mean(values: np.ndarray) -> float:
    """Return the 25%-trimmed mean with fractional boundary observations."""

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    ordered = ordered[np.isfinite(ordered)]
    if not len(ordered):
        return float("nan")
    lower = 0.25 * len(ordered)
    upper = 0.75 * len(ordered)
    total = 0.0
    weight = 0.0
    for index, value in enumerate(ordered):
        overlap = max(0.0, min(index + 1.0, upper) - max(float(index), lower))
        total += overlap * float(value)
        weight += overlap
    return total / weight if weight else float(np.median(ordered))


def robust_statistics(
    values: np.ndarray | pd.Series,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 1729,
) -> dict[str, float | int]:
    """Compute robust summaries from finite seed-level observations."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "mean": float("nan"),
            "sem": float("nan"),
            "median": float("nan"),
            "iqm": float("nan"),
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "quantile_10": float("nan"),
            "quantile_90": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "seed_count": 0,
        }
    rng = np.random.default_rng(seed)
    bootstrap_samples = max(1, int(bootstrap_samples))
    draws = rng.choice(finite, size=(bootstrap_samples, len(finite)), replace=True)
    bootstrap_iqm = np.asarray([interquartile_mean(row) for row in draws])
    sem = float(np.std(finite, ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else float("nan")
    return {
        "mean": float(np.mean(finite)),
        "sem": sem,
        "median": float(np.median(finite)),
        "iqm": interquartile_mean(finite),
        "bootstrap_ci_low": float(np.quantile(bootstrap_iqm, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap_iqm, 0.975)),
        "quantile_10": float(np.quantile(finite, 0.10)),
        "quantile_90": float(np.quantile(finite, 0.90)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "seed_count": int(len(finite)),
    }


def _failure_thresholds(config: Mapping[str, Any], environment: str) -> dict[str, float]:
    specification = config.get("catastrophic_failure", {})
    thresholds: dict[str, float] = {}
    thresholds.update(specification.get("global", {}))
    thresholds.update(specification.get("environments", {}).get(environment, {}))
    return {str(key): float(value) for key, value in thresholds.items()}


def classify_catastrophic_failure(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[bool, str]:
    """Apply fixed config thresholds without inspecting other evaluation seeds."""

    reasons: list[str] = []
    environment = str(row["environment"])
    required = (
        "final_performance",
        "final_window_reward",
        "final_parameter_norm",
        "max_control_update_norm",
    )
    if any(not np.isfinite(float(row.get(metric, np.nan))) for metric in required):
        reasons.append("nonfinite_required_metric")
    if int(row.get("divergence_flag", 0)) != 0:
        reasons.append("divergence_flag")
    for rule, threshold in _failure_thresholds(config, environment).items():
        if rule.endswith("_below"):
            metric = rule[: -len("_below")]
            value = float(row.get(metric, np.nan))
            if np.isfinite(value) and value < threshold:
                reasons.append(f"{metric}<{threshold:g}")
        elif rule.endswith("_above"):
            metric = rule[: -len("_above")]
            value = float(row.get(metric, np.nan))
            if np.isfinite(value) and value > threshold:
                reasons.append(f"{metric}>{threshold:g}")
        else:
            raise ValueError(
                f"catastrophic-failure rule {rule!r} must end in _below or _above"
            )
    return bool(reasons), ";".join(reasons)


def build_robust_summaries(
    summaries: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build long-form robust metrics and seed/condition failure tables."""

    classified = summaries.copy()
    outcomes = [classify_catastrophic_failure(row, config) for row in classified.to_dict("records")]
    classified["catastrophic_failure"] = [int(item[0]) for item in outcomes]
    classified["catastrophic_failure_reason"] = [item[1] for item in outcomes]
    group_columns = ["environment", "condition"]
    for optional in ("experiment_stage", "controller_alpha_mode", "candidate_alpha"):
        if optional in classified.columns and not classified[optional].isna().all():
            group_columns.append(optional)
    bootstrap_samples = int(config.get("robust_bootstrap_samples", 2000))
    bootstrap_seed = int(config.get("robust_bootstrap_seed", 1729))
    robust_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for group_index, (keys, frame) in enumerate(classified.groupby(group_columns, dropna=False)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, keys))
        failure_count = int(frame["catastrophic_failure"].sum())
        failure_rows.append(
            {
                **identity,
                "seed_count": int(len(frame)),
                "catastrophic_failure_count": failure_count,
                "catastrophic_failure_rate": failure_count / len(frame),
                "failure_reasons": ";".join(
                    sorted(filter(None, frame["catastrophic_failure_reason"].unique()))
                ),
            }
        )
        for metric_index, metric in enumerate(ROBUST_METRICS):
            if metric not in frame:
                continue
            statistics = robust_statistics(
                frame[metric],
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed + 1009 * group_index + metric_index,
            )
            robust_rows.append(
                {
                    **identity,
                    "metric": metric,
                    **statistics,
                    "catastrophic_failure_count": failure_count,
                    "catastrophic_failure_rate": failure_count / len(frame),
                }
            )
    return pd.DataFrame(robust_rows), pd.DataFrame(failure_rows), classified
