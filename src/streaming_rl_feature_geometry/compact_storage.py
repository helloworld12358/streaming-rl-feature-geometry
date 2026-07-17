"""Compact per-run storage and online hidden-velocity summaries."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


COMPACT_SCHEMA_VERSION = 2
COMPACT_TRACE_FILES = {
    "strided_trace.npz",
    "decision_event_trace.npz",
    "disturbance_event_trace.npz",
    "prediction_feature_summary.npz",
    "model_state.npz",
}


class ScalarMoments:
    """Constant-memory scalar mean, variance, extrema, and count."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value: float) -> None:
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError("online scalar summary received NaN or Inf")
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n else 0.0


def write_columnar_npz(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write records as compressed typed columns without object arrays."""

    target = Path(path)
    records = [dict(row) for row in rows]
    columns = sorted({key for row in records for key in row})
    payload: dict[str, np.ndarray] = {"columns": np.asarray(columns, dtype="U")}
    for column in columns:
        values = [row.get(column) for row in records]
        if any(isinstance(value, str) for value in values if value is not None):
            payload[f"col__{column}"] = np.asarray(
                ["" if value is None else str(value) for value in values], dtype="U"
            )
        elif any(value is None for value in values):
            payload[f"col__{column}"] = np.asarray(
                [np.nan if value is None else float(value) for value in values],
                dtype=np.float64,
            )
        elif values and all(isinstance(value, (bool, np.bool_)) for value in values):
            payload[f"col__{column}"] = np.asarray(values, dtype=np.bool_)
        elif values and all(isinstance(value, (int, np.integer)) for value in values):
            payload[f"col__{column}"] = np.asarray(values, dtype=np.int64)
        else:
            payload[f"col__{column}"] = np.asarray(values, dtype=np.float64)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
    return target


def read_columnar_npz(path: str | Path) -> pd.DataFrame:
    with np.load(Path(path), allow_pickle=False) as payload:
        columns = [str(value) for value in payload["columns"]]
        return pd.DataFrame({column: payload[f"col__{column}"] for column in columns})


class HiddenVelocityAccumulator:
    """Online authoritative cost/recovery summary with bounded final-window memory."""

    MEAN_FIELDS = (
        "position_cost",
        "velocity_cost",
        "action_cost",
        "control_effort",
        "total_cost",
        "position_squared",
        "velocity_squared",
        "abs_position",
        "abs_velocity",
        "stabilized",
        "boundary_hit",
    )

    def __init__(
        self,
        *,
        environment: str,
        final_window: int,
        settling_consecutive_steps: int,
        recovery_consecutive_steps: int,
    ) -> None:
        self.environment = str(environment)
        self.final_window = deque(maxlen=max(1, int(final_window)))
        self.stats = {field: ScalarMoments() for field in self.MEAN_FIELDS}
        self.maximum_abs_position = 0.0
        self.maximum_abs_velocity = 0.0
        self.time_outside_stable_region = 0
        self.settling_consecutive_steps = int(settling_consecutive_steps)
        self.recovery_consecutive_steps = int(recovery_consecutive_steps)
        self._settling_streak = 0
        self.settling_time = float("nan")
        self._active_disturbance: dict[str, Any] | None = None
        self._recovery_streak = 0
        self.recovery_times: list[float] = []
        self.event_rows: list[dict[str, Any]] = []
        self.post_stats = {
            "position_squared": ScalarMoments(),
            "velocity_squared": ScalarMoments(),
            "stabilized": ScalarMoments(),
        }

    def _close_unrecovered(self) -> None:
        if self._active_disturbance is None:
            return
        self.recovery_times.append(float("nan"))
        self.event_rows.append(
            {
                **self._active_disturbance,
                "event": "recovery_not_reached",
                "recovery_time": np.nan,
            }
        )
        self._active_disturbance = None
        self._recovery_streak = 0

    def update(self, interaction: int, diagnostics: Mapping[str, Any]) -> None:
        row = dict(diagnostics)
        if not np.isclose(float(row["reward"]), -float(row["total_cost"])):
            raise AssertionError("hidden-velocity reward must equal negative total cost")
        for field in self.MEAN_FIELDS:
            self.stats[field].update(float(row[field]))
        self.final_window.append({field: float(row[field]) for field in self.MEAN_FIELDS})
        self.maximum_abs_position = max(self.maximum_abs_position, float(row["abs_position"]))
        self.maximum_abs_velocity = max(self.maximum_abs_velocity, float(row["abs_velocity"]))
        stable = bool(row["stabilized"])
        self.time_outside_stable_region += int(not stable)
        self._settling_streak = self._settling_streak + 1 if stable else 0
        if (
            not np.isfinite(self.settling_time)
            and self._settling_streak >= self.settling_consecutive_steps
        ):
            self.settling_time = float(interaction - self.settling_consecutive_steps + 1)

        if bool(row.get("disturbance_active", False)):
            self._close_unrecovered()
            self._active_disturbance = {
                "interaction_index": int(interaction),
                "disturbance_magnitude": float(row.get("disturbance_magnitude", 0.0)),
            }
            self._recovery_streak = 0
            self.event_rows.append({**self._active_disturbance, "event": "disturbance"})
        if self._active_disturbance is not None:
            self._recovery_streak = self._recovery_streak + 1 if stable else 0
            if self._recovery_streak >= self.recovery_consecutive_steps:
                start = int(self._active_disturbance["interaction_index"])
                recovery = float(
                    interaction - self.recovery_consecutive_steps + 1 - start
                )
                self.recovery_times.append(recovery)
                self.event_rows.append(
                    {
                        **self._active_disturbance,
                        "event": "recovered",
                        "recovery_time": recovery,
                        "recovery_interaction": int(interaction),
                    }
                )
                self._active_disturbance = None
                self._recovery_streak = 0
        if bool(row.get("post_disturbance", False)):
            for field, statistic in self.post_stats.items():
                statistic.update(float(row[field]))

    def finalize(self) -> dict[str, Any]:
        if self.environment == "hidden_velocity_informative":
            self._close_unrecovered()
        final_rows = list(self.final_window)

        def final_mean(field: str) -> float:
            return float(np.mean([row[field] for row in final_rows])) if final_rows else np.nan

        result: dict[str, Any] = {
            "mean_position_cost": self.stats["position_cost"].mean,
            "mean_velocity_cost": self.stats["velocity_cost"].mean,
            "mean_action_cost": self.stats["action_cost"].mean,
            "mean_control_effort": self.stats["control_effort"].mean,
            "mean_total_cost": self.stats["total_cost"].mean,
            "position_rmse": float(np.sqrt(self.stats["position_squared"].mean)),
            "velocity_rmse": float(np.sqrt(self.stats["velocity_squared"].mean)),
            "mean_abs_position": self.stats["abs_position"].mean,
            "mean_abs_velocity": self.stats["abs_velocity"].mean,
            "stabilization_rate": self.stats["stabilized"].mean,
            "boundary_hit_count": int(round(self.stats["boundary_hit"].mean * self.stats["boundary_hit"].n)),
            "boundary_hit_rate": self.stats["boundary_hit"].mean,
            "final_window_position_cost": final_mean("position_cost"),
            "final_window_velocity_cost": final_mean("velocity_cost"),
            "final_window_action_cost": final_mean("action_cost"),
            "final_window_control_effort": final_mean("control_effort"),
            "final_window_total_cost": final_mean("total_cost"),
            "final_window_position_rmse": float(np.sqrt(final_mean("position_squared"))),
            "final_window_velocity_rmse": float(np.sqrt(final_mean("velocity_squared"))),
            "final_window_mean_abs_position": final_mean("abs_position"),
            "final_window_mean_abs_velocity": final_mean("abs_velocity"),
            "final_window_stabilization_rate": final_mean("stabilized"),
            "final_window_boundary_hit_rate": final_mean("boundary_hit"),
            "maximum_abs_position": self.maximum_abs_position,
            "maximum_abs_velocity": self.maximum_abs_velocity,
            "time_outside_stable_region": self.time_outside_stable_region,
            "settling_time": self.settling_time,
            "settling_time_status": "ok" if np.isfinite(self.settling_time) else "not_reached",
            "settling_time_applicable": True,
        }
        if self.environment != "hidden_velocity_informative":
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
        successful = np.asarray(
            [value for value in self.recovery_times if np.isfinite(value)], dtype=np.float64
        )
        count = len(self.recovery_times)
        success_rate = float(len(successful) / count) if count else np.nan
        mean_recovery = float(successful.mean()) if len(successful) else np.nan
        result.update(
            recovery_time=mean_recovery,
            recovery_success=success_rate,
            recovery_time_status=("ok" if len(successful) else "not_reached") if count else "no_disturbance",
            recovery_time_applicable=True,
            disturbance_recovery_time=mean_recovery,
            recovery_success_rate=success_rate,
            post_disturbance_position_rmse=(
                float(np.sqrt(self.post_stats["position_squared"].mean))
                if self.post_stats["position_squared"].n else np.nan
            ),
            post_disturbance_velocity_rmse=(
                float(np.sqrt(self.post_stats["velocity_squared"].mean))
                if self.post_stats["velocity_squared"].n else np.nan
            ),
            post_disturbance_stabilization_rate=(
                self.post_stats["stabilized"].mean
                if self.post_stats["stabilized"].n else np.nan
            ),
        )
        return result
