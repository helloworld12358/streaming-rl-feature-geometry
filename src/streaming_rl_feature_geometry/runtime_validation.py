"""Fail-closed per-run numerical validity tracking for streaming experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def write_json(path: Path, value: Any) -> None:
    """Write validity evidence without importing either experiment runner."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


DEFAULT_EXTREME_FINITE_LIMIT = 1.0e12


@dataclass(frozen=True)
class RuntimeFailure:
    environment: str
    condition: str
    seed: int
    candidate_alpha: float | None
    interaction_index: int
    metric: str
    observed_value: float
    threshold: float
    relevant_feature_norm: float
    parameter_norm: float
    update_norm: float
    transform_state: dict[str, Any]
    error_type: str
    failure_classification: str


class RuntimeValidityError(FloatingPointError):
    """Raised immediately when a run becomes non-finite or scientifically extreme."""

    def __init__(self, failure: RuntimeFailure) -> None:
        self.failure = failure
        super().__init__(
            f"{failure.failure_classification}: {failure.metric}="
            f"{failure.observed_value!r} at interaction {failure.interaction_index} "
            f"(threshold={failure.threshold:g})"
        )

    def __reduce__(self) -> tuple[type[RuntimeValidityError], tuple[RuntimeFailure]]:
        """Rebuild from the structured failure instead of BaseException.args."""

        return type(self), (self.failure,)


def _scalars(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.float64)
    try:
        return np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float64)


class RuntimeValidityTracker:
    """Track numerical state every transition and preserve first-failure evidence."""

    def __init__(
        self,
        *,
        environment: str,
        condition: str,
        seed: int,
        candidate_alpha: float | None,
        run_dir: str | Path,
        extreme_finite_limit: float = DEFAULT_EXTREME_FINITE_LIMIT,
        context_length: int = 4,
    ) -> None:
        if not np.isfinite(extreme_finite_limit) or extreme_finite_limit <= 0:
            raise ValueError("extreme_finite_limit must be finite and positive")
        self.environment = str(environment)
        self.condition = str(condition)
        self.seed = int(seed)
        self.candidate_alpha = None if candidate_alpha is None else float(candidate_alpha)
        self.run_dir = Path(run_dir)
        self.extreme_finite_limit = float(extreme_finite_limit)
        self._context: deque[dict[str, Any]] = deque(maxlen=max(1, int(context_length)))
        self.nan_count = 0
        self.inf_count = 0
        self.divergence_flag = 0
        self.observations = 0
        self.max_abs: dict[str, float] = {}
        self.first_failure: RuntimeFailure | None = None

    @property
    def path(self) -> Path:
        return self.run_dir / "runtime_validity.json"

    def _record_failure(
        self,
        *,
        interaction: int,
        metric: str,
        value: float,
        feature_norm: float,
        parameter_norm: float,
        update_norm: float,
        transform_state: Mapping[str, Any] | None,
        classification: str,
        error_type: str,
    ) -> None:
        failure = RuntimeFailure(
            environment=self.environment,
            condition=self.condition,
            seed=self.seed,
            candidate_alpha=self.candidate_alpha,
            interaction_index=int(interaction),
            metric=str(metric),
            observed_value=float(value),
            threshold=self.extreme_finite_limit,
            relevant_feature_norm=float(feature_norm),
            parameter_norm=float(parameter_norm),
            update_norm=float(update_norm),
            transform_state=dict(transform_state or {}),
            error_type=str(error_type),
            failure_classification=str(classification),
        )
        self.first_failure = failure
        self.divergence_flag = 1
        write_json(
            self.path,
            {
                "status": "invalid",
                "extreme_finite_limit": self.extreme_finite_limit,
                "nan_count": self.nan_count,
                "inf_count": self.inf_count,
                "divergence_flag": self.divergence_flag,
                "first_failure": asdict(failure),
                "preceding_context": list(self._context),
                "resume_eligible": False,
            },
        )
        raise RuntimeValidityError(failure)

    def observe(
        self,
        interaction: int,
        metrics: Mapping[str, Any],
        *,
        feature_norm: float,
        parameter_norm: float,
        update_norm: float,
        transform_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Observe post-update state; raises and writes evidence on first invalid value."""

        self.observations += 1
        compact: dict[str, Any] = {"interaction_index": int(interaction)}
        for metric, raw in metrics.items():
            values = _scalars(raw)
            if values.size == 0:
                continue
            nan_here = int(np.isnan(values).sum())
            inf_here = int(np.isinf(values).sum())
            self.nan_count += nan_here
            self.inf_count += inf_here
            finite = values[np.isfinite(values)]
            maximum = float(np.max(np.abs(finite), initial=0.0))
            self.max_abs[metric] = max(self.max_abs.get(metric, 0.0), maximum)
            compact[metric] = maximum
            if nan_here or inf_here:
                bad = values[~np.isfinite(values)][0]
                self._record_failure(
                    interaction=interaction,
                    metric=metric,
                    value=float(bad),
                    feature_norm=feature_norm,
                    parameter_norm=parameter_norm,
                    update_norm=update_norm,
                    transform_state=transform_state,
                    classification="non_finite_numerical_failure",
                    error_type="FloatingPointError",
                )
            if maximum > self.extreme_finite_limit:
                offending = float(finite[np.argmax(np.abs(finite))])
                self._record_failure(
                    interaction=interaction,
                    metric=metric,
                    value=offending,
                    feature_norm=feature_norm,
                    parameter_norm=parameter_norm,
                    update_norm=update_norm,
                    transform_state=transform_state,
                    classification="finite_extreme_numerical_divergence",
                    error_type="RuntimeValidityError",
                )
        self._context.append(compact)

    def record_exception(
        self,
        interaction: int,
        error: BaseException,
        *,
        feature_norm: float = float("nan"),
        parameter_norm: float = float("nan"),
        update_norm: float = float("nan"),
        transform_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Preserve non-tracker numerical exceptions using the same evidence schema."""

        if self.first_failure is not None:
            return
        value = float("nan")
        self.nan_count += 1
        self._record_failure(
            interaction=interaction,
            metric="runtime_exception",
            value=value,
            feature_norm=feature_norm,
            parameter_norm=parameter_norm,
            update_norm=update_norm,
            transform_state=transform_state,
            classification="runtime_numerical_exception",
            error_type=type(error).__name__,
        )

    def finalize(self) -> dict[str, Any]:
        if self.first_failure is not None:
            raise RuntimeError("cannot finalize an invalid runtime tracker")
        value = {
            "status": "valid",
            "extreme_finite_limit": self.extreme_finite_limit,
            "observations": self.observations,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "divergence_flag": self.divergence_flag,
            "max_abs_metrics": self.max_abs,
            "first_failure": None,
            "resume_eligible": True,
        }
        write_json(self.path, value)
        return value
