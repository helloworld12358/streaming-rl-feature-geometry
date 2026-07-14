"""Strictly online predictive features built from a fixed trace state."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GVFDefinition:
    name: str
    cumulant: str
    observation_index: int | None
    continuation: float
    effective_horizon: float
    interpretation: str


@dataclass(frozen=True)
class GVFUpdate:
    deltas: np.ndarray
    cumulants: np.ndarray
    mse: float
    update_norm: float
    parameter_norm: float


class TraceGVFBank:
    """A non-deep bank of linear TD(0) general value functions.

    The recurrent carrier is a fixed set of leaky observation traces. Only the
    linear GVF readouts learn. Positive/negative outcome cumulants use the
    reward observed on the current transition, after the junction action.
    """

    mixed_targets = (
        ("left_echo", "observation", 6, "discounted future left-cue echo"),
        ("right_echo", "observation", 7, "discounted future right-cue echo"),
        ("junction", "observation", 4, "discounted future junction occupancy"),
        ("positive_outcome", "positive_reward", None, "discounted future positive outcome"),
        ("negative_outcome", "negative_reward", None, "discounted future negative outcome"),
    )
    cue_only_targets = mixed_targets[:2]

    def __init__(
        self,
        obs_dim: int,
        seed: int = 0,
        alpha: float = 0.01,
        gammas: tuple[float, ...] = (0.60, 0.90),
        trace_dim: int = 12,
        bank: str = "mixed",
    ) -> None:
        if obs_dim < 1:
            raise ValueError("obs_dim must be positive")
        if not gammas or any(not 0.0 <= gamma < 1.0 for gamma in gammas):
            raise ValueError("all GVF continuations must be in [0, 1)")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if bank not in {"mixed", "cue_only"}:
            raise ValueError("bank must be 'mixed' or 'cue_only'")

        self.obs_dim = int(obs_dim)
        self.alpha = float(alpha)
        self.bank = bank
        targets = self.mixed_targets if bank == "mixed" else self.cue_only_targets
        self.defs = tuple(
            GVFDefinition(
                name=f"{name}_g{gamma:g}",
                cumulant=cumulant,
                observation_index=index,
                continuation=float(gamma),
                effective_horizon=float(1.0 / (1.0 - gamma)),
                interpretation=interpretation,
            )
            for gamma in gammas
            for name, cumulant, index, interpretation in targets
        )
        self.d = len(self.defs)
        self.feature_names = tuple(definition.name for definition in self.defs)

        self.trace_dim = max(int(trace_dim), self.obs_dim)
        rng = np.random.default_rng(seed)
        self.rhos = np.linspace(0.35, 0.95, self.trace_dim)
        self.B = rng.normal(
            0.0, 1.0 / np.sqrt(self.obs_dim), size=(self.trace_dim, self.obs_dim)
        )
        self.B[: self.obs_dim] = np.eye(self.obs_dim)

        self.input_dim = self.obs_dim + self.trace_dim + 1
        self.w = np.zeros((self.d, self.input_dim), dtype=np.float64)
        self.m = np.zeros(self.trace_dim, dtype=np.float64)
        self.current_x: np.ndarray | None = None
        self.current_predictions: np.ndarray | None = None
        self.awaiting_update = False

    @property
    def trace_state(self) -> np.ndarray:
        return self.m.copy()

    def definition_records(self) -> list[dict[str, object]]:
        return [asdict(definition) for definition in self.defs]

    def _project(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"expected observation shape {(self.obs_dim,)}, got {obs.shape}")
        drive = self.B @ obs
        next_m = self.rhos * self.m + (1.0 - self.rhos) * drive
        x = np.concatenate((obs, next_m, np.ones(1, dtype=np.float64)))
        return next_m, x

    def preview(self, obs: np.ndarray) -> np.ndarray:
        """Predict for the next observation without changing recurrent state."""

        _, next_x = self._project(obs)
        return self.w @ next_x

    def features(self, obs: np.ndarray) -> np.ndarray:
        """Consume exactly one observation and return current predictions."""

        if self.awaiting_update:
            raise RuntimeError("the current transition has not been updated")
        self.m, self.current_x = self._project(obs)
        self.current_predictions = self.w @ self.current_x
        if not np.isfinite(self.current_predictions).all():
            raise FloatingPointError("non-finite GVF prediction")
        self.awaiting_update = True
        return self.current_predictions.copy()

    def _cumulants(self, next_obs: np.ndarray, reward: float) -> np.ndarray:
        values = []
        for definition in self.defs:
            if definition.cumulant == "observation":
                assert definition.observation_index is not None
                value = float(next_obs[definition.observation_index])
            elif definition.cumulant == "positive_reward":
                value = float(reward > 0.0)
            elif definition.cumulant == "negative_reward":
                value = float(reward < 0.0)
            else:
                raise RuntimeError(f"unsupported cumulant {definition.cumulant}")
            values.append(value)
        return np.asarray(values, dtype=np.float64)

    def update(self, next_obs: np.ndarray, reward: float) -> GVFUpdate:
        """Apply one TD(0) update using the cached current state."""

        if (
            self.current_x is None
            or self.current_predictions is None
            or not self.awaiting_update
        ):
            raise RuntimeError("features(obs) must be called before update(next_obs, reward)")
        _, next_x = self._project(next_obs)
        next_predictions = self.w @ next_x
        cumulants = self._cumulants(next_obs, reward)
        continuations = np.asarray(
            [definition.continuation for definition in self.defs], dtype=np.float64
        )
        deltas = cumulants + continuations * next_predictions - self.current_predictions
        parameter_update = self.alpha * deltas[:, None] * self.current_x[None, :]
        self.w += parameter_update
        self.awaiting_update = False
        if not np.isfinite(self.w).all():
            raise FloatingPointError("non-finite GVF parameters")
        return GVFUpdate(
            deltas=deltas.copy(),
            cumulants=cumulants,
            mse=float(np.mean(np.square(deltas))),
            update_norm=float(np.linalg.norm(parameter_update)),
            parameter_norm=float(np.linalg.norm(self.w)),
        )
