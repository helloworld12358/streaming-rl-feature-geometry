"""Environment-agnostic fixed-trace linear GVFs for the extension study."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


COMPACT_EVENTS = {
    "tmaze": ("left_echo", "right_echo", "junction"),
    "ringworld": ("landmark_a", "landmark_b", "decision"),
    "two_loop": ("identity_cue_0", "identity_cue_1", "landmark", "decision"),
    "hidden_velocity": ("position_left", "position_right", "near_center"),
    "hidden_velocity_informative": ("position_left", "position_right", "near_center"),
}


@dataclass(frozen=True)
class PredictiveDefinition:
    name: str
    cumulant: str
    continuation: float
    effective_horizon: float
    interpretation: str


@dataclass(frozen=True)
class PredictiveUpdate:
    deltas: np.ndarray
    cumulants: np.ndarray
    mse: float
    update_norm: float
    parameter_norm: float


class CausalPredictiveBank:
    """One-pass TD(0) predictions from a deterministic leaky trace state."""

    def __init__(
        self,
        environment: str,
        obs_dim: int,
        event_names: tuple[str, ...],
        seed: int = 0,
        alpha: float = 0.006,
        continuations: tuple[float, ...] = (0.65, 0.9),
        trace_dim: int = 16,
        bank: str = "mixed",
    ) -> None:
        if obs_dim < 1 or trace_dim < obs_dim or alpha <= 0:
            raise ValueError("invalid predictive-bank dimensions or step size")
        if bank not in {"compact", "mixed"}:
            raise ValueError("bank must be compact or mixed")
        if not continuations or any(not 0 <= value < 1 for value in continuations):
            raise ValueError("continuations must lie in [0, 1)")
        event_names = tuple(event_names)
        selected = COMPACT_EVENTS.get(environment, event_names) if bank == "compact" else event_names
        if not set(selected) <= set(event_names):
            raise ValueError("compact event definition is incompatible with environment")

        self.environment = environment
        self.obs_dim = int(obs_dim)
        self.event_names = event_names
        self.event_indices = tuple(event_names.index(name) for name in selected)
        self.bank = bank
        self.alpha = float(alpha)
        self.defs = tuple(
            PredictiveDefinition(
                name=f"{event}_g{continuation:g}",
                cumulant=event,
                continuation=float(continuation),
                effective_horizon=float(1.0 / (1.0 - continuation)),
                interpretation=f"discounted future occupancy/value of {event}",
            )
            for continuation in continuations
            for event in selected
        )
        self.d = len(self.defs)
        self.feature_names = tuple(definition.name for definition in self.defs)
        self.trace_dim = int(trace_dim)
        self.rhos = np.linspace(0.25, 0.97, self.trace_dim)
        rng = np.random.default_rng(seed)
        self.B = rng.normal(0.0, 1.0 / np.sqrt(obs_dim), (self.trace_dim, obs_dim))
        self.B[:obs_dim] = np.eye(obs_dim)
        self.m = np.zeros(self.trace_dim, dtype=np.float64)
        self.input_dim = obs_dim + self.trace_dim + 1
        self.w = np.zeros((self.d, self.input_dim), dtype=np.float64)
        self.current_x: np.ndarray | None = None
        self.current_predictions: np.ndarray | None = None
        self.awaiting_update = False

    @property
    def trace_state(self) -> np.ndarray:
        return self.m.copy()

    def definition_records(self) -> list[dict[str, object]]:
        return [asdict(definition) for definition in self.defs]

    def _project(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (self.obs_dim,):
            raise ValueError(f"expected observation shape {(self.obs_dim,)}, got {observation.shape}")
        drive = self.B @ observation
        next_trace = self.rhos * self.m + (1.0 - self.rhos) * drive
        state = np.concatenate((observation, next_trace, [1.0]))
        return next_trace, state

    def features(self, observation: np.ndarray) -> np.ndarray:
        if self.awaiting_update:
            raise RuntimeError("previous transition must be updated exactly once")
        self.m, self.current_x = self._project(observation)
        self.current_predictions = self.w @ self.current_x
        self.awaiting_update = True
        if not np.isfinite(self.current_predictions).all():
            raise FloatingPointError("non-finite predictive features")
        return self.current_predictions.copy()

    def preview(self, observation: np.ndarray) -> np.ndarray:
        _, state = self._project(observation)
        return self.w @ state

    def update(self, next_observation: np.ndarray, event_values: tuple[float, ...]) -> PredictiveUpdate:
        if self.current_x is None or self.current_predictions is None or not self.awaiting_update:
            raise RuntimeError("features must precede exactly one update")
        events = np.asarray(event_values, dtype=np.float64)
        if events.shape != (len(self.event_names),):
            raise ValueError("event vector does not match registered event names")
        _, next_state = self._project(next_observation)
        next_predictions = self.w @ next_state
        selected = events[list(self.event_indices)]
        repeats = len(self.defs) // len(selected)
        cumulants = np.tile(selected, repeats)
        continuations = np.asarray([definition.continuation for definition in self.defs])
        deltas = cumulants + continuations * next_predictions - self.current_predictions
        change = self.alpha * deltas[:, None] * self.current_x[None, :]
        self.w += change
        self.awaiting_update = False
        if not np.isfinite(self.w).all():
            raise FloatingPointError("non-finite predictive parameters")
        return PredictiveUpdate(
            deltas=deltas.copy(),
            cumulants=cumulants.copy(),
            mse=float(np.mean(deltas**2)),
            update_norm=float(np.linalg.norm(change)),
            parameter_norm=float(np.linalg.norm(self.w)),
        )
