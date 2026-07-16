"""Common continuing-environment interface for cross-environment experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .env import ContinuingTMaze
from .hidden_velocity import HiddenVelocity, HiddenVelocityInformative
from .ringworld import AliasedRingworld
from .two_loop import AliasedTwoLoop


@dataclass(frozen=True)
class CrossStepInfo:
    """Diagnostic metadata that is never part of the normal agent observation."""

    environment: str
    step: int
    group: int
    phase: str
    position: int | None
    decision: bool
    correct: bool | None
    latent: tuple[float, ...]
    events: tuple[float, ...]
    stabilized: bool | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class StreamingEnvironment(Protocol):
    env_id: str
    event_names: tuple[str, ...]
    n_actions: int

    @property
    def obs_dim(self) -> int: ...

    @property
    def observation(self) -> np.ndarray: ...

    @property
    def oracle_features(self) -> np.ndarray: ...

    @property
    def latent_state(self) -> np.ndarray: ...

    def step(self, action: int) -> tuple[np.ndarray, float, CrossStepInfo]: ...


class TMazeAdapter:
    """Expose the verified stage-A T-maze through the common protocol."""

    env_id = "tmaze"
    event_names = (
        "left_echo",
        "right_echo",
        "junction",
        "positive_reward",
        "negative_reward",
    )
    n_actions = 2

    def __init__(self, corridor_length: int = 5, seed: int = 0) -> None:
        self.inner = ContinuingTMaze(corridor_length=corridor_length, seed=seed)

    @property
    def obs_dim(self) -> int:
        return self.inner.obs_dim

    @property
    def observation(self) -> np.ndarray:
        return self.inner.observation

    @property
    def oracle_features(self) -> np.ndarray:
        return np.asarray([float(self.inner.cue)], dtype=np.float64)

    @property
    def latent_state(self) -> np.ndarray:
        return self.oracle_features

    @property
    def corridor_length(self) -> int:
        return self.inner.corridor_length

    def set_corridor_length(self, corridor_length: int) -> None:
        """Apply the single scheduled E1 change through the verified core guard."""

        self.inner.set_corridor_length(corridor_length)

    def step(self, action: int) -> tuple[np.ndarray, float, CrossStepInfo]:
        next_obs, reward, _, info = self.inner.step(action)
        position = info.corridor_position
        if info.junction:
            position = info.corridor_length + 1
        events = (
            float(next_obs[6]),
            float(next_obs[7]),
            float(next_obs[4]),
            float(reward > 0.0),
            float(reward < 0.0),
        )
        common = CrossStepInfo(
            environment=self.env_id,
            step=self.inner.t,
            group=info.trial,
            phase=info.phase,
            position=position,
            decision=info.junction,
            correct=info.correct,
            latent=(float(info.cue),),
            events=events,
        )
        return next_obs, float(reward), common


ENVIRONMENT_IDS = (
    "tmaze",
    "ringworld",
    "two_loop",
    "hidden_velocity",
    "hidden_velocity_informative",
)


def make_environment(env_id: str, seed: int = 0, **kwargs: object) -> StreamingEnvironment:
    """Construct one registered environment without exposing latent state to it."""

    if env_id == "tmaze":
        return TMazeAdapter(seed=seed, **kwargs)
    if env_id == "ringworld":
        return AliasedRingworld(seed=seed, **kwargs)
    if env_id == "two_loop":
        return AliasedTwoLoop(seed=seed, **kwargs)
    if env_id == "hidden_velocity":
        return HiddenVelocity(seed=seed, **kwargs)
    if env_id == "hidden_velocity_informative":
        return HiddenVelocityInformative(seed=seed, **kwargs)
    raise ValueError(f"unknown environment {env_id!r}; expected one of {ENVIRONMENT_IDS}")
