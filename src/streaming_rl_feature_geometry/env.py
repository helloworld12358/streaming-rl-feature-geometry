"""Continuing partially observable T-maze used by the experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_LEFT, A_RIGHT = 0, 1


@dataclass(frozen=True)
class StepInfo:
    """Metadata for the state on which an action was selected."""

    trial: int
    phase: str
    cue: int
    junction: bool
    outcome: bool
    correct: bool | None
    next_phase: str
    corridor_position: int | None
    corridor_length: int


class ContinuingTMaze:
    """A continuing T-maze with a cue that is hidden at the junction.

    A trial visits ``cue -> corridor... -> junction -> outcome``.  The
    junction observation is identical for both cues.  The outcome observation,
    returned only after the junction action, contains a delayed cue echo that a
    predictive learner can use as a training cumulant.  Taking an action from
    the outcome state starts the next trial without resetting any agent state.
    """

    names = (
        "bias_obs",
        "cue_left",
        "cue_right",
        "corridor",
        "junction",
        "outcome",
        "echo_left",
        "echo_right",
    )

    def __init__(self, corridor_length: int = 4, seed: int = 0) -> None:
        if corridor_length < 1:
            raise ValueError("corridor_length must be at least 1")
        self.corridor_length = int(corridor_length)
        self.rng = np.random.default_rng(seed)
        self.trial = -1
        self.t = 0
        self.phase_i = 0
        self.cue = -1
        self.last_correct: bool | None = None
        self._start_trial()

    @property
    def obs_dim(self) -> int:
        return len(self.names)

    @property
    def phase(self) -> str:
        if self.phase_i == 0:
            return "cue"
        if 1 <= self.phase_i <= self.corridor_length:
            return "corridor"
        if self.phase_i == self.corridor_length + 1:
            return "junction"
        return "outcome"

    @property
    def observation(self) -> np.ndarray:
        return self._obs()

    def _start_trial(self) -> None:
        self.trial += 1
        self.phase_i = 0
        self.cue = int(self.rng.choice((-1, 1)))
        self.last_correct = None

    def reset_stream(self) -> np.ndarray:
        """Start a new trial; intended for diagnostics, not normal training."""

        self._start_trial()
        return self.observation

    def set_corridor_length(self, corridor_length: int) -> None:
        """Apply the single configured non-stationary change at a trial start."""

        if corridor_length < 1:
            raise ValueError("corridor_length must be at least 1")
        if self.phase != "cue" or self.phase_i != 0:
            raise RuntimeError("corridor length may change only at the start of a trial")
        self.corridor_length = int(corridor_length)

    def _phase(self) -> str:
        """Compatibility shim for older experiment code."""

        return self.phase

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float64)
        obs[0] = 1.0
        if self.phase == "cue":
            obs[1 if self.cue < 0 else 2] = 1.0
        elif self.phase == "corridor":
            obs[3] = 1.0
        elif self.phase == "junction":
            obs[4] = 1.0
        else:
            obs[5] = 1.0
            obs[6 if self.cue < 0 else 7] = 1.0
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        if int(action) not in (A_LEFT, A_RIGHT):
            raise ValueError(f"invalid action {action!r}; expected 0 or 1")

        phase = self.phase
        trial = self.trial
        cue = self.cue
        corridor_position = self.phase_i if phase == "corridor" else None
        reward = 0.0
        correct: bool | None = None
        if phase == "junction":
            correct_action = A_LEFT if cue < 0 else A_RIGHT
            correct = int(action) == correct_action
            self.last_correct = correct
            reward = 1.0 if correct else -1.0

        if phase == "outcome":
            self._start_trial()
        else:
            self.phase_i += 1

        self.t += 1
        info = StepInfo(
            trial=trial,
            phase=phase,
            cue=cue,
            junction=phase == "junction",
            outcome=phase == "outcome",
            correct=correct,
            next_phase=self.phase,
            corridor_position=corridor_position,
            corridor_length=self.corridor_length,
        )
        return self.observation, reward, False, info
