"""Continuing two-loop aliased POMDP with identity and phase memory."""

from __future__ import annotations

import numpy as np


class AliasedTwoLoop:
    """Two recurrent loops with brief identity cues and aliased decisions."""

    env_id = "two_loop"
    event_names = (
        "identity_cue_0",
        "identity_cue_1",
        "landmark",
        "decision",
        "positive_reward",
        "negative_reward",
    )
    n_actions = 2
    observation_names = (
        "bias",
        "generic",
        "identity_cue_0",
        "identity_cue_1",
        "landmark",
        "decision",
    )

    def __init__(self, loop_lengths: tuple[int, int] = (8, 10), seed: int = 0) -> None:
        if len(loop_lengths) != 2 or min(loop_lengths) < 6 or loop_lengths[0] == loop_lengths[1]:
            raise ValueError("two distinct loop lengths of at least six are required")
        self.loop_lengths = tuple(map(int, loop_lengths))
        self.rng = np.random.default_rng(seed)
        self.identity = int(self.rng.integers(2))
        self.phase = 0
        self.segment = 0
        self.t = 0

    @property
    def loop_length(self) -> int:
        return self.loop_lengths[self.identity]

    @property
    def obs_dim(self) -> int:
        return len(self.observation_names)

    @property
    def observation(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float64)
        obs[0] = 1.0
        if self.phase == 0:
            obs[2 + self.identity] = 1.0
        elif self.phase == self.loop_length // 3:
            obs[4] = 1.0
        elif self.phase == self.loop_length // 2:
            obs[5] = 1.0
        else:
            obs[1] = 1.0
        return obs

    @property
    def phase_angle(self) -> float:
        return float(2.0 * np.pi * self.phase / self.loop_length)

    @property
    def oracle_features(self) -> np.ndarray:
        angle = self.phase_angle
        identity = np.asarray([self.identity == 0, self.identity == 1], dtype=np.float64)
        return np.concatenate((identity, [np.cos(angle), np.sin(angle)]))

    @property
    def latent_state(self) -> np.ndarray:
        return np.asarray(
            [float(self.identity), float(self.phase), self.phase_angle], dtype=np.float64
        )

    def step(self, action: int):
        from .cross_env import CrossStepInfo

        if int(action) not in (0, 1):
            raise ValueError("two-loop action must be 0 or 1")
        old_identity = self.identity
        old_phase = self.phase
        old_length = self.loop_length
        decision = old_phase == old_length // 2
        correct: bool | None = None
        reward = 0.0
        if decision:
            correct = int(action) == old_identity
            reward = 1.0 if correct else -1.0

        self.phase += 1
        if self.phase >= old_length:
            self.phase = 0
            self.segment += 1
            self.identity = int(self.rng.integers(2))
        self.t += 1
        next_obs = self.observation
        events = (
            float(next_obs[2]),
            float(next_obs[3]),
            float(next_obs[4]),
            float(next_obs[5]),
            float(reward > 0.0),
            float(reward < 0.0),
        )
        info = CrossStepInfo(
            environment=self.env_id,
            step=self.t,
            group=self.segment,
            phase=f"loop_{old_identity}_phase_{old_phase:02d}",
            position=old_phase,
            decision=decision,
            correct=correct,
            latent=(
                float(old_identity),
                float(old_phase),
                float(2.0 * np.pi * old_phase / old_length),
            ),
            events=events,
        )
        return next_obs, reward, info
