"""Aliased continuing Ringworld with a phase-dependent action pattern."""

from __future__ import annotations

import numpy as np


class AliasedRingworld:
    """A deterministic cycle whose two decision states share one observation.

    Position advances clockwise independently of the binary response action.
    At the two decision phases the correct response differs. Landmarks at
    phases 0 and N/2 make phase inferable from causal history but do not reveal
    the current decision identity.
    """

    env_id = "ringworld"
    event_names = (
        "landmark_a",
        "landmark_b",
        "decision",
        "positive_reward",
        "negative_reward",
    )
    n_actions = 2
    observation_names = ("bias", "generic", "landmark_a", "landmark_b", "decision")

    def __init__(self, size: int = 12, seed: int = 0) -> None:
        if size < 8 or size % 4:
            raise ValueError("Ringworld size must be a multiple of four and at least eight")
        self.size = int(size)
        self.rng = np.random.default_rng(seed)
        self.position = int(self.rng.integers(self.size))
        self.t = 0
        self.cycle = 0
        self.decision_positions = (self.size // 4, 3 * self.size // 4)

    @property
    def obs_dim(self) -> int:
        return len(self.observation_names)

    @property
    def observation(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float64)
        obs[0] = 1.0
        if self.position == 0:
            obs[2] = 1.0
        elif self.position == self.size // 2:
            obs[3] = 1.0
        elif self.position in self.decision_positions:
            obs[4] = 1.0
        else:
            obs[1] = 1.0
        return obs

    @property
    def phase_angle(self) -> float:
        return float(2.0 * np.pi * self.position / self.size)

    @property
    def oracle_features(self) -> np.ndarray:
        angle = self.phase_angle
        return np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)

    @property
    def latent_state(self) -> np.ndarray:
        return np.asarray([float(self.position), self.phase_angle], dtype=np.float64)

    def step(self, action: int):
        from .cross_env import CrossStepInfo

        if int(action) not in (0, 1):
            raise ValueError("Ringworld action must be 0 or 1")
        old_position = self.position
        decision = old_position in self.decision_positions
        correct: bool | None = None
        reward = 0.0
        if decision:
            correct_action = 0 if old_position == self.decision_positions[0] else 1
            correct = int(action) == correct_action
            reward = 1.0 if correct else -1.0

        self.position = (self.position + 1) % self.size
        if self.position == 0:
            self.cycle += 1
        self.t += 1
        next_obs = self.observation
        events = (
            float(next_obs[2]),
            float(next_obs[3]),
            float(next_obs[4]),
            float(reward > 0.0),
            float(reward < 0.0),
        )
        info = CrossStepInfo(
            environment=self.env_id,
            step=self.t,
            group=self.cycle,
            phase=f"phase_{old_position:02d}",
            position=old_position,
            decision=decision,
            correct=correct,
            latent=(float(old_position), float(2.0 * np.pi * old_position / self.size)),
            events=events,
        )
        return next_obs, reward, info
