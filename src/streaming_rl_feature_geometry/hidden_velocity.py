"""Bounded continuing position control with unobserved velocity."""

from __future__ import annotations

import numpy as np


class HiddenVelocity:
    """A one-dimensional linear system with physical reflecting boundaries."""

    env_id = "hidden_velocity"
    event_names = (
        "position_left",
        "position_right",
        "near_center",
        "low_cost",
        "high_cost",
    )
    n_actions = 3
    observation_names = ("bias", "position", "position_left", "position_right")

    def __init__(
        self,
        seed: int = 0,
        rho: float = 0.9,
        acceleration: float = 0.12,
        noise_std: float = 0.015,
        position_limit: float = 2.5,
        velocity_limit: float = 0.6,
    ) -> None:
        if not 0.0 <= rho < 1.0 or acceleration <= 0 or noise_std < 0:
            raise ValueError("invalid hidden-velocity dynamics")
        if position_limit <= 0 or velocity_limit <= 0:
            raise ValueError("state limits must be positive")
        self.rng = np.random.default_rng(seed)
        self.rho = float(rho)
        self.acceleration = float(acceleration)
        self.noise_std = float(noise_std)
        self.position_limit = float(position_limit)
        self.velocity_limit = float(velocity_limit)
        self.x = float(self.rng.uniform(-1.5, 1.5))
        self.v = float(self.rng.uniform(-0.25, 0.25))
        self.t = 0

    @property
    def obs_dim(self) -> int:
        return len(self.observation_names)

    @property
    def observation(self) -> np.ndarray:
        return np.asarray(
            [1.0, self.x / self.position_limit, float(self.x < 0.0), float(self.x >= 0.0)],
            dtype=np.float64,
        )

    @property
    def oracle_features(self) -> np.ndarray:
        return np.asarray(
            [self.x / self.position_limit, self.v / self.velocity_limit], dtype=np.float64
        )

    @property
    def latent_state(self) -> np.ndarray:
        return np.asarray([self.x, self.v], dtype=np.float64)

    def _reflect(self, position: float, velocity: float) -> tuple[float, float]:
        """Reflect at documented physical walls; this is environment dynamics."""

        while abs(position) > self.position_limit:
            if position > self.position_limit:
                position = 2.0 * self.position_limit - position
                velocity = -abs(velocity)
            elif position < -self.position_limit:
                position = -2.0 * self.position_limit - position
                velocity = abs(velocity)
        if abs(velocity) > self.velocity_limit:
            velocity = np.sign(velocity) * (
                2.0 * self.velocity_limit - min(abs(velocity), 2.0 * self.velocity_limit)
            )
        return float(position), float(velocity)

    def step(self, action: int):
        from .cross_env import CrossStepInfo

        if int(action) not in (0, 1, 2):
            raise ValueError("hidden-velocity action must be 0, 1, or 2")
        old_x, old_v = self.x, self.v
        acceleration = (int(action) - 1) * self.acceleration
        noise = float(self.rng.normal(0.0, self.noise_std))
        next_v = self.rho * self.v + acceleration + noise
        next_x = self.x + next_v
        self.x, self.v = self._reflect(next_x, next_v)
        action_cost = 0.02 * (int(action) - 1) ** 2
        cost = self.x**2 + 0.25 * self.v**2 + action_cost
        reward = -float(cost)
        stabilized = bool(abs(self.x) < 0.4 and abs(self.v) < 0.12)
        self.t += 1
        next_obs = self.observation
        events = (
            float(self.x < -0.25),
            float(self.x > 0.25),
            float(abs(self.x) <= 0.25),
            float(cost < 0.25),
            float(cost > 1.0),
        )
        info = CrossStepInfo(
            environment=self.env_id,
            step=self.t,
            group=self.t // 32,
            phase="continuous",
            position=None,
            decision=True,
            correct=None,
            latent=(old_x, old_v),
            events=events,
            stabilized=stabilized,
        )
        return next_obs, reward, info
