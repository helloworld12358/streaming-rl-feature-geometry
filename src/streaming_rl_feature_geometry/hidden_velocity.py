"""Bounded continuing position control with unobserved velocity.

Reward components are computed here once and exposed as diagnostics.  Consumers
must log these values rather than duplicating the reward equation.
"""

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
        position_cost_weight: float = 1.0,
        velocity_cost_weight: float = 0.25,
        action_cost_weight: float = 0.02,
        stable_position: float = 0.4,
        stable_velocity: float = 0.12,
    ) -> None:
        if not 0.0 <= rho < 1.0 or acceleration <= 0 or noise_std < 0:
            raise ValueError("invalid hidden-velocity dynamics")
        if position_limit <= 0 or velocity_limit <= 0:
            raise ValueError("state limits must be positive")
        if min(position_cost_weight, velocity_cost_weight, action_cost_weight) < 0:
            raise ValueError("cost weights must be non-negative")
        if stable_position <= 0 or stable_velocity <= 0:
            raise ValueError("stable-region limits must be positive")
        self.rng = np.random.default_rng(seed)
        self.rho = float(rho)
        self.acceleration = float(acceleration)
        self.noise_std = float(noise_std)
        self.position_limit = float(position_limit)
        self.velocity_limit = float(velocity_limit)
        self.position_cost_weight = float(position_cost_weight)
        self.velocity_cost_weight = float(velocity_cost_weight)
        self.action_cost_weight = float(action_cost_weight)
        self.stable_position = float(stable_position)
        self.stable_velocity = float(stable_velocity)
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

    def _reflect(self, position: float, velocity: float) -> tuple[float, float, bool, bool]:
        """Reflect at documented physical walls; this is environment dynamics."""

        boundary_hit = bool(abs(position) > self.position_limit)
        boundary_violation = boundary_hit
        while abs(position) > self.position_limit:
            if position > self.position_limit:
                position = 2.0 * self.position_limit - position
                velocity = -abs(velocity)
            elif position < -self.position_limit:
                position = -2.0 * self.position_limit - position
                velocity = abs(velocity)
        if abs(velocity) > self.velocity_limit:
            boundary_hit = True
            boundary_violation = True
            velocity = np.sign(velocity) * (
                2.0 * self.velocity_limit - min(abs(velocity), 2.0 * self.velocity_limit)
            )
        return float(position), float(velocity), boundary_hit, boundary_violation

    def _disturbance(self) -> tuple[bool, float, bool, float]:
        """Return disturbance metadata; the original environment has none."""

        return False, 0.0, False, 0.0

    def _reward_components(self, action: int) -> dict[str, float | bool]:
        """Compute the authoritative per-step cost and reward decomposition."""

        position_squared = float(self.x**2)
        velocity_squared = float(self.v**2)
        control_effort = float((int(action) - 1) ** 2)
        position_cost = self.position_cost_weight * position_squared
        velocity_cost = self.velocity_cost_weight * velocity_squared
        action_cost = self.action_cost_weight * control_effort
        total_cost = position_cost + velocity_cost + action_cost
        stabilized = bool(
            abs(self.x) < self.stable_position and abs(self.v) < self.stable_velocity
        )
        return {
            "position": float(self.x),
            "velocity": float(self.v),
            "abs_position": float(abs(self.x)),
            "abs_velocity": float(abs(self.v)),
            "position_squared": position_squared,
            "velocity_squared": velocity_squared,
            "position_cost": float(position_cost),
            "velocity_cost": float(velocity_cost),
            "action_cost": float(action_cost),
            "control_effort": control_effort,
            "total_cost": float(total_cost),
            "reward": -float(total_cost),
            "stabilized": stabilized,
        }

    def step(self, action: int):
        from .cross_env import CrossStepInfo

        if int(action) not in (0, 1, 2):
            raise ValueError("hidden-velocity action must be 0, 1, or 2")
        old_x, old_v = self.x, self.v
        acceleration = (int(action) - 1) * self.acceleration
        noise = float(self.rng.normal(0.0, self.noise_std))
        disturbance_active, disturbance_magnitude, post_disturbance, steps_since = (
            self._disturbance()
        )
        next_v = self.rho * self.v + acceleration + noise + disturbance_magnitude
        next_x = self.x + next_v
        self.x, self.v, boundary_hit, boundary_violation = self._reflect(next_x, next_v)
        diagnostics = self._reward_components(action)
        diagnostics.update(
            boundary_hit=boundary_hit,
            boundary_violation=boundary_violation,
            disturbance_active=disturbance_active,
            disturbance_magnitude=float(disturbance_magnitude),
            post_disturbance=post_disturbance,
            steps_since_disturbance=float(steps_since),
        )
        reward = float(diagnostics["reward"])
        stabilized = bool(diagnostics["stabilized"])
        self.t += 1
        next_obs = self.observation
        events = (
            float(self.x < -0.25),
            float(self.x > 0.25),
            float(abs(self.x) <= 0.25),
            float(float(diagnostics["total_cost"]) < 0.25),
            float(float(diagnostics["total_cost"]) > 1.0),
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
            diagnostics=diagnostics,
        )
        return next_obs, reward, info


class HiddenVelocityInformative(HiddenVelocity):
    """Higher-inertia variant with sparse bounded velocity impulses.

    The observation remains position-only.  The minimal mechanism change is a
    stronger velocity penalty plus occasional impulses, making velocity memory
    useful without adding labels, future data, or large observation noise.
    """

    env_id = "hidden_velocity_informative"

    def __init__(
        self,
        seed: int = 0,
        rho: float = 0.97,
        acceleration: float = 0.09,
        noise_std: float = 0.01,
        position_limit: float = 2.5,
        velocity_limit: float = 0.75,
        position_cost_weight: float = 1.0,
        velocity_cost_weight: float = 1.0,
        action_cost_weight: float = 0.015,
        stable_position: float = 0.35,
        stable_velocity: float = 0.10,
        disturbance_interval_min: int = 96,
        disturbance_interval_max: int = 144,
        disturbance_magnitude_min: float = 0.24,
        disturbance_magnitude_max: float = 0.38,
        recovery_window: int = 32,
    ) -> None:
        super().__init__(
            seed=seed,
            rho=rho,
            acceleration=acceleration,
            noise_std=noise_std,
            position_limit=position_limit,
            velocity_limit=velocity_limit,
            position_cost_weight=position_cost_weight,
            velocity_cost_weight=velocity_cost_weight,
            action_cost_weight=action_cost_weight,
            stable_position=stable_position,
            stable_velocity=stable_velocity,
        )
        if disturbance_interval_min < 2 or disturbance_interval_max < disturbance_interval_min:
            raise ValueError("invalid disturbance interval")
        if not 0 < disturbance_magnitude_min <= disturbance_magnitude_max:
            raise ValueError("invalid disturbance magnitude")
        if recovery_window < 1:
            raise ValueError("recovery_window must be positive")
        self.disturbance_interval_min = int(disturbance_interval_min)
        self.disturbance_interval_max = int(disturbance_interval_max)
        self.disturbance_magnitude_min = float(disturbance_magnitude_min)
        self.disturbance_magnitude_max = float(disturbance_magnitude_max)
        self.recovery_window = int(recovery_window)
        self._last_disturbance_step: int | None = None
        self._next_disturbance_step = self._draw_next_disturbance(0)

    def _draw_next_disturbance(self, current_step: int) -> int:
        interval = int(
            self.rng.integers(self.disturbance_interval_min, self.disturbance_interval_max + 1)
        )
        return current_step + interval

    def _disturbance(self) -> tuple[bool, float, bool, float]:
        upcoming_step = self.t + 1
        active = upcoming_step == self._next_disturbance_step
        magnitude = 0.0
        if active:
            sign = -1.0 if self.rng.random() < 0.5 else 1.0
            magnitude = sign * float(
                self.rng.uniform(self.disturbance_magnitude_min, self.disturbance_magnitude_max)
            )
            self._last_disturbance_step = upcoming_step
            self._next_disturbance_step = self._draw_next_disturbance(upcoming_step)
        if self._last_disturbance_step is None:
            return active, magnitude, False, 0.0
        steps_since = float(upcoming_step - self._last_disturbance_step)
        post = bool(0.0 <= steps_since <= self.recovery_window)
        return active, magnitude, post, steps_since
