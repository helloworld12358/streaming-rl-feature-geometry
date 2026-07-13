"""Online linear control algorithms."""

from __future__ import annotations

import numpy as np


class SarsaLambda:
    """Accumulating-trace, semi-gradient SARSA(lambda) for continuing tasks."""

    def __init__(
        self,
        n_actions: int,
        feat_dim: int,
        seed: int = 0,
        alpha: float = 0.03,
        gamma: float = 0.98,
        lam: float = 0.8,
        epsilon: float = 0.08,
    ) -> None:
        if n_actions < 2 or feat_dim < 1:
            raise ValueError("controller dimensions must be positive")
        if alpha <= 0 or not 0 <= gamma < 1 or not 0 <= lam <= 1 or not 0 <= epsilon <= 1:
            raise ValueError("invalid SARSA hyperparameters")
        self.rng = np.random.default_rng(seed)
        self.n_actions = int(n_actions)
        self.feat_dim = int(feat_dim)
        self.w = np.zeros((self.n_actions, self.feat_dim), dtype=np.float64)
        self.e = np.zeros_like(self.w)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.epsilon = float(epsilon)

    def q(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.feat_dim,):
            raise ValueError(f"expected controller feature shape {(self.feat_dim,)}, got {features.shape}")
        return self.w @ features

    def act(self, features: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.q(features)
        maxima = np.flatnonzero(np.isclose(values, values.max(), rtol=1e-12, atol=1e-12))
        return int(self.rng.choice(maxima))

    def update(
        self,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        next_action: int,
    ) -> tuple[float, float, float]:
        current_value = self.q(features)[int(action)]
        next_value = self.q(next_features)[int(next_action)]
        delta = float(reward + self.gamma * next_value - current_value)
        self.e *= self.gamma * self.lam
        self.e[int(action)] += features
        update = self.alpha * delta * self.e
        self.w += update
        if not np.isfinite(self.w).all():
            raise FloatingPointError("non-finite SARSA parameters")
        return delta, float(np.linalg.norm(update)), float(np.linalg.norm(self.w))
