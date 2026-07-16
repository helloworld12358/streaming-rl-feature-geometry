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
        alpha_mode: str = "fixed",
        norm_ema_beta: float = 0.01,
        norm_epsilon: float = 1e-6,
        alpha_min: float = 1e-6,
        alpha_max: float | None = None,
    ) -> None:
        if n_actions < 2 or feat_dim < 1:
            raise ValueError("controller dimensions must be positive")
        if alpha <= 0 or not 0 <= gamma < 1 or not 0 <= lam <= 1 or not 0 <= epsilon <= 1:
            raise ValueError("invalid SARSA hyperparameters")
        if alpha_mode not in {"fixed", "norm_scaled"}:
            raise ValueError("alpha_mode must be fixed or norm_scaled")
        if alpha_mode == "norm_scaled" and (not 0 < norm_ema_beta <= 1 or norm_epsilon <= 0):
            raise ValueError("invalid norm-scaling statistics")
        if alpha_mode == "norm_scaled" and (
            alpha_min <= 0 or (alpha_max is not None and alpha_max < alpha_min)
        ):
            raise ValueError("invalid effective-alpha bounds")
        self.rng = np.random.default_rng(seed)
        self.n_actions = int(n_actions)
        self.feat_dim = int(feat_dim)
        self.w = np.zeros((self.n_actions, self.feat_dim), dtype=np.float64)
        self.e = np.zeros_like(self.w)
        self.alpha = float(alpha)
        self.alpha_mode = alpha_mode
        self.norm_ema_beta = float(norm_ema_beta)
        self.norm_epsilon = float(norm_epsilon)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha if alpha_max is None else alpha_max)
        self._feature_norm_ema = 0.0
        self._alpha_count = 0
        self._alpha_sum = 0.0
        self._alpha_min_seen = float("inf")
        self._alpha_max_seen = 0.0
        self._feature_norm_sum = 0.0
        self._feature_norm_max = 0.0
        self._alpha_clip_count = 0
        self.last_effective_alpha = self.alpha
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
        features = np.asarray(features, dtype=np.float64)
        squared_norm = float(features @ features)
        if self.alpha_mode == "fixed":
            effective_alpha = self.alpha
            clipped = False
        else:
            if self._alpha_count == 0:
                self._feature_norm_ema = squared_norm
            else:
                self._feature_norm_ema += self.norm_ema_beta * (
                    squared_norm - self._feature_norm_ema
                )
            raw_alpha = self.alpha / (self.norm_epsilon + self._feature_norm_ema)
            effective_alpha = float(np.clip(raw_alpha, self.alpha_min, self.alpha_max))
            clipped = not np.isclose(effective_alpha, raw_alpha, rtol=0.0, atol=0.0)
        if not np.isfinite(effective_alpha):
            raise FloatingPointError("non-finite effective controller alpha")
        self.last_effective_alpha = effective_alpha
        self._alpha_count += 1
        self._alpha_sum += effective_alpha
        self._alpha_min_seen = min(self._alpha_min_seen, effective_alpha)
        self._alpha_max_seen = max(self._alpha_max_seen, effective_alpha)
        self._feature_norm_sum += squared_norm
        self._feature_norm_max = max(self._feature_norm_max, squared_norm)
        self._alpha_clip_count += int(clipped)
        current_value = self.q(features)[int(action)]
        next_value = self.q(next_features)[int(next_action)]
        delta = float(reward + self.gamma * next_value - current_value)
        self.e *= self.gamma * self.lam
        self.e[int(action)] += features
        update = effective_alpha * delta * self.e
        self.w += update
        if not np.isfinite(self.w).all():
            raise FloatingPointError("non-finite SARSA parameters")
        return delta, float(np.linalg.norm(update)), float(np.linalg.norm(self.w))

    @property
    def actual_alpha(self) -> float:
        """Most recently observed effective alpha, or base alpha before updates."""

        if self._alpha_count == 0:
            return self.alpha
        return self.last_effective_alpha

    def alpha_metrics(self) -> dict[str, float | str]:
        count = max(self._alpha_count, 1)
        return {
            "controller_alpha_mode": self.alpha_mode,
            "base_controller_alpha": self.alpha,
            "actual_controller_alpha": self.alpha if self.alpha_mode == "fixed" else self.actual_alpha,
            "effective_alpha_mean": self._alpha_sum / count if self._alpha_count else self.alpha,
            "effective_alpha_min": self._alpha_min_seen if self._alpha_count else self.alpha,
            "effective_alpha_max": self._alpha_max_seen if self._alpha_count else self.alpha,
            "mean_effective_alpha": self._alpha_sum / count if self._alpha_count else self.alpha,
            "min_effective_alpha": self._alpha_min_seen if self._alpha_count else self.alpha,
            "max_effective_alpha": self._alpha_max_seen if self._alpha_count else self.alpha,
            "mean_feature_squared_norm": self._feature_norm_sum / count,
            "max_feature_squared_norm": self._feature_norm_max,
            "effective_alpha_clip_fraction": self._alpha_clip_count / count,
        }
