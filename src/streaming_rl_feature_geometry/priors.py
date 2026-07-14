"""Task-matched causal representation priors without latent-label inputs."""

from __future__ import annotations

import numpy as np

from .transforms import OnlineMoments


MATCHED_PRIOR_FOR_ENV = {
    "tmaze": "simplex",
    "ringworld": "circular",
    "two_loop": "block",
    "hidden_velocity": "anisotropic",
}


class TaskMatchedTransform:
    """Fixed semantic transforms of predictive features only.

    This API intentionally has no latent-label argument. Pair/block indices are
    predeclared from predictive-bank semantics, never fitted from diagnostics.
    """

    kinds = {"simplex", "circular", "block", "anisotropic"}

    def __init__(
        self,
        kind: str,
        d: int,
        eps: float = 1e-3,
        identity_indices: tuple[int, int] = (0, 1),
        phase_indices: tuple[int, int] = (2, 3),
        anisotropic_ratio: float = 3.0,
    ) -> None:
        if kind not in self.kinds or d < 2 or eps <= 0 or anisotropic_ratio <= 1:
            raise ValueError("invalid task-matched transform")
        for index in (*identity_indices, *phase_indices):
            if index < 0 or index >= d:
                raise ValueError("semantic block index outside predictive vector")
        self.kind = kind
        self.d = int(d)
        self.eps = float(eps)
        self.identity_indices = identity_indices
        self.phase_indices = phase_indices
        self.stats = OnlineMoments(d)
        self.target_scales = np.ones(d, dtype=np.float64)
        self.target_scales[0] = np.sqrt(float(anisotropic_ratio))
        self.target_scales[1] = 1.0 / np.sqrt(float(anisotropic_ratio))
        self.last_output = np.zeros(self.output_dim, dtype=np.float64)

    @property
    def output_dim(self) -> int:
        if self.kind in {"simplex", "circular"}:
            return 2
        if self.kind == "block":
            return 4
        return self.d

    def _causal_standardize(self, features: np.ndarray) -> np.ndarray:
        if self.stats.n < 2:
            return features.copy()
        return (features - self.stats.mean) / np.sqrt(self.stats.var() + self.eps)

    @staticmethod
    def _simplex(pair: np.ndarray) -> np.ndarray:
        shifted = pair - np.max(pair)
        weights = np.exp(np.clip(shifted, -30.0, 0.0))
        return weights / weights.sum()

    def _circle(self, pair: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(pair))
        return np.asarray([1.0, 0.0]) if norm < self.eps else pair / norm

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.d,):
            raise ValueError(f"expected predictive feature shape {(self.d,)}, got {features.shape}")
        standardized = self._causal_standardize(features)
        identity_pair = standardized[list(self.identity_indices)]
        phase_pair = standardized[list(self.phase_indices)]
        if self.kind == "simplex":
            output = self._simplex(identity_pair)
        elif self.kind == "circular":
            output = self._circle(identity_pair)
        elif self.kind == "block":
            output = np.concatenate((self._simplex(identity_pair), self._circle(phase_pair)))
        else:
            output = standardized * self.target_scales
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite task-matched representation")
        self.last_output = output.copy()
        self.stats.update(features)
        return output

    def state_metrics(self) -> dict[str, float]:
        result = {
            "matched_output_norm": float(np.linalg.norm(self.last_output)),
            "matched_output_min": float(self.last_output.min(initial=0.0)),
            "matched_output_max": float(self.last_output.max(initial=0.0)),
        }
        if self.kind in {"simplex", "block"}:
            result["simplex_sum_error"] = float(abs(self.last_output[:2].sum() - 1.0))
            result["simplex_negative_fraction"] = float(np.mean(self.last_output[:2] < 0.0))
        if self.kind in {"circular", "block"}:
            block = self.last_output if self.kind == "circular" else self.last_output[2:4]
            result["circle_radius_error"] = float(abs(np.linalg.norm(block) - 1.0))
        if self.kind == "anisotropic":
            result["anisotropic_target_ratio"] = float(
                (self.target_scales[0] / self.target_scales[1]) ** 2
            )
        return result
