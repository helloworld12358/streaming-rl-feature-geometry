"""Causal online feature transforms and offline representation diagnostics."""

from __future__ import annotations

import numpy as np


class OnlineMoments:
    """Full-covariance Welford statistics with no look-ahead."""

    def __init__(self, d: int) -> None:
        if d < 1:
            raise ValueError("feature dimension must be positive")
        self.d = int(d)
        self.n = 0
        self.mean = np.zeros(self.d, dtype=np.float64)
        self.M2 = np.zeros((self.d, self.d), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.d,):
            raise ValueError(f"expected feature shape {(self.d,)}, got {x.shape}")
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += np.outer(delta, x - self.mean)

    def cov(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.M2)
        covariance = self.M2 / (self.n - 1)
        return 0.5 * (covariance + covariance.T)

    def var(self) -> np.ndarray:
        return np.maximum(np.diag(self.cov()), 0.0)


class FeatureTransform:
    """Apply exactly one property constraint using past statistics only."""

    aliases = {"gaussian": "gaussian_moment"}
    predictive_kinds = {
        "raw",
        "rms_raw",
        "standardized",
        "decorrelated",
        "whitened",
        "gaussian_moment",
        "unit_sphere",
    }

    def __init__(
        self,
        kind: str,
        d: int,
        eps: float = 1e-3,
        update_every: int = 25,
        min_samples: int = 32,
        moment_beta: float = 0.005,
        moment_learning_rate: float = 0.0005,
        gaussian_skew_bound: float = 0.75,
        gaussian_tail_bounds: tuple[float, float] = (0.6, 2.0),
    ) -> None:
        kind = self.aliases.get(kind, kind)
        if kind not in self.predictive_kinds:
            raise ValueError(f"unknown feature transform {kind!r}")
        if eps <= 0 or update_every < 1 or min_samples < 2:
            raise ValueError("invalid transform hyperparameters")
        if not 0.0 < moment_beta <= 1.0 or moment_learning_rate < 0:
            raise ValueError("invalid Gaussian-moment hyperparameters")
        if gaussian_skew_bound <= 0 or not 0 < gaussian_tail_bounds[0] <= gaussian_tail_bounds[1]:
            raise ValueError("invalid Gaussian-moment parameter bounds")

        self.kind = kind
        self.stats = OnlineMoments(d)
        self.eps = float(eps)
        self.update_every = int(update_every)
        self.min_samples = int(min_samples)
        self.W = np.eye(d, dtype=np.float64)
        self.last_refresh = -10**12
        self.last_matrix_change = 0.0
        self.matrix_change_sum = 0.0
        self.matrix_change_max = 0.0
        self.matrix_refreshes = 0

        self.raw_mean_square = 0.0
        self.output_mean_square = 0.0
        self.last_scale_factor = 1.0
        self.last_raw_rms = 0.0
        self.last_output_rms = 0.0

        self.moment_beta = float(moment_beta)
        self.moment_learning_rate = float(moment_learning_rate)
        self.gaussian_skew_bound = float(gaussian_skew_bound)
        self.gaussian_tail_bounds = tuple(map(float, gaussian_tail_bounds))
        self.gaussian_skew_parameter = np.zeros(d, dtype=np.float64)
        self.gaussian_tail_parameter = np.ones(d, dtype=np.float64)
        self.gaussian_raw_moments = np.zeros((4, d), dtype=np.float64)
        self.gaussian_moment_updates = 0
        self.gaussian_parameter_change_sum = 0.0
        self.gaussian_parameter_change_max = 0.0
        self.gaussian_output_stats = OnlineMoments(d)

    def _record_matrix(self, matrix: np.ndarray) -> None:
        change = float(np.linalg.norm(matrix - self.W, ord="fro"))
        self.W = matrix
        self.last_matrix_change = change
        self.matrix_change_sum += change
        self.matrix_change_max = max(self.matrix_change_max, change)
        self.matrix_refreshes += 1
        self.last_refresh = self.stats.n

    def _refresh(self) -> None:
        covariance = self.stats.cov()
        if self.kind == "decorrelated":
            standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0) + self.eps)
            inverse_scale = np.diag(1.0 / standard_deviation)
            correlation = inverse_scale @ covariance @ inverse_scale
            correlation = 0.5 * (correlation + correlation.T)
            values, vectors = np.linalg.eigh(correlation)
            values = np.maximum(values, self.eps)
            inverse_root = vectors @ np.diag(values ** -0.5) @ vectors.T
            matrix = np.diag(standard_deviation) @ inverse_root @ inverse_scale
        else:
            values, vectors = np.linalg.eigh(covariance)
            values = np.maximum(values, self.eps)
            matrix = vectors @ np.diag(values ** -0.5) @ vectors.T
        self._record_matrix(matrix)

    def _matrix_transform(self, features: np.ndarray) -> np.ndarray:
        ready = self.stats.n >= self.min_samples
        due = self.stats.n - self.last_refresh >= self.update_every
        if ready and due:
            self._refresh()
        return features.copy() if not ready else self.W @ (features - self.stats.mean)

    def _update_gaussian_moments(self, shaped: np.ndarray) -> None:
        beta = self.moment_beta
        powers = np.vstack((shaped, shaped**2, shaped**3, shaped**4))
        if self.gaussian_moment_updates == 0:
            self.gaussian_raw_moments = powers
        else:
            self.gaussian_raw_moments = (1.0 - beta) * self.gaussian_raw_moments + beta * powers
        self.gaussian_moment_updates += 1
        if self.gaussian_moment_updates < self.min_samples:
            return

        first, second, third, fourth = self.gaussian_raw_moments
        variance = np.maximum(second - first**2, self.eps)
        central_third = third - 3.0 * first * second + 2.0 * first**3
        central_fourth = fourth - 4.0 * first * third + 6.0 * first**2 * second - 3.0 * first**4
        skewness = central_third / variance**1.5
        kurtosis = central_fourth / variance**2

        old_skew = self.gaussian_skew_parameter.copy()
        old_tail = self.gaussian_tail_parameter.copy()
        self.gaussian_skew_parameter -= self.moment_learning_rate * np.clip(skewness, -2.0, 2.0)
        self.gaussian_tail_parameter += self.moment_learning_rate * np.clip(kurtosis - 3.0, -3.0, 3.0)
        self.gaussian_skew_parameter = np.clip(
            self.gaussian_skew_parameter, -self.gaussian_skew_bound, self.gaussian_skew_bound
        )
        self.gaussian_tail_parameter = np.clip(
            self.gaussian_tail_parameter, *self.gaussian_tail_bounds
        )
        change = float(
            np.linalg.norm(self.gaussian_skew_parameter - old_skew)
            + np.linalg.norm(self.gaussian_tail_parameter - old_tail)
        )
        self.gaussian_parameter_change_sum += change
        self.gaussian_parameter_change_max = max(self.gaussian_parameter_change_max, change)

    def _gaussian_transform(self, whitened: np.ndarray) -> np.ndarray:
        with np.errstate(over="raise", invalid="raise"):
            shaped = np.sinh(
                (np.arcsinh(whitened) + self.gaussian_skew_parameter)
                / self.gaussian_tail_parameter
            )
        if self.gaussian_output_stats.n >= 2:
            output = (shaped - self.gaussian_output_stats.mean) / np.sqrt(
                self.gaussian_output_stats.var() + self.eps
            )
        else:
            output = shaped.copy()
        self._update_gaussian_moments(shaped)
        self.gaussian_output_stats.update(shaped)
        return output

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.stats.d,):
            raise ValueError(
                f"expected predictive feature shape {(self.stats.d,)}, got {features.shape}"
            )

        raw_rms = float(np.sqrt(np.mean(np.square(features))))
        if self.kind == "raw":
            output = features.copy()
        elif self.kind == "rms_raw":
            scale = 1.0 if self.stats.n < 2 else 1.0 / np.sqrt(self.raw_mean_square + self.eps)
            self.last_scale_factor = float(scale)
            output = features * scale
        elif self.kind == "standardized":
            output = (
                features.copy()
                if self.stats.n < 2
                else (features - self.stats.mean) / np.sqrt(self.stats.var() + self.eps)
            )
        elif self.kind == "unit_sphere":
            output = features / (np.linalg.norm(features) + self.eps)
        else:
            output = self._matrix_transform(features)
            if self.kind == "gaussian_moment" and self.stats.n >= self.min_samples:
                output = self._gaussian_transform(output)

        if not np.isfinite(output).all():
            raise FloatingPointError(f"non-finite output from {self.kind} transform")

        # All transform decisions above used statistics through t-1. Updates below
        # make the current raw/output sample available only at t+1.
        new_count = self.stats.n + 1
        self.raw_mean_square += (raw_rms**2 - self.raw_mean_square) / new_count
        output_rms = float(np.sqrt(np.mean(np.square(output))))
        self.output_mean_square += (output_rms**2 - self.output_mean_square) / new_count
        self.last_raw_rms = raw_rms
        self.last_output_rms = output_rms
        self.stats.update(features)
        return output

    def state_metrics(self) -> dict[str, float]:
        transform_condition = float(np.linalg.cond(self.W)) if self.W.size else 0.0
        result = {
            "raw_rms": float(np.sqrt(max(self.raw_mean_square, 0.0))),
            "scale_factor": self.last_scale_factor,
            "transformed_rms": float(np.sqrt(max(self.output_mean_square, 0.0))),
            "transform_condition_number": transform_condition,
            "transform_matrix_change_last": self.last_matrix_change,
            "transform_matrix_change_mean": (
                self.matrix_change_sum / self.matrix_refreshes if self.matrix_refreshes else 0.0
            ),
            "transform_matrix_change_max": self.matrix_change_max,
            "transform_refreshes": float(self.matrix_refreshes),
        }
        if self.kind == "gaussian_moment":
            first, second, third, fourth = self.gaussian_raw_moments
            variance = np.maximum(second - first**2, self.eps)
            central_third = third - 3.0 * first * second + 2.0 * first**3
            central_fourth = fourth - 4.0 * first * third + 6.0 * first**2 * second - 3.0 * first**4
            result.update(
                {
                    "gaussian_parameter_change_mean": (
                        self.gaussian_parameter_change_sum
                        / max(self.gaussian_moment_updates - self.min_samples + 1, 1)
                    ),
                    "gaussian_parameter_change_max": self.gaussian_parameter_change_max,
                    "gaussian_skew_parameter_mean": float(self.gaussian_skew_parameter.mean()),
                    "gaussian_tail_parameter_mean": float(self.gaussian_tail_parameter.mean()),
                    "gaussian_ew_skewness_error": float(
                        np.mean(np.abs(central_third / variance**1.5))
                    ),
                    "gaussian_ew_kurtosis_error": float(
                        np.mean(np.abs(central_fourth / variance**2 - 3.0))
                    ),
                }
            )
        return result


def rep_metrics(samples: np.ndarray) -> dict[str, float]:
    """Return finite offline diagnostics for a matrix of feature samples."""

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("representation samples must be a two-dimensional matrix")
    n, d = samples.shape
    empty = {
        "n_samples": float(n),
        "feature_dim": float(d),
        "active_dimensions": 0.0,
        "mean_active_dimensions": 0.0,
        "near_zero_fraction": 0.0,
        "mean_error": 0.0,
        "variance_mean": 0.0,
        "variance_min": 0.0,
        "variance_max": 0.0,
        "variance_imbalance": 0.0,
        "mean_abs_corr": 0.0,
        "effective_rank": 0.0,
        "isotropy_error": 0.0,
        "condition_number": 0.0,
        "skewness_error": 0.0,
        "kurtosis_error": 0.0,
        "norm_mean": 0.0,
        "norm_var": 0.0,
        "norm_max": 0.0,
    }
    if n < 3 or d == 0:
        return empty
    if not np.isfinite(samples).all():
        raise ValueError("representation samples contain NaN or Inf")

    mean = samples.mean(axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / (n - 1)
    covariance = 0.5 * (covariance + covariance.T)
    variances = np.maximum(np.diag(covariance), 0.0)
    scale_floor = max(float(variances.max(initial=0.0)) * 1e-10, 1e-12)
    active = variances > scale_floor
    active_count = int(active.sum())

    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    largest = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
    condition_number = largest / (float(eigenvalues[0]) + 1e-8) if largest else 0.0
    total = float(eigenvalues.sum())
    if total > 0:
        probabilities = eigenvalues[eigenvalues > 0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        normalized_covariance = covariance / max(total / d, 1e-12)
        isotropy_error = float(np.linalg.norm(normalized_covariance - np.eye(d), ord="fro"))
    else:
        effective_rank = isotropy_error = 0.0

    if active_count >= 2:
        correlations = np.corrcoef(samples[:, active], rowvar=False)
        off_diagonal = correlations[~np.eye(active_count, dtype=bool)]
        mean_abs_corr = float(np.mean(np.abs(off_diagonal)))
    else:
        mean_abs_corr = 0.0

    if active_count:
        standardized = centered[:, active] / np.sqrt(variances[active])
        skewness = np.mean(standardized**3, axis=0)
        kurtosis = np.mean(standardized**4, axis=0)
        variance_imbalance = float(variances[active].max() / variances[active].min())
        skewness_error = float(np.mean(np.abs(skewness)))
        kurtosis_error = float(np.mean(np.abs(kurtosis - 3.0)))
    else:
        variance_imbalance = skewness_error = kurtosis_error = 0.0

    norms = np.linalg.norm(samples, axis=1)
    result = {
        "n_samples": float(n),
        "feature_dim": float(d),
        "active_dimensions": float(active_count),
        "mean_active_dimensions": float(np.mean(np.count_nonzero(np.abs(samples) > 1e-8, axis=1))),
        "near_zero_fraction": float(np.mean(np.abs(samples) <= 1e-8)),
        "mean_error": float(np.linalg.norm(mean)),
        "variance_mean": float(variances.mean()),
        "variance_min": float(variances.min()),
        "variance_max": float(variances.max()),
        "variance_imbalance": variance_imbalance,
        "mean_abs_corr": mean_abs_corr,
        "effective_rank": effective_rank,
        "isotropy_error": isotropy_error,
        "condition_number": float(condition_number),
        "skewness_error": skewness_error,
        "kurtosis_error": kurtosis_error,
        "norm_mean": float(norms.mean()),
        "norm_var": float(norms.var()),
        "norm_max": float(norms.max()),
    }
    for index, value in enumerate(eigenvalues[::-1]):
        result[f"eigenvalue_{index:02d}"] = float(value)
    return result
