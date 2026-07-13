"""Held-out cross-environment probes; outputs never feed agent learning."""

from __future__ import annotations

import numpy as np

from .metrics import cue_separation_metrics


class DiagnosticReservoir:
    """Bounded uniform reservoir for representation, latent, and group tuples."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 8:
            raise ValueError("diagnostic reservoir capacity must be at least eight")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.features: list[np.ndarray] = []
        self.latents: list[np.ndarray] = []
        self.groups: list[int] = []

    def add(self, features: np.ndarray, latent: np.ndarray, group: int) -> None:
        self.seen += 1
        item = (np.asarray(features, dtype=np.float64).copy(), np.asarray(latent).copy(), int(group))
        if len(self.features) < self.capacity:
            self.features.append(item[0])
            self.latents.append(item[1])
            self.groups.append(item[2])
            return
        index = int(self.rng.integers(self.seen))
        if index < self.capacity:
            self.features[index], self.latents[index], self.groups[index] = item

    def arrays(self, feature_dim: int, latent_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.features:
            return (
                np.empty((0, feature_dim), dtype=np.float64),
                np.empty((0, latent_dim), dtype=np.float64),
                np.empty(0, dtype=int),
            )
        return np.vstack(self.features), np.vstack(self.latents), np.asarray(self.groups, dtype=int)


def _group_split(groups: np.ndarray, seed: int, test_fraction: float = 0.3):
    unique = np.unique(groups)
    if len(unique) < 4:
        return None
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = min(max(1, int(round(test_fraction * len(unique)))), len(unique) - 2)
    test_groups = unique[:n_test]
    train = np.flatnonzero(~np.isin(groups, test_groups))
    test = np.flatnonzero(np.isin(groups, test_groups))
    return train, test


def ridge_probe(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    seed: int,
    ridge: float = 1e-3,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Deterministic grouped held-out multivariate ridge regression."""

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    if features.ndim != 2 or len(features) != len(targets) or features.shape[1] == 0:
        return {"probe_mse": 0.0, "probe_r2": 0.0}, np.empty_like(targets[:0]), np.empty_like(targets[:0])
    split = _group_split(np.asarray(groups), seed)
    if split is None:
        return {"probe_mse": 0.0, "probe_r2": 0.0}, np.empty_like(targets[:0]), np.empty_like(targets[:0])
    train, test = split
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    train_x = np.column_stack(((features[train] - mean) / scale, np.ones(len(train))))
    test_x = np.column_stack(((features[test] - mean) / scale, np.ones(len(test))))
    penalty = ridge * np.eye(train_x.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ targets[train])
    predictions = test_x @ weights
    truth = targets[test]
    mse = float(np.mean((predictions - truth) ** 2))
    denominator = float(np.sum((truth - truth.mean(axis=0)) ** 2))
    r2 = 1.0 - float(np.sum((predictions - truth) ** 2)) / max(denominator, 1e-12)
    return {"probe_mse": mse, "probe_r2": r2}, truth, predictions


def task_information_metrics(
    environment: str,
    features: np.ndarray,
    latents: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute environment-specific held-out information diagnostics."""

    payload: dict[str, np.ndarray] = {}
    if len(features) < 16:
        return {"task_decodability": 0.0}, payload
    if environment == "tmaze":
        cue = latents[:, 0]
        metrics = cue_separation_metrics(features, cue, groups, seed)
        metrics["task_decodability"] = metrics["cue_decodability"]
        return metrics, payload
    if environment == "ringworld":
        angle = latents[:, 1]
        targets = np.column_stack((np.cos(angle), np.sin(angle)))
        probe, truth, predictions = ridge_probe(features, targets, groups, seed)
        if not len(truth):
            return {
                "phase_mse": 0.0,
                "phase_r2": 0.0,
                "phase_absolute_error": float(np.pi),
                "circular_correlation": 0.0,
                "neighborhood_preservation": 0.0,
                "task_decodability": 0.0,
            }, payload
        true_angle = np.arctan2(truth[:, 1], truth[:, 0])
        pred_angle = np.arctan2(predictions[:, 1], predictions[:, 0])
        error = np.angle(np.exp(1j * (pred_angle - true_angle)))
        phase_order = np.argsort(true_angle)
        ordered_prediction = pred_angle[phase_order]
        adjacent_difference = np.angle(
            np.exp(1j * (np.roll(ordered_prediction, -1) - ordered_prediction))
        )
        neighborhood = float(np.mean(np.cos(adjacent_difference)))
        metrics = {
            "phase_mse": probe["probe_mse"],
            "phase_r2": probe["probe_r2"],
            "phase_absolute_error": float(np.mean(np.abs(error))),
            "circular_correlation": float(np.mean(np.cos(error))),
            "neighborhood_preservation": neighborhood,
            "task_decodability": float(np.mean(np.cos(error))),
        }
        payload.update(true_phase=true_angle, predicted_phase=pred_angle)
        return metrics, payload
    if environment == "two_loop":
        identity = latents[:, 0]
        cue_metrics = cue_separation_metrics(features, np.where(identity > 0, 1, -1), groups, seed)
        angle = latents[:, 2]
        targets = np.column_stack((np.cos(angle), np.sin(angle)))
        probe, truth, predictions = ridge_probe(features, targets, groups, seed + 1)
        if not len(truth):
            return {
                "identity_decodability": cue_metrics["cue_decodability"],
                "identity_margin": cue_metrics["cue_margin"],
                "phase_mse": 0.0,
                "phase_r2": 0.0,
                "phase_absolute_error": float(np.pi),
                "joint_state_decodability": 0.0,
                "task_decodability": cue_metrics["cue_decodability"],
            }, payload
        true_angle = np.arctan2(truth[:, 1], truth[:, 0])
        pred_angle = np.arctan2(predictions[:, 1], predictions[:, 0])
        error = np.angle(np.exp(1j * (pred_angle - true_angle)))
        metrics = {
            "identity_decodability": cue_metrics["cue_decodability"],
            "identity_margin": cue_metrics["cue_margin"],
            "phase_mse": probe["probe_mse"],
            "phase_r2": probe["probe_r2"],
            "phase_absolute_error": float(np.mean(np.abs(error))),
            "joint_state_decodability": float(
                cue_metrics["cue_decodability"] * np.mean(np.cos(error))
            ),
            "task_decodability": float(cue_metrics["cue_decodability"]),
        }
        payload.update(true_phase=true_angle, predicted_phase=pred_angle)
        return metrics, payload
    if environment == "hidden_velocity":
        probe, truth, predictions = ridge_probe(features, latents[:, 1], groups, seed)
        if not len(truth):
            return {
                "velocity_decoding_mse": 0.0,
                "velocity_decoding_r2": 0.0,
                "state_estimation_error": 0.0,
                "task_decodability": 0.0,
            }, payload
        metrics = {
            "velocity_decoding_mse": probe["probe_mse"],
            "velocity_decoding_r2": probe["probe_r2"],
            "state_estimation_error": probe["probe_mse"],
            "task_decodability": probe["probe_r2"],
        }
        payload.update(true_velocity=truth.ravel(), predicted_velocity=predictions.ravel())
        return metrics, payload
    raise ValueError(f"unsupported task-information environment {environment!r}")
