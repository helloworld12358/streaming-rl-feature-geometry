"""Offline diagnostics; none of these values feed back into the agent."""

from __future__ import annotations

import numpy as np


class ReservoirSampler:
    """Bounded-memory uniform sampling from a stream with trial identifiers."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.features: list[np.ndarray] = []
        self.labels: list[int] = []
        self.groups: list[int] = []

    def add(self, features: np.ndarray, label: int, group: int) -> None:
        self.seen += 1
        feature_copy = np.asarray(features, dtype=np.float64).copy()
        if len(self.features) < self.capacity:
            self.features.append(feature_copy)
            self.labels.append(int(label))
            self.groups.append(int(group))
            return
        index = int(self.rng.integers(self.seen))
        if index < self.capacity:
            self.features[index] = feature_copy
            self.labels[index] = int(label)
            self.groups[index] = int(group)

    def arrays(self, feature_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.features:
            return (
                np.empty((0, feature_dim), dtype=np.float64),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
            )
        return (
            np.vstack(self.features),
            np.asarray(self.labels, dtype=int),
            np.asarray(self.groups, dtype=int),
        )


def _grouped_split(
    labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator, test_fraction: float
) -> tuple[np.ndarray, np.ndarray] | None:
    train_groups: list[np.ndarray] = []
    test_groups: list[np.ndarray] = []
    for label in (0, 1):
        label_groups = np.unique(groups[labels == label])
        if len(label_groups) < 4:
            return None
        rng.shuffle(label_groups)
        n_test = max(1, int(round(test_fraction * len(label_groups))))
        n_test = min(n_test, len(label_groups) - 2)
        test_groups.append(label_groups[:n_test])
        train_groups.append(label_groups[n_test:])
    train_group_values = np.concatenate(train_groups)
    test_group_values = np.concatenate(test_groups)
    return np.flatnonzero(np.isin(groups, train_group_values)), np.flatnonzero(
        np.isin(groups, test_group_values)
    )


def cue_probe_score(
    features: np.ndarray,
    cues: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
    test_fraction: float = 0.3,
    ridge: float = 1e-3,
) -> float:
    """Held-out balanced accuracy of a deterministic linear cue decoder."""

    features = np.asarray(features, dtype=np.float64)
    cues = np.asarray(cues)
    if features.ndim != 2 or len(features) != len(cues) or features.shape[1] == 0:
        return 0.5
    labels = (cues > 0).astype(int)
    groups = np.arange(len(labels)) if groups is None else np.asarray(groups)
    if len(groups) != len(labels):
        raise ValueError("probe groups must align with samples")
    rng = np.random.default_rng(seed)
    split = _grouped_split(labels, groups, rng, test_fraction)
    if split is None:
        return 0.5
    train, test = split

    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    train_x = (features[train] - mean) / scale
    test_x = (features[test] - mean) / scale
    train_x = np.column_stack((train_x, np.ones(len(train_x))))
    test_x = np.column_stack((test_x, np.ones(len(test_x))))
    targets = np.where(labels[train] == 1, 1.0, -1.0)
    penalty = ridge * np.eye(train_x.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ targets)
    predictions = (test_x @ weights >= 0.0).astype(int)
    recalls = [float(np.mean(predictions[labels[test] == label] == label)) for label in (0, 1)]
    return float(np.mean(recalls))


def cue_separation_metrics(
    features: np.ndarray,
    cues: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Held-out decodability and label-informed geometric diagnostics."""

    features = np.asarray(features, dtype=np.float64)
    cues = np.asarray(cues)
    if len(features) < 8 or len(np.unique(cues)) < 2:
        return {
            "cue_decodability": 0.5,
            "cue_mean_distance": 0.0,
            "cue_margin": 0.0,
            "between_within_ratio": 0.0,
        }
    left = features[cues < 0]
    right = features[cues > 0]
    left_mean = left.mean(axis=0)
    right_mean = right.mean(axis=0)
    difference = right_mean - left_mean
    distance = float(np.linalg.norm(difference))
    direction = difference / max(distance, 1e-12)
    left_projection = left @ direction
    right_projection = right @ direction
    pooled_standard_deviation = np.sqrt(
        0.5 * (left_projection.var() + right_projection.var()) + 1e-12
    )
    margin = float((right_projection.mean() - left_projection.mean()) / pooled_standard_deviation)
    within = float(
        0.5
        * (
            np.mean(np.sum((left - left_mean) ** 2, axis=1))
            + np.mean(np.sum((right - right_mean) ** 2, axis=1))
        )
    )
    between = float(0.25 * np.sum(difference**2))
    return {
        "cue_decodability": cue_probe_score(features, cues, seed, groups=groups),
        "cue_mean_distance": distance,
        "cue_margin": margin,
        "between_within_ratio": between / max(within, 1e-12),
    }
