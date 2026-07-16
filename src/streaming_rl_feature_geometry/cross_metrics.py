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
        self.positions: list[int | None] = []
        self.decisions: list[bool] = []
        self.phases: list[str] = []

    def add(
        self,
        features: np.ndarray,
        latent: np.ndarray,
        group: int,
        *,
        position: int | None = None,
        decision: bool = False,
        phase: str = "",
    ) -> None:
        self.seen += 1
        item = (
            np.asarray(features, dtype=np.float64).copy(),
            np.asarray(latent).copy(),
            int(group),
            position,
            bool(decision),
            str(phase),
        )
        if len(self.features) < self.capacity:
            self.features.append(item[0])
            self.latents.append(item[1])
            self.groups.append(item[2])
            self.positions.append(item[3])
            self.decisions.append(item[4])
            self.phases.append(item[5])
            return
        index = int(self.rng.integers(self.seen))
        if index < self.capacity:
            self.features[index], self.latents[index], self.groups[index] = item[:3]
            self.positions[index], self.decisions[index], self.phases[index] = item[3:]

    def arrays(self, feature_dim: int, latent_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.features:
            return (
                np.empty((0, feature_dim), dtype=np.float64),
                np.empty((0, latent_dim), dtype=np.float64),
                np.empty(0, dtype=int),
            )
        return np.vstack(self.features), np.vstack(self.latents), np.asarray(self.groups, dtype=int)

    def metadata_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return diagnostic-only position, decision, and phase metadata."""

        return (
            np.asarray([np.nan if value is None else value for value in self.positions], dtype=float),
            np.asarray(self.decisions, dtype=bool),
            np.asarray(self.phases, dtype=str),
        )


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


def grouped_split_indices(
    groups: np.ndarray, seed: int, test_fraction: float = 0.3
) -> tuple[np.ndarray, np.ndarray] | None:
    """Public grouped split helper used by leak-prevention tests."""

    return _group_split(np.asarray(groups), seed, test_fraction)


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
    if environment in {"hidden_velocity", "hidden_velocity_informative"}:
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


DECISION_METRIC_DEFAULTS: dict[str, float | str] = {
    "cue_decodability_at_decision": np.nan,
    "cue_margin_at_decision": np.nan,
    "cue_decodability_before_decision": np.nan,
    "identity_decodability_at_decision": np.nan,
    "identity_margin_at_decision": np.nan,
    "phase_r2_at_decision": np.nan,
    "phase_mse_at_decision": np.nan,
    "phase_absolute_error_at_decision": np.nan,
    "joint_state_decodability_at_decision": np.nan,
    "decision_probe_sample_count": 0,
    "decision_probe_group_count": 0,
    "decision_probe_train_group_count": 0,
    "decision_probe_test_group_count": 0,
    "decision_probe_types": "",
    "decision_probe_status": "not_applicable",
    "decision_probe_failure_reason": "environment_has_no_supported_decision_probe",
}


def _probe_counts(groups: np.ndarray, seed: int) -> dict[str, int] | None:
    split = _group_split(groups, seed)
    if split is None:
        return None
    train, test = split
    return {
        "decision_probe_sample_count": int(len(groups)),
        "decision_probe_group_count": int(len(np.unique(groups))),
        "decision_probe_train_group_count": int(len(np.unique(groups[train]))),
        "decision_probe_test_group_count": int(len(np.unique(groups[test]))),
    }


def _cue_probe_or_nan(
    features: np.ndarray, cues: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[float, float, str]:
    if len(features) < 8 or len(np.unique(groups)) < 4 or len(np.unique(cues)) < 2:
        return np.nan, np.nan, "insufficient_grouped_samples"
    metrics = cue_separation_metrics(features, cues, groups, seed)
    return float(metrics["cue_decodability"]), float(metrics["cue_margin"]), ""


def decision_conditioned_metrics(
    environment: str,
    features: np.ndarray,
    latents: np.ndarray,
    groups: np.ndarray,
    positions: np.ndarray,
    decisions: np.ndarray,
    phases: np.ndarray,
    seed: int,
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    """Held-out probes restricted to true decisions; never feeds agent state."""

    metrics = dict(DECISION_METRIC_DEFAULTS)
    by_position: list[dict[str, float | str]] = []
    if environment not in {"tmaze", "two_loop"}:
        return metrics, by_position
    mask = np.asarray(decisions, dtype=bool)
    if not mask.any():
        metrics.update(
            decision_probe_status="insufficient",
            decision_probe_failure_reason="no_true_decision_samples",
        )
        return metrics, by_position
    decision_features = features[mask]
    decision_latents = latents[mask]
    decision_groups = groups[mask]
    counts = _probe_counts(decision_groups, seed)
    if counts is None:
        metrics.update(
            decision_probe_sample_count=int(mask.sum()),
            decision_probe_group_count=int(len(np.unique(decision_groups))),
            decision_probe_status="insufficient",
            decision_probe_failure_reason="fewer_than_four_decision_groups",
        )
        return metrics, by_position
    metrics.update(counts)
    metrics.update(decision_probe_status="ok", decision_probe_failure_reason="")

    if environment == "tmaze":
        cues = decision_latents[:, 0]
        score, margin, reason = _cue_probe_or_nan(
            decision_features, cues, decision_groups, seed
        )
        metrics["cue_decodability_at_decision"] = score
        metrics["cue_margin_at_decision"] = margin
        if reason:
            metrics.update(decision_probe_status="insufficient", decision_probe_failure_reason=reason)
        before_mask = ~mask & np.isfinite(positions)
        if before_mask.any():
            last_position = float(np.nanmax(positions[before_mask]))
            before_mask &= positions == last_position
            before_score, _, _ = _cue_probe_or_nan(
                features[before_mask], latents[before_mask, 0], groups[before_mask], seed
            )
            metrics["cue_decodability_before_decision"] = before_score
        for position in sorted(np.unique(positions[np.isfinite(positions)])):
            position_mask = positions == position
            position_score, position_margin, failure = _cue_probe_or_nan(
                features[position_mask],
                latents[position_mask, 0],
                groups[position_mask],
                seed,
            )
            by_position.append(
                {
                    "corridor_position": int(position),
                    "cue_decodability": position_score,
                    "cue_margin": position_margin,
                    "sample_count": int(position_mask.sum()),
                    "group_count": int(len(np.unique(groups[position_mask]))),
                    "probe_status": "ok" if not failure else "insufficient",
                    "failure_reason": failure,
                }
            )
        metrics["decision_probe_types"] = "junction"
        return metrics, by_position

    identities = np.where(decision_latents[:, 0] > 0, 1, -1)
    identity_score, identity_margin, identity_failure = _cue_probe_or_nan(
        decision_features, identities, decision_groups, seed
    )
    phase_probe, phase_truth, phase_predictions = ridge_probe(
        decision_features, decision_latents[:, 1], decision_groups, seed + 1
    )
    metrics["identity_decodability_at_decision"] = identity_score
    metrics["identity_margin_at_decision"] = identity_margin
    if len(phase_truth):
        metrics["phase_r2_at_decision"] = phase_probe["probe_r2"]
        metrics["phase_mse_at_decision"] = phase_probe["probe_mse"]
        metrics["phase_absolute_error_at_decision"] = float(
            np.mean(np.abs(phase_predictions - phase_truth))
        )
        metrics["joint_state_decodability_at_decision"] = float(
            identity_score * max(phase_probe["probe_r2"], 0.0)
        )
    else:
        metrics.update(
            decision_probe_status="insufficient",
            decision_probe_failure_reason="phase_probe_group_split_failed",
        )
    if identity_failure:
        metrics.update(decision_probe_status="insufficient", decision_probe_failure_reason=identity_failure)
    decision_phases = phases[mask]
    metrics["decision_probe_types"] = ",".join(
        sorted({value.split("_phase_")[0] for value in decision_phases})
    )
    return metrics, by_position
