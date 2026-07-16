import numpy as np

from streaming_rl_feature_geometry.cross_metrics import (
    decision_conditioned_metrics,
    grouped_split_indices,
    ridge_probe,
)


def _tmaze_samples(seed=5, random_decisions=False):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(200), 2)
    cues = np.where(groups % 2, 1.0, -1.0)
    decisions = np.tile([False, True], 200)
    positions = np.tile([1.0, 3.0], 200)
    features = np.column_stack((cues, rng.normal(0, 0.01, len(cues))))
    if random_decisions:
        features[decisions] = rng.normal(size=(decisions.sum(), 2))
    latents = cues[:, None]
    phases = np.where(decisions, "junction", "corridor")
    return features, latents, groups, positions, decisions, phases


def test_grouped_split_has_no_group_leakage():
    groups = np.repeat(np.arange(20), 3)
    train, test = grouped_split_indices(groups, seed=9)
    assert not set(groups[train]) & set(groups[test])


def test_perfect_tmaze_encoding_decodes_at_true_decisions_and_by_position():
    arrays = _tmaze_samples()
    metrics, by_position = decision_conditioned_metrics("tmaze", *arrays, seed=71)
    assert metrics["decision_probe_status"] == "ok"
    assert metrics["cue_decodability_at_decision"] > 0.98
    assert metrics["decision_probe_sample_count"] == 200
    assert {row["corridor_position"] for row in by_position} == {1, 3}


def test_decision_mask_does_not_use_nondecision_latent_encoding():
    arrays = _tmaze_samples(random_decisions=True)
    metrics, _ = decision_conditioned_metrics("tmaze", *arrays, seed=71)
    assert 0.35 <= metrics["cue_decodability_at_decision"] <= 0.65


def test_two_loop_perfect_features_support_identity_and_phase_probes():
    groups = np.arange(240)
    identity = groups % 2
    phase = (groups % 7).astype(float)
    angle = phase / 7 * 2 * np.pi
    latents = np.column_stack((identity, phase, angle))
    features = np.column_stack((identity, phase, np.sin(angle), np.cos(angle)))
    decisions = np.ones(len(groups), dtype=bool)
    positions = phase.copy()
    phases = np.where(identity == 0, "loop_0_phase_decision", "loop_1_phase_decision")
    metrics, _ = decision_conditioned_metrics(
        "two_loop", features, latents, groups, positions, decisions, phases, seed=17
    )
    assert metrics["identity_decodability_at_decision"] > 0.98
    assert metrics["phase_r2_at_decision"] > 0.98
    assert metrics["decision_probe_types"] == "loop_0,loop_1"


def test_random_features_have_low_grouped_regression_r2():
    rng = np.random.default_rng(88)
    groups = np.arange(400)
    features = rng.normal(size=(400, 6))
    targets = rng.normal(size=400)
    metrics, truth, _ = ridge_probe(features, targets, groups, seed=19)
    assert len(truth) > 0
    assert metrics["probe_r2"] < 0.2


def test_inapplicable_and_insufficient_probe_metrics_are_nan_with_status():
    arrays = _tmaze_samples()
    inapplicable, _ = decision_conditioned_metrics("ringworld", *arrays, seed=3)
    assert inapplicable["decision_probe_status"] == "not_applicable"
    assert np.isnan(inapplicable["cue_decodability_at_decision"])
    insufficient_arrays = tuple(value[:4] for value in arrays)
    insufficient, _ = decision_conditioned_metrics("tmaze", *insufficient_arrays, seed=3)
    assert insufficient["decision_probe_status"] == "insufficient"
    assert np.isnan(insufficient["cue_decodability_at_decision"])
