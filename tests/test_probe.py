import numpy as np

from streaming_rl_feature_geometry.metrics import cue_probe_score, cue_separation_metrics


def test_probe_split_is_trial_grouped_and_deterministic():
    rng = np.random.default_rng(12)
    groups = np.arange(200)
    cues = np.where(groups % 2 == 0, -1, 1)
    features = np.column_stack(
        (cues + rng.normal(0, 0.15, len(cues)), rng.normal(size=len(cues)))
    )
    first = cue_probe_score(features, cues, seed=3, groups=groups)
    second = cue_probe_score(features, cues, seed=3, groups=groups)
    assert first == second and first > 0.95


def test_probe_is_pure_offline_analysis():
    rng = np.random.default_rng(13)
    groups = np.arange(100)
    cues = np.where(groups % 2, 1, -1)
    features = rng.normal(size=(100, 4))
    original = features.copy()
    cue_separation_metrics(features, cues, groups, seed=4)
    assert np.array_equal(features, original)

