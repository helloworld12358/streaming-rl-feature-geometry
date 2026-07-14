import inspect

import numpy as np
import pytest

from streaming_rl_feature_geometry.cross_env import make_environment
from streaming_rl_feature_geometry.predictive import CausalPredictiveBank
from streaming_rl_feature_geometry.priors import TaskMatchedTransform
from streaming_rl_feature_geometry.transforms import FeatureTransform, rep_metrics


@pytest.mark.parametrize("env_id", ("tmaze", "ringworld", "two_loop", "hidden_velocity"))
@pytest.mark.parametrize("bank_kind", ("compact", "mixed"))
def test_common_predictive_bank_is_one_pass_finite_and_semantic(env_id, bank_kind):
    env = make_environment(env_id, seed=2)
    bank = CausalPredictiveBank(
        env_id, env.obs_dim, env.event_names, seed=2, bank=bank_kind
    )
    predictions = bank.features(env.observation)
    with pytest.raises(RuntimeError):
        bank.features(env.observation)
    next_obs, _, info = env.step(0)
    update = bank.update(next_obs, info.events)
    assert predictions.shape == (bank.d,)
    assert len(bank.definition_records()) == bank.d
    assert np.isfinite(update.deltas).all() and np.isfinite(bank.w).all()
    with pytest.raises(RuntimeError):
        bank.update(next_obs, info.events)


def test_sparse_and_bounded_properties_are_explicit_and_causal():
    sample = np.asarray([5.0, -4.0, 3.0, -2.0, 1.0, 0.5])
    sparse = FeatureTransform("sparse", 6, sparse_top_k=2)
    bounded = FeatureTransform("bounded", 6, bounded_alpha=0.7)
    sparse_output = sparse.transform(sample)
    bounded_output = bounded.transform(sample)
    assert np.count_nonzero(sparse_output) == 2
    assert np.max(np.abs(bounded_output)) < 1.0
    assert np.allclose(sparse_output[np.argsort(np.abs(sample))[-2:]], sample[np.argsort(np.abs(sample))[-2:]])


def test_simplex_and_circular_priors_fulfill_declared_geometry():
    simplex = TaskMatchedTransform("simplex", 4)
    circle = TaskMatchedTransform("circular", 4)
    for index in range(200):
        angle = index * 0.1
        features = np.asarray([np.cos(angle), np.sin(angle), angle % 1.0, 1.0])
        simplex_value = simplex.transform(features)
        circle_value = circle.transform(features)
        assert (simplex_value >= 0).all() and simplex_value.sum() == pytest.approx(1.0)
        assert np.linalg.norm(circle_value) == pytest.approx(1.0)


def test_block_prior_keeps_simplex_and_circle_semantics():
    transform = TaskMatchedTransform("block", 6)
    output = transform.transform(np.asarray([2.0, -1.0, 0.5, 0.5, 0.0, 0.0]))
    assert output.shape == (4,)
    assert output[:2].sum() == pytest.approx(1.0)
    assert (output[:2] >= 0).all()
    assert np.linalg.norm(output[2:]) == pytest.approx(1.0)


def test_anisotropic_prior_approaches_predeclared_covariance_ratio():
    rng = np.random.default_rng(9)
    transform = TaskMatchedTransform("anisotropic", 4, anisotropic_ratio=3.0)
    outputs = []
    for sample in rng.normal(size=(6000, 4)):
        value = transform.transform(sample)
        if transform.stats.n > 500:
            outputs.append(value)
    variances = np.diag(np.cov(np.asarray(outputs), rowvar=False))
    assert variances[0] / variances[1] == pytest.approx(9.0, rel=0.2)
    assert rep_metrics(np.asarray(outputs))["effective_rank"] > 1.0


def test_task_matched_transform_api_cannot_accept_latent_labels():
    signature = inspect.signature(TaskMatchedTransform.transform)
    assert tuple(signature.parameters) == ("self", "features")
