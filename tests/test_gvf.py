import numpy as np
import pytest

from streaming_rl_feature_geometry.env import A_LEFT, ContinuingTMaze
from streaming_rl_feature_geometry.gvf import TraceGVFBank


def test_mixed_gvf_definitions_dimensions_and_horizons():
    bank = TraceGVFBank(obs_dim=8, seed=0, gammas=(0.6, 0.9))
    assert bank.d == 10
    assert bank.w.shape == (10, bank.input_dim)
    records = bank.definition_records()
    assert {record["cumulant"] for record in records} == {
        "observation",
        "positive_reward",
        "negative_reward",
    }
    assert {record["effective_horizon"] for record in records} == {2.5, 10.000000000000002}


def test_online_td_update_is_finite_and_cannot_be_reused():
    env = ContinuingTMaze(corridor_length=2, seed=4)
    bank = TraceGVFBank(env.obs_dim, seed=9)
    prediction = bank.features(env.observation)
    next_observation, reward, _, _ = env.step(A_LEFT)
    update = bank.update(next_observation, reward)
    assert prediction.shape == update.deltas.shape == update.cumulants.shape == (bank.d,)
    assert np.isfinite(update.deltas).all() and np.isfinite(bank.w).all()
    with pytest.raises(RuntimeError):
        bank.update(next_observation, reward)


def test_reward_cumulants_are_observed_only_after_transition():
    env = ContinuingTMaze(corridor_length=1, seed=2)
    env.phase_i = env.corridor_length + 1
    env.cue = -1
    bank = TraceGVFBank(env.obs_dim, seed=5)
    bank.features(env.observation)
    next_observation, reward, _, _ = env.step(A_LEFT)
    update = bank.update(next_observation, reward)
    positive = [i for i, definition in enumerate(bank.defs) if definition.cumulant == "positive_reward"]
    negative = [i for i, definition in enumerate(bank.defs) if definition.cumulant == "negative_reward"]
    assert np.all(update.cumulants[positive] == 1.0)
    assert np.all(update.cumulants[negative] == 0.0)


def test_same_seed_and_transition_stream_reproduce_raw_features():
    env_a = ContinuingTMaze(corridor_length=2, seed=8)
    env_b = ContinuingTMaze(corridor_length=2, seed=8)
    bank_a = TraceGVFBank(env_a.obs_dim, seed=10)
    bank_b = TraceGVFBank(env_b.obs_dim, seed=10)
    assert np.allclose(bank_a.features(env_a.observation), bank_b.features(env_b.observation))
    for action in [0, 1, 1, 0] * 20:
        next_a, reward_a, _, _ = env_a.step(action)
        next_b, reward_b, _, _ = env_b.step(action)
        bank_a.update(next_a, reward_a)
        bank_b.update(next_b, reward_b)
        assert np.allclose(bank_a.features(next_a), bank_b.features(next_b))
        assert np.allclose(bank_a.trace_state, bank_b.trace_state)
        assert np.allclose(bank_a.w, bank_b.w)

