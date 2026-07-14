import numpy as np
import pytest

from streaming_rl_feature_geometry.cross_env import ENVIRONMENT_IDS, make_environment


@pytest.mark.parametrize("env_id", ENVIRONMENT_IDS)
def test_registered_environments_are_continuing_finite_and_reproducible(env_id):
    first = make_environment(env_id, seed=17)
    second = make_environment(env_id, seed=17)
    for step in range(80):
        assert np.allclose(first.observation, second.observation)
        action = step % first.n_actions
        a_obs, a_reward, a_info = first.step(action)
        b_obs, b_reward, b_info = second.step(action)
        assert np.allclose(a_obs, b_obs)
        assert a_reward == pytest.approx(b_reward)
        assert a_info == b_info
        assert np.isfinite(a_obs).all() and np.isfinite(a_reward)
        assert len(a_info.events) == len(first.event_names)


def test_ringworld_decisions_are_observation_aliased_but_oracle_distinct():
    env = make_environment("ringworld", seed=0, size=12)
    observations = {}
    oracles = {}
    for _ in range(24):
        position = int(env.latent_state[0])
        if position in (3, 9):
            observations[position] = env.observation.copy()
            oracles[position] = env.oracle_features.copy()
        env.step(0)
    assert np.allclose(observations[3], observations[9])
    assert not np.allclose(oracles[3], oracles[9])


def test_two_loop_decision_hides_identity_and_identity_cue_is_brief():
    env = make_environment("two_loop", seed=3, loop_lengths=(8, 10))
    decision_observations = []
    cue_observations = []
    for _ in range(400):
        identity, phase, _ = env.latent_state
        if phase == 0:
            cue_observations.append((int(identity), env.observation.copy()))
        if env.observation[5] == 1.0:
            decision_observations.append((int(identity), env.observation.copy()))
        env.step(0)
    assert {identity for identity, _ in cue_observations} == {0, 1}
    left = next(obs for identity, obs in decision_observations if identity == 0)
    right = next(obs for identity, obs in decision_observations if identity == 1)
    assert np.allclose(left, right)


def test_hidden_velocity_observation_omits_velocity_and_state_is_bounded():
    env = make_environment("hidden_velocity", seed=4)
    assert env.observation.shape == (4,)
    assert env.oracle_features.shape == (2,)
    for step in range(1000):
        env.step(step % 3)
        x, v = env.latent_state
        assert abs(x) <= env.position_limit + 1e-12
        assert abs(v) <= env.velocity_limit + 1e-12


@pytest.mark.parametrize("env_id", ("ringworld", "two_loop", "hidden_velocity"))
def test_latent_values_are_not_present_as_extra_observation_fields(env_id):
    env = make_environment(env_id, seed=11)
    assert env.observation.shape == (env.obs_dim,)
    assert env.oracle_features.shape != env.observation.shape or not np.allclose(
        env.oracle_features, env.observation
    )
