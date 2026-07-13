import numpy as np

from streaming_rl_feature_geometry.env import ContinuingTMaze
from streaming_rl_feature_geometry.experiment import controller_features
from streaming_rl_feature_geometry.gvf import TraceGVFBank
from streaming_rl_feature_geometry.transforms import FeatureTransform


def generate_prefix(seed):
    env = ContinuingTMaze(corridor_length=2, seed=seed)
    bank = TraceGVFBank(env.obs_dim, seed=seed + 1)
    transform = FeatureTransform("rms_raw", bank.d)
    observation = env.observation
    prediction = bank.features(observation)
    rows = []
    for action in [0, 1, 0, 0, 1] * 20:
        state = transform.transform(prediction)
        rows.append(controller_features(observation, state, "rms_raw", env.cue))
        next_observation, reward, _, _ = env.step(action)
        bank.update(next_observation, reward)
        prediction = bank.features(next_observation)
        observation = next_observation
    return np.vstack(rows)


def test_same_seed_and_config_reproduce_feature_prefix():
    assert np.allclose(generate_prefix(21), generate_prefix(21))

