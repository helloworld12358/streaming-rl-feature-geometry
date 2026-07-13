import numpy as np

from streaming_rl_feature_geometry.controller import SarsaLambda


def test_controller_update_dimensions_and_eligibility_trace():
    controller = SarsaLambda(2, 3, seed=1, alpha=0.1, gamma=0.9, lam=0.8, epsilon=0)
    features = np.array([1.0, 2.0, -1.0])
    controller.update(features, 0, 1.0, features, 0)
    assert controller.w.shape == controller.e.shape == (2, 3)
    assert np.allclose(controller.e[0], features)
    controller.update(features, 0, 0.0, features, 0)
    assert np.allclose(controller.e[0], (0.9 * 0.8 + 1.0) * features)


def test_fixed_seed_action_sequence_is_reproducible():
    first = SarsaLambda(2, 2, seed=9, epsilon=0.4)
    second = SarsaLambda(2, 2, seed=9, epsilon=0.4)
    features = np.ones(2)
    assert [first.act(features) for _ in range(100)] == [second.act(features) for _ in range(100)]


def test_simple_continuing_bandit_sanity():
    controller = SarsaLambda(2, 1, seed=3, alpha=0.03, gamma=0.0, lam=0.0, epsilon=0.1)
    features = np.ones(1)
    action = controller.act(features)
    for _ in range(1500):
        reward = 1.0 if action == 0 else 0.0
        next_action = controller.act(features)
        controller.update(features, action, reward, features, next_action)
        action = next_action
    assert controller.q(features)[0] > controller.q(features)[1] + 0.5

