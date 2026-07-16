import numpy as np
import pytest

from streaming_rl_feature_geometry.controller import SarsaLambda
from streaming_rl_feature_geometry.cross_env import make_environment


@pytest.mark.parametrize("environment", ["hidden_velocity", "hidden_velocity_informative"])
def test_hidden_velocity_reward_components_are_authoritative_and_bounded(environment):
    env = make_environment(environment, seed=23)
    for step in range(320):
        observation, reward, info = env.step(step % env.n_actions)
        diagnostics = info.diagnostics
        assert reward == pytest.approx(-diagnostics["total_cost"])
        assert diagnostics["total_cost"] == pytest.approx(
            diagnostics["position_cost"]
            + diagnostics["velocity_cost"]
            + diagnostics["action_cost"]
        )
        assert abs(diagnostics["position"]) <= env.position_limit
        assert abs(diagnostics["velocity"]) <= env.velocity_limit
        assert np.isfinite(observation).all() and np.isfinite(reward)


def test_informative_observation_excludes_velocity_and_disturbances_reproduce():
    first = make_environment("hidden_velocity_informative", seed=31)
    second = make_environment("hidden_velocity_informative", seed=31)
    assert "velocity" not in first.observation_names
    disturbances = 0
    for step in range(400):
        action = step % first.n_actions
        _, _, first_info = first.step(action)
        _, _, second_info = second.step(action)
        assert first_info == second_info
        disturbances += int(first_info.diagnostics["disturbance_active"])
        assert len(first.oracle_features) == 2
    assert disturbances >= 2


def test_fixed_alpha_matches_original_update_equation_exactly():
    controller = SarsaLambda(2, 3, seed=1, alpha=0.03, gamma=0.9, lam=0.8, epsilon=0)
    features = np.asarray([1.0, -2.0, 0.5])
    next_features = np.asarray([0.5, 1.0, -1.0])
    expected_trace = np.zeros((2, 3))
    expected_trace[1] += features
    expected_delta = 2.0
    expected_weights = 0.03 * expected_delta * expected_trace
    delta, _, _ = controller.update(features, 1, 2.0, next_features, 0)
    assert delta == expected_delta
    assert np.array_equal(controller.w, expected_weights)
    assert controller.alpha_metrics()["controller_alpha_mode"] == "fixed"


def test_norm_scaled_alpha_is_causal_finite_bounded_and_scale_robust():
    def one_update(scale):
        controller = SarsaLambda(
            2,
            2,
            seed=2,
            alpha=0.1,
            gamma=0.9,
            lam=0.0,
            epsilon=0,
            alpha_mode="norm_scaled",
            alpha_min=1e-7,
            alpha_max=0.1,
        )
        features = scale * np.asarray([1.0, 2.0])
        _, update_norm, _ = controller.update(features, 0, 1.0, features, 0)
        return controller, update_norm

    base, base_update = one_update(1.0)
    scaled, scaled_update = one_update(100.0)
    assert scaled_update < 10.0 * base_update
    for controller in (base, scaled):
        metrics = controller.alpha_metrics()
        assert np.isfinite(metrics["effective_alpha_mean"])
        assert 1e-7 <= metrics["effective_alpha_min"] <= metrics["effective_alpha_max"] <= 0.1
