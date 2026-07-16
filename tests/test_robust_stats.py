import numpy as np
import pandas as pd

from streaming_rl_feature_geometry.robust_stats import (
    build_robust_summaries,
    classify_catastrophic_failure,
    interquartile_mean,
    robust_statistics,
)


def test_iqm_and_bootstrap_summary_preserve_long_tail_information():
    values = np.asarray([0.0, 1.0, 2.0, 100.0])
    stats = robust_statistics(values, bootstrap_samples=100, seed=4)
    assert interquartile_mean(values) == 1.5
    assert stats["mean"] > stats["iqm"]
    assert stats["minimum"] == 0 and stats["maximum"] == 100
    assert stats["seed_count"] == 4


def test_configured_catastrophic_failure_has_reason_and_condition_aggregation():
    config = {
        "robust_bootstrap_samples": 20,
        "catastrophic_failure": {
            "global": {"max_control_update_norm_above": 10},
            "environments": {"hidden_velocity": {"final_window_reward_below": -5}},
        },
    }
    base = {
        "environment": "hidden_velocity",
        "condition": "raw",
        "final_performance": -1,
        "final_window_reward": -6,
        "final_parameter_norm": 1,
        "max_control_update_norm": 1,
        "divergence_flag": 0,
        "final_window_stabilization_rate": 0.2,
        "final_window_position_rmse": 1,
        "final_window_velocity_rmse": 1,
        "final_window_control_effort": 1,
        "settling_time": 2,
        "recovery_time": np.nan,
    }
    failed, reason = classify_catastrophic_failure(base, config)
    assert failed and "final_window_reward" in reason
    frame = pd.DataFrame([{**base, "seed": 0}, {**base, "seed": 1, "final_window_reward": -1}])
    robust, failures, classified = build_robust_summaries(frame, config)
    assert failures.iloc[0].catastrophic_failure_count == 1
    assert failures.iloc[0].catastrophic_failure_rate == 0.5
    assert set(robust.metric) >= {"final_performance", "recovery_time"}
    assert classified.catastrophic_failure.sum() == 1
