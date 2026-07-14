"""Regression tests for the two critical defects found in PR #1."""

import numpy as np

from streaming_rl_feature_geometry.env import A_LEFT, A_RIGHT, ContinuingTMaze
from streaming_rl_feature_geometry.transforms import FeatureTransform


def test_outcome_advances_to_next_trial():
    env = ContinuingTMaze(corridor_length=2, seed=0)
    for _ in range(3):
        env.step(A_LEFT)
    correct_action = A_LEFT if env.cue < 0 else A_RIGHT
    env.step(correct_action)
    assert env.phase == "outcome"
    next_cue, _, _, info = env.step(A_LEFT)
    assert info.outcome and env.phase == "cue" and env.trial == 1
    assert next_cue[1] + next_cue[2] == 1


def test_first_standardized_sample_is_not_epsilon_amplified():
    transform = FeatureTransform("standardized", 2)
    first = np.array([10.0, -5.0])
    assert np.allclose(transform.transform(first), first)
