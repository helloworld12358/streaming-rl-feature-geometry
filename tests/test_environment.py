import numpy as np
import pytest

from streaming_rl_feature_geometry.env import A_LEFT, A_RIGHT, ContinuingTMaze


def test_phase_transitions_and_configured_corridor_length():
    env = ContinuingTMaze(corridor_length=3, seed=1)
    phases = [env.phase]
    for _ in range(5):
        env.step(A_LEFT)
        phases.append(env.phase)
    assert phases == ["cue", "corridor", "corridor", "corridor", "junction", "outcome"]


def test_junction_observation_is_identical_for_both_cues():
    observations = []
    for cue in (-1, 1):
        env = ContinuingTMaze(corridor_length=2, seed=2)
        env.cue = cue
        env.phase_i = env.corridor_length + 1
        observations.append(env.observation)
    assert np.array_equal(observations[0], observations[1])
    assert observations[0][[1, 2, 6, 7]].sum() == 0


def test_delayed_echo_and_reward_match_correct_direction():
    for cue, action in ((-1, A_LEFT), (1, A_RIGHT)):
        env = ContinuingTMaze(corridor_length=1, seed=3)
        env.cue = cue
        env.phase_i = env.corridor_length + 1
        assert env.observation[6] + env.observation[7] == 0
        outcome, reward, _, info = env.step(action)
        assert info.correct and reward == 1.0
        assert outcome[6 if cue < 0 else 7] == 1
        env = ContinuingTMaze(corridor_length=1, seed=3)
        env.cue = cue
        env.phase_i = env.corridor_length + 1
        _, reward, _, info = env.step(A_RIGHT if action == A_LEFT else A_LEFT)
        assert not info.correct and reward == -1.0


def test_fixed_seed_is_reproducible():
    first = ContinuingTMaze(corridor_length=2, seed=19)
    second = ContinuingTMaze(corridor_length=2, seed=19)
    for action in [0, 1, 0, 1] * 10:
        output_a = first.step(action)
        output_b = second.step(action)
        assert np.array_equal(output_a[0], output_b[0])
        assert output_a[1:] == output_b[1:]


def test_single_corridor_change_requires_trial_start():
    env = ContinuingTMaze(corridor_length=2, seed=0)
    env.set_corridor_length(4)
    assert env.corridor_length == 4
    env.step(A_LEFT)
    with pytest.raises(RuntimeError):
        env.set_corridor_length(6)

