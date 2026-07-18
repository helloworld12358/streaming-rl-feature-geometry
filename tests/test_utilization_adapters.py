import math

import numpy as np
import pytest

from streaming_rl_feature_geometry.utilization_adapters import (
    RFF_WIDTH,
    TILE_NUM_TILINGS,
    TILE_TABLE_SIZE,
    derive_adapter_seed,
    make_utilization_adapter,
    normalize_adapter_config,
)


IDENTITY = {"name": "identity"}
RFF = {
    "name": "residual_rff",
    "width": 64,
    "frequency_scale": 1.0,
    "residual": True,
}
TILE = {
    "name": "tile_coding",
    "num_tilings": 8,
    "table_size": 512,
    "tiles_per_unit": 4.0,
    "residual": True,
}


def make(config, *, environment="tmaze", seed=7, dimension=3):
    return make_utilization_adapter(
        config, environment=environment, run_seed=seed, input_dim=dimension
    )


def test_missing_and_explicit_identity_have_one_canonical_noop_path():
    assert normalize_adapter_config(None) == normalize_adapter_config(IDENTITY)
    state = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
    implicit = make(None)
    explicit = make(IDENTITY)
    assert implicit.transform(state) is state
    assert explicit.transform(state) is state
    assert implicit.metadata().as_dict() == explicit.metadata().as_dict()


def test_rff_is_fixed_residual_finite_condition_independent_and_seeded():
    state = np.asarray([0.25, -1.5, 2.0], dtype=np.float64)
    first = make(RFF, seed=11)
    same = make(RFF, seed=11)
    different = make(RFF, seed=12)
    output = first.transform(state)
    assert output.shape == (len(state) + RFF_WIDTH,)
    np.testing.assert_array_equal(output[: len(state)], state)
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(first.W, same.W)
    np.testing.assert_array_equal(first.b, same.b)
    assert not np.array_equal(first.W, different.W)
    assert not first.W.flags.writeable and not first.b.flags.writeable
    expected = math.sqrt(2.0 / RFF_WIDTH) * np.cos(first.W @ state + first.b)
    np.testing.assert_allclose(output[len(state) :], expected, rtol=0.0, atol=0.0)


def test_adapter_seed_is_stable_and_does_not_use_condition_or_global_rng():
    before = np.random.get_state()
    seed = derive_adapter_seed("ringworld", 5)
    first = make(RFF, environment="ringworld", seed=5)
    second = make(RFF, environment="ringworld", seed=5)
    after = np.random.get_state()
    assert seed == derive_adapter_seed("ringworld", 5)
    assert seed != derive_adapter_seed("ringworld", 6)
    np.testing.assert_array_equal(first.W, second.W)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_complete_controller_input_must_include_at_least_observation_and_bias():
    with pytest.raises(ValueError, match="complete controller input"):
        make(IDENTITY, dimension=0)


def test_tile_coding_has_eight_deterministic_activations_and_collision_accumulation(
    monkeypatch,
):
    state = np.asarray([0.5, -0.25, 1.25], dtype=np.float64)
    first = make(TILE, seed=4)
    second = make(TILE, seed=4)
    indices = first.tile_indices(state)
    assert indices.shape == (TILE_NUM_TILINGS,)
    np.testing.assert_array_equal(indices, second.tile_indices(state))
    assert not np.array_equal(first._salts, make(TILE, seed=5)._salts)
    output = first.transform(state)
    assert output.shape == (len(state) + TILE_TABLE_SIZE,)
    np.testing.assert_array_equal(output[: len(state)], state)
    assert np.count_nonzero(output[len(state) :]) <= TILE_NUM_TILINGS
    assert output[len(state) :].sum() == pytest.approx(math.sqrt(TILE_NUM_TILINGS))

    monkeypatch.setattr(
        first,
        "tile_indices",
        lambda unused: np.asarray([3, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64),
    )
    collided = first.transform(state)[len(state) :]
    assert collided[3] == pytest.approx(2.0 / math.sqrt(TILE_NUM_TILINGS))


@pytest.mark.parametrize(
    "config, message",
    [
        ({"name": "learned"}, "name"),
        ({"name": "identity", "scope": "controller_state"}, "unknown"),
        ({**RFF, "width": 32}, "width"),
        ({**RFF, "frequency_scale": 0.5}, "frequency_scale"),
        ({**TILE, "table_size": 128}, "table_size"),
        ({**TILE, "tiles_per_unit": 2.0}, "tiles_per_unit"),
        ({**IDENTITY, "fallback": True}, "unknown"),
    ],
)
def test_illegal_adapter_configuration_fails_closed(config, message):
    with pytest.raises(ValueError, match=message):
        normalize_adapter_config(config)


@pytest.mark.parametrize("bad", [np.asarray([np.nan]), np.asarray([np.inf])])
def test_nonfinite_adapter_input_fails_instead_of_being_replaced(bad):
    with pytest.raises(FloatingPointError):
        make(RFF, dimension=1).transform(bad.astype(np.float64))


def test_adapter_rejects_shape_and_dtype_changes():
    adapter = make(IDENTITY, dimension=2)
    with pytest.raises(TypeError, match="float64"):
        adapter.transform(np.ones(2, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        adapter.transform(np.ones((1, 2), dtype=np.float64))
