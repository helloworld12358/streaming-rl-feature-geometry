import numpy as np

from streaming_rl_feature_geometry.transforms import FeatureTransform


def test_transform_prefix_does_not_depend_on_future_samples():
    prefix = [np.array([i, -0.5 * i], dtype=float) for i in range(20)]
    first = FeatureTransform("whitened", 2, min_samples=8, update_every=4)
    second = FeatureTransform("whitened", 2, min_samples=8, update_every=4)
    first_prefix = [first.transform(value) for value in prefix]
    second_prefix = [second.transform(value) for value in prefix]
    second.transform(np.array([1e6, -1e6]))
    assert np.allclose(first_prefix, second_prefix)


def test_current_sample_updates_statistics_only_after_output():
    transform = FeatureTransform("standardized", 2)
    sample = np.array([7.0, -3.0])
    output = transform.transform(sample)
    assert np.array_equal(output, sample)
    assert transform.stats.n == 1
    assert np.array_equal(transform.stats.mean, sample)

