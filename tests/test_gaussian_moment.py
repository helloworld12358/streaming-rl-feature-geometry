import numpy as np

from streaming_rl_feature_geometry.metrics import cue_probe_score
from streaming_rl_feature_geometry.transforms import FeatureTransform, rep_metrics


def apply_gaussian(stream):
    transform = FeatureTransform(
        "gaussian_moment",
        stream.shape[1],
        min_samples=64,
        update_every=20,
        moment_beta=0.005,
        moment_learning_rate=0.0005,
    )
    output = np.vstack([transform.transform(row) for row in stream])
    return output, transform


def test_gaussian_moment_reduces_skew_and_heavy_tail_error():
    rng = np.random.default_rng(13)
    positive = (rng.exponential(size=7000) - 1.0)[:, None]
    heavy = (rng.standard_t(3, size=7000) / np.sqrt(3.0))[:, None]
    positive_output, _ = apply_gaussian(positive)
    heavy_output, _ = apply_gaussian(heavy)
    assert rep_metrics(positive_output[1500:])["skewness_error"] < rep_metrics(positive[1500:])["skewness_error"]
    assert rep_metrics(heavy_output[1500:])["kurtosis_error"] < rep_metrics(heavy[1500:])["kurtosis_error"]


def test_gaussian_moment_is_finite_and_parameters_are_bounded():
    rng = np.random.default_rng(14)
    stream = rng.normal(size=(5000, 2))
    output, transform = apply_gaussian(stream)
    assert np.isfinite(output).all()
    assert np.all(np.abs(transform.gaussian_skew_parameter) <= 0.75)
    assert np.all((0.6 <= transform.gaussian_tail_parameter) & (transform.gaussian_tail_parameter <= 2.0))


def test_monotone_moment_shaping_retains_bimodal_separation():
    rng = np.random.default_rng(15)
    cues = np.repeat([-1, 1], 3000)
    stream = np.concatenate((rng.normal(-2, 0.3, 3000), rng.normal(2, 0.3, 3000)))[:, None]
    permutation = rng.permutation(len(cues))
    stream, cues = stream[permutation], cues[permutation]
    output, _ = apply_gaussian(stream)
    before = cue_probe_score(stream[1000:], cues[1000:], seed=2)
    after = cue_probe_score(output[1000:], cues[1000:], seed=2)
    assert before > 0.95 and after > 0.95

