import numpy as np

from streaming_rl_feature_geometry.transforms import FeatureTransform, rep_metrics


def transformed(kind, data):
    transform = FeatureTransform(kind, data.shape[1], min_samples=40, update_every=20)
    return np.vstack([transform.transform(row) for row in data])


def test_standardization_handles_constant_shifted_and_scaled_streams():
    constant = np.ones((300, 2)) * [4.0, -2.0]
    assert np.isfinite(transformed("standardized", constant)).all()
    rng = np.random.default_rng(4)
    stream = rng.normal(loc=[8.0, -3.0], scale=[5.0, 0.2], size=(1200, 2))
    output = transformed("standardized", stream)[300:]
    metrics = rep_metrics(output)
    assert metrics["mean_error"] < 0.35
    assert 0.75 < metrics["variance_mean"] < 1.25


def test_rms_raw_preserves_correlation_and_matches_scale():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(1200, 2))
    stream = base @ np.array([[4.0, 2.0], [0.0, 0.2]])
    output = transformed("rms_raw", stream)[300:]
    before = rep_metrics(stream[300:])
    after = rep_metrics(output)
    assert abs(after["mean_abs_corr"] - before["mean_abs_corr"]) < 0.03
    assert 0.7 < np.sqrt(after["variance_mean"]) < 1.3


def test_decorrelation_and_whitening_fulfill_distinct_properties():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(1800, 3))
    stream = base @ np.array([[3.0, 1.5, 0.8], [0.0, 0.3, 0.2], [0.0, 0.0, 0.1]])
    raw = rep_metrics(stream[400:])
    decorrelated = rep_metrics(transformed("decorrelated", stream)[400:])
    whitened = rep_metrics(transformed("whitened", stream)[400:])
    assert decorrelated["mean_abs_corr"] < raw["mean_abs_corr"]
    assert whitened["isotropy_error"] < raw["isotropy_error"]


def test_whitening_is_finite_for_singular_and_repeated_features():
    values = np.linspace(-1, 1, 500)
    stream = np.column_stack((values, values, np.ones_like(values)))
    transform = FeatureTransform("whitened", 3, min_samples=40, update_every=20)
    output = np.vstack([transform.transform(row) for row in stream])
    assert np.isfinite(output).all()
    metrics = transform.state_metrics()
    assert 0.0 <= metrics["covariance_min_eigenvalue"] <= metrics["covariance_max_eigenvalue"]
    assert metrics["regularized_min_eigenvalue"] >= transform.eps
    assert np.isfinite(metrics["whitening_gain"])


def test_unit_sphere_has_unit_norm_after_nonzero_warmup():
    rng = np.random.default_rng(7)
    stream = rng.normal(size=(500, 4))
    output = transformed("unit_sphere", stream)
    norms = np.linalg.norm(output, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01)
