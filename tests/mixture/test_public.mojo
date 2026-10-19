"""Concrete public imports and fluent trait dispatch in both native dtypes."""
from numojo import NDArray, arange, Shape, NDArrayStrides, zeros
from std.math import abs
from std.testing import assert_equal, assert_true, assert_false
from moxie.base import clone
from moxie._random import RandomState
from moxie.mixture import (
    GaussianMixture, GaussianMixtureParams,
    BayesianGaussianMixture, BayesianGaussianMixtureParams, MixtureDensity,
)


def generic_fit[M: MixtureDensity](
    mut model: M, params: M.Params, X: M.Input,
) raises -> M.Prediction where conforms_to(M.Target, Deinitable):
    return model.set_params(params).fit(X).predict(X)


def check[dtype: DType]() raises:
    var X = arange[dtype](12).reshape(Shape(6, 2))
    var gaussian = GaussianMixture[dtype](covariance_type="diag", random_state=7)
    var bayesian = BayesianGaussianMixture[dtype](covariance_type="diag", random_state=7)
    var gp: GaussianMixtureParams[dtype] = gaussian.get_params()
    var bp: BayesianGaussianMixtureParams[dtype] = bayesian.get_params()
    _ = gaussian.set_params(gp).fit(X)
    _ = bayesian.set_params(bp).fit(X)
    assert_true(abs(Float64(gaussian.means_.item(0, 0)) - 5) < 1e-5)
    assert_true(abs(Float64(gaussian.means_.item(0, 1)) - 6) < 1e-5)
    assert_true(abs(Float64(bayesian.means_.item(0, 0)) - 5) < 1e-5)
    var gaussian_labels = generic_fit(gaussian, gp, X)
    var bayesian_labels = generic_fit(bayesian, bp, X)
    assert_equal(gaussian_labels.shape[0], 6)
    assert_equal(bayesian_labels.shape[0], 6)
    var wide = zeros[dtype](Shape(6, 4))
    for i in range(6):
        wide.unsafe_set(i * 4 + 3, X.unsafe_get(i * 2))
        wide.unsafe_set(i * 4 + 1, X.unsafe_get(i * 2 + 1))
    var view = wide.view_with_layout(Shape(6, 2), NDArrayStrides(4, -2), 3)
    var strided_gaussian = clone(gaussian)
    var strided_bayesian = clone(bayesian)
    _ = strided_gaussian.fit(view)
    _ = strided_bayesian.fit(view)
    assert_equal(strided_gaussian.lower_bound_, gaussian.lower_bound_)
    assert_equal(strided_bayesian.lower_bound_, bayesian.lower_bound_)
    wide.unsafe_set(3, 99)
    assert_true(abs(Float64(strided_gaussian.means_.item(0, 0)) - 5) < 1e-5)
    assert_true(abs(Float64(strided_bayesian.means_.item(0, 0)) - 5) < 1e-5)
    var fresh = clone(gaussian)
    assert_false(fresh.__sklearn_is_fitted__())
    var fresh_bayesian = clone(bayesian)
    assert_false(fresh_bayesian.__sklearn_is_fitted__())
    var copied = gaussian.copy()
    copied.means_.unsafe_set(0, 99)
    assert_true(abs(Float64(gaussian.means_.item(0, 0)) - 5) < 1e-5)
    var copied_bayesian = bayesian.copy()
    copied_bayesian.means_.unsafe_set(0, 99)
    assert_true(abs(Float64(bayesian.means_.item(0, 0)) - 5) < 1e-5)
    var rng = RandomState(19)
    var samples: GaussianMixture[dtype].Samples = gaussian.sample_with_rng(4, rng)
    var samples_bayesian: BayesianGaussianMixture[dtype].Samples = bayesian.sample_with_rng(4, rng)
    assert_equal(samples[0].shape[0], 4)
    assert_equal(samples_bayesian[0].shape[0], 4)
    var replay_rng = RandomState(19)
    var replay_samples = gaussian.sample_with_rng(4, replay_rng)
    var replay_bayesian = bayesian.sample_with_rng(4, replay_rng)
    assert_equal(rng.next_uint32(), replay_rng.next_uint32())
    for i in range(samples[0].size):
        assert_equal(samples[0].unsafe_get(i), replay_samples[0].unsafe_get(i))
        assert_equal(samples_bayesian[0].unsafe_get(i), replay_bayesian[0].unsafe_get(i))
    var proba = gaussian.predict_proba(X)
    var bayesian_proba = bayesian.predict_proba(X)
    assert_equal(proba.shape[1], 1)
    assert_equal(bayesian_proba.shape[1], 1)
    assert_true(gaussian.aic(X) > 0)
    assert_true(gaussian.bic(X) > 0)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS concrete public mixture estimators, fluent dispatch and independent state")
