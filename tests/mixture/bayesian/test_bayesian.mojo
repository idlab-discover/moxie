"""BayesianGaussianMixture native contracts and variational-update checks."""
from numojo import NDArray, Shape, zeros
from std.math import abs, isinf
from std.testing import assert_true, assert_false, assert_equal, assert_raises
from moxie._random import RandomState
from moxie.base import clone
from moxie.mixture import BayesianGaussianMixture, GaussianMixture, MixtureDensity
from moxie._mixture_linalg.special_gamma import digamma


def blobs[dtype: DType]() raises -> NDArray[dtype]:
    var rng = RandomState(1)
    var X = zeros[dtype](Shape(90, 2))
    for i in range(90):
        var c = Float64(i // 30) * 6.0
        X.unsafe_set(i * 2, (c + 0.5 * rng.standard_normal()).cast[dtype]())
        X.unsafe_set(i * 2 + 1, (Float64((i // 30) % 2) * 6.0 + 0.5 * rng.standard_normal()).cast[dtype]())
    return X^


def is_density[M: MixtureDensity](model: M) -> Bool:
    return not model.__sklearn_is_fitted__()


def fit_new(var model: BayesianGaussianMixture[DType.float64], X: NDArray[DType.float64]) raises:
    _ = model.fit(X)


def shapes[dtype: DType]() raises:
    var X = blobs[dtype]()
    var kinds: List[String] = ["full", "tied", "diag", "spherical"]
    var priors: List[String] = ["dirichlet_process", "dirichlet_distribution"]
    for prior in priors:
        for kind in kinds:
            var model = BayesianGaussianMixture[dtype](n_components=4, covariance_type=kind, weight_concentration_prior_type=prior, max_iter=300, random_state=0)
            assert_true(is_density(model))
            var labels = model.fit_predict(X)
            var predicted = model.predict(X)
            for i in range(90):
                assert_equal(labels.unsafe_get(i), predicted.unsafe_get(i))
            assert_equal(len(model.weight_concentration_), 2 if prior == "dirichlet_process" else 1)
            assert_equal(model.mean_precision_.size, 4)
            assert_equal(model.degrees_of_freedom_.size, 1 if kind == "tied" else 4)
            assert_equal(model.mean_prior_.size, 2)
            assert_equal(model.covariance_prior_.size, 4 if kind == "full" or kind == "tied" else (2 if kind == "diag" else 1))
            assert_true(abs(model.weight_concentration_prior_ - 0.25) < 1e-15, "default 1/k")
            assert_equal(model.mean_precision_prior_, 1.0)
            assert_equal(model.degrees_of_freedom_prior_, 2.0)
            var total = 0.0
            for j in range(4):
                total += Float64(model.weights_.unsafe_get(j))
            assert_true(abs(total - 1.0) < 1e-5)
            # Variational lower bound is nondecreasing along the fit.
            for i in range(1, len(model.lower_bounds_)):
                assert_true(model.lower_bounds_[i] >= model.lower_bounds_[i - 1] - 1e-6 * abs(model.lower_bounds_[i]), kind + " " + prior)
            var proba = model.predict_proba(X)
            assert_equal(proba.shape[1], 4)
            var drawn = model.sample(9)
            assert_equal(drawn[0].shape[0], 9)


def parameters_and_errors() raises:
    var X = blobs[DType.float64]()
    with assert_raises(contains="weight_concentration_prior_type"):
        fit_new(BayesianGaussianMixture[DType.float64](weight_concentration_prior_type="dirichlet"), X)
    with assert_raises(contains="weight_concentration_prior"):
        fit_new(BayesianGaussianMixture[DType.float64](weight_concentration_prior=0.0), X)
    with assert_raises(contains="mean_precision_prior"):
        fit_new(BayesianGaussianMixture[DType.float64](mean_precision_prior=-1.0), X)
    with assert_raises(contains="should be greater than 1"):
        fit_new(BayesianGaussianMixture[DType.float64](degrees_of_freedom_prior=0.5), X)
    with assert_raises(contains="'means' should have the shape"):
        fit_new(BayesianGaussianMixture[DType.float64](mean_prior=zeros[DType.float64](Shape(3))), X)
    var negative = zeros[DType.float64](Shape(2, 2))
    negative.unsafe_set(0, -1)
    negative.unsafe_set(3, -1)
    with assert_raises(contains="symmetric, positive-definite"):
        fit_new(BayesianGaussianMixture[DType.float64](covariance_prior=negative^), X)
    with assert_raises(contains="should be positive"):
        fit_new(BayesianGaussianMixture[DType.float64](covariance_type="diag", covariance_prior=zeros[DType.float64](Shape(2))), X)
    with assert_raises(contains="ValueError: invalid BayesianGaussianMixture parameters"):
        fit_new(BayesianGaussianMixture[DType.float64](n_components=0), X)
    var model = BayesianGaussianMixture[DType.float64](n_components=2)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(X)
    with assert_raises(contains="NotFittedError"):
        _ = model.sample(2)


def lifecycle() raises:
    var X = blobs[DType.float64]()
    var model = BayesianGaussianMixture[DType.float64](n_components=3, random_state=2, weight_concentration_prior=0.5)
    _ = model.fit(X)
    var bound = model.lower_bound_
    _ = model.fit(X)
    assert_equal(bound, model.lower_bound_, "integer seed repeats")
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.weight_concentration_prior.value(), 0.5)
    var params = model.get_params()
    params.covariance_type = "diag"
    _ = model.set_params(params)
    assert_equal(model.covariance_type, "diag")
    # Keyword set_params: partial, fluent, keeps learned state until the refit.
    var fitted_bound = model.lower_bound_
    _ = model.set_params(weight_concentration_prior_type="dirichlet_distribution", weight_concentration_prior=2.0,
                         mean_precision_prior=0.8, warm_start=False)
    assert_equal(model.weight_concentration_prior_type, "dirichlet_distribution")
    assert_equal(model.weight_concentration_prior.value(), 2.0)
    assert_equal(model.mean_precision_prior.value(), 0.8)
    assert_equal(model.covariance_type, "diag")
    assert_equal(model.lower_bound_, fitted_bound, "keyword set_params keeps learned state")
    _ = model.set_params(n_components=2).fit(X)
    var configured = BayesianGaussianMixture[DType.float64](
        n_components=2, covariance_type="diag", random_state=2, weight_concentration_prior_type="dirichlet_distribution",
        weight_concentration_prior=2.0, mean_precision_prior=0.8)
    _ = configured.fit(X)
    assert_equal(model.lower_bound_, configured.lower_bound_, "keyword refit equals constructor configuration")
    with assert_raises(contains="weight_concentration_prior_type"):
        _ = model.set_params(weight_concentration_prior_type="bogus").fit(X)
    # max_iter = 0, nonconvergence, warm start.
    var init_only = BayesianGaussianMixture[DType.float64](n_components=3, max_iter=0, random_state=0)
    _ = init_only.fit(X)
    assert_equal(init_only.n_iter_, 0)
    assert_true(isinf(init_only.lower_bound_))
    assert_equal(init_only.convergence_warning_, "")
    var short = BayesianGaussianMixture[DType.float64](n_components=3, max_iter=1, tol=0.0, random_state=0)
    _ = short.fit(X)
    assert_true(short.convergence_warning_.startswith("ConvergenceWarning"))
    var warm = BayesianGaussianMixture[DType.float64](n_components=3, max_iter=1, tol=0.0, warm_start=True, random_state=0)
    _ = warm.fit(X)
    for _ in range(3):
        var previous = warm.lower_bound_
        _ = warm.fit(X)
        assert_equal(warm.n_iter_, 1)
        assert_true(warm.lower_bound_ >= previous - 1e-9)
    # Borrowed RNG: same stream, same result.
    var r1 = RandomState(4)
    var r2 = RandomState(4)
    var a = BayesianGaussianMixture[DType.float64](n_components=3, init_params="random")
    var b = BayesianGaussianMixture[DType.float64](n_components=3, init_params="random")
    _ = a.fit_with_rng(X, r1)
    _ = b.fit_with_rng(X, r2)
    assert_equal(a.lower_bound_, b.lower_bound_)
    # Both estimators share the same pinned EM/initialization: with a strong
    # data-dominated posterior, the Bayesian labels agree with GaussianMixture.
    var gm = GaussianMixture[DType.float64](3, random_state=0)
    var bgm = BayesianGaussianMixture[DType.float64](n_components=3, random_state=0, weight_concentration_prior_type="dirichlet_distribution", weight_concentration_prior=1000.0)
    var la = gm.fit_predict(X)
    var lb = bgm.fit_predict(X)
    for i in range(90):
        assert_equal(la.unsafe_get(i), lb.unsafe_get(i))
    # Special function used by the expected log weights.
    assert_true(abs(digamma(1.0) + 0.5772156649015329) < 1e-15)


def main() raises:
    shapes[DType.float64]()
    shapes[DType.float32]()
    parameters_and_errors()
    lifecycle()
    print("PASS BayesianGaussianMixture contracts: priors, shapes, dtypes, errors, lifecycle, shared EM")
