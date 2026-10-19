"""GaussianMixture native contracts: parameters, lifecycle, shapes, errors."""
from numojo import NDArray, Shape, zeros, NDArrayStrides
from std.math import abs, isinf
from std.testing import assert_true, assert_false, assert_equal, assert_raises
from moxie._random import RandomState
from moxie.base import clone
from moxie.mixture import GaussianMixture, MixtureDensity, InformationCriteria
from moxie.mixture._scalar_math import log64


def blobs[dtype: DType](n_per: Int = 20) raises -> NDArray[dtype]:
    """Three deterministic 2-D blobs (no RNG), 3 * n_per rows."""
    var X = zeros[dtype](Shape(3 * n_per, 2))
    var centers: List[Float64] = [0.0, 0.0, 6.0, 6.0, 0.0, 8.0]
    for c in range(3):
        for i in range(n_per):
            var t = Float64(i)
            var row = c * n_per + i
            X.unsafe_set(row * 2, (centers[2 * c] + 0.5 * ((t * 0.37) % 1.0 - 0.5)).cast[dtype]())
            X.unsafe_set(row * 2 + 1, (centers[2 * c + 1] + 0.5 * ((t * 0.61) % 1.0 - 0.5)).cast[dtype]())
    return X^


def fits_density[M: MixtureDensity & InformationCriteria](model: M) -> Bool:
    """Compile-time capability check: GaussianMixture is a density with AIC/BIC."""
    return not model.__sklearn_is_fitted__()


def shapes[dtype: DType]() raises:
    var X = blobs[dtype]()
    var kinds: List[String] = ["full", "tied", "diag", "spherical"]
    for kind in kinds:
        var model = GaussianMixture[dtype](3, covariance_type=kind, random_state=0)
        assert_true(fits_density(model))
        var labels = model.fit_predict(X)
        assert_true(model.converged_, kind)
        assert_equal(model.weights_.shape[0], 3)
        assert_equal(model.means_.shape[0], 3)
        assert_equal(model.means_.shape[1], 2)
        var ndim = 3 if kind == "full" else (2 if kind == "tied" or kind == "diag" else 1)
        assert_equal(model.covariances_.ndim, ndim)
        assert_equal(model.precisions_.ndim, ndim)
        assert_equal(model.precisions_cholesky_.ndim, ndim)
        assert_equal(model.n_features_in_, 2)
        assert_equal(len(model.lower_bounds_), model.n_iter_)
        assert_equal(model.lower_bounds_[len(model.lower_bounds_) - 1], model.lower_bound_)
        # fit_predict equals predict on the fitted model; blobs separate.
        var predicted = model.predict(X)
        for i in range(X.shape[0]):
            assert_equal(predicted.unsafe_get(i), labels.unsafe_get(i))
            assert_equal(labels.unsafe_get(i) == labels.unsafe_get(0), i < 20)
        var proba = model.predict_proba(X)
        assert_equal(proba.shape[1], 3)
        for i in range(X.shape[0]):
            var total = 0.0
            for j in range(3):
                total += Float64(proba.unsafe_get(i * 3 + j))
            assert_true(abs(total - 1.0) < 1e-5)
        var scores = model.score_samples(X)
        var mean = 0.0
        for i in range(scores.size):
            mean += Float64(scores.unsafe_get(i))
        mean /= Float64(scores.size)
        assert_true(abs(model.score(X) - mean) < 1e-5 * (1 + abs(mean)), "score is the mean")
        var n = Float64(X.shape[0])
        assert_true(abs((model.bic(X) - model.aic(X)) - Float64(_parameters(kind, 3, 2)) * (_log(n) - 2.0)) < 1e-6 * abs(model.bic(X)))
        var drawn = model.sample(17)
        assert_equal(drawn[0].shape[0], 17)
        assert_equal(drawn[0].shape[1], 2)
        for i in range(1, 17):
            assert_true(drawn[1].unsafe_get(i) >= drawn[1].unsafe_get(i - 1), "samples grouped by component")
        # Integer random_state: sampling repeats.
        var again = model.sample(17)
        for i in range(34):
            assert_equal(again[0].unsafe_get(i), drawn[0].unsafe_get(i))


def _log(x: Float64) -> Float64:
    return log64(x)


def _parameters(kind: String, k: Int, d: Int) -> Int:
    var cov = k * d * (d + 1) // 2 if kind == "full" else (k * d if kind == "diag" else (d * (d + 1) // 2 if kind == "tied" else k))
    return cov + k * d + k - 1


def fit_new(var model: GaussianMixture[DType.float64], X: NDArray[DType.float64]) raises:
    _ = model.fit(X)


def parameters_and_errors() raises:
    var X = blobs[DType.float64]()
    var bad: List[GaussianMixture[DType.float64]] = []
    bad.append(GaussianMixture[DType.float64](0))
    bad.append(GaussianMixture[DType.float64](2, covariance_type="banana"))
    bad.append(GaussianMixture[DType.float64](2, tol=-1.0))
    bad.append(GaussianMixture[DType.float64](2, reg_covar=-1.0))
    bad.append(GaussianMixture[DType.float64](2, max_iter=-1))
    bad.append(GaussianMixture[DType.float64](2, n_init=0))
    bad.append(GaussianMixture[DType.float64](2, init_params="kmeans2"))
    bad.append(GaussianMixture[DType.float64](2, verbose_interval=0))
    for i in range(len(bad)):
        with assert_raises(contains="ValueError: invalid GaussianMixture parameters"):
            _ = bad[i].fit(X)
        assert_false(bad[i].__sklearn_is_fitted__())
    var model = GaussianMixture[DType.float64](2, random_state=0)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(X)
    with assert_raises(contains="NotFittedError"):
        _ = model.score_samples(X)
    with assert_raises(contains="NotFittedError"):
        _ = model.sample(3)
    with assert_raises(contains="ValueError"):
        _ = model.fit(zeros[DType.float64](Shape(1, 2)))  # fewer than 2 samples
    with assert_raises(contains="n_samples >= n_components"):
        fit_new(GaussianMixture[DType.float64](5), zeros[DType.float64](Shape(4, 2)))
    var with_nan = X.copy()
    with_nan.unsafe_set(3, Float64(0) / Float64(0))
    with assert_raises(contains="finite"):
        _ = model.fit(with_nan)
    _ = model.fit(X)
    with assert_raises(contains="feature count"):
        _ = model.predict(zeros[DType.float64](Shape(3, 3)))
    with assert_raises(contains="Invalid value for 'n_samples'"):
        _ = model.sample(0)
    # Initialization array validation.
    var w = zeros[DType.float64](Shape(2))
    w.unsafe_set(0, 0.7)
    w.unsafe_set(1, 0.7)
    with assert_raises(contains="should be normalized"):
        fit_new(GaussianMixture[DType.float64](2, weights_init=w.copy()), X)
    with assert_raises(contains="should have the shape"):
        fit_new(GaussianMixture[DType.float64](3, weights_init=w.copy()), X)
    w.unsafe_set(0, 1.3)
    w.unsafe_set(1, -0.3)
    with assert_raises(contains="range [0, 1]"):
        fit_new(GaussianMixture[DType.float64](2, weights_init=w.copy()), X)
    with assert_raises(contains="'means' should have the shape"):
        fit_new(GaussianMixture[DType.float64](2, means_init=zeros[DType.float64](Shape(2, 3))), X)
    var not_pd = zeros[DType.float64](Shape(2, 2, 2))
    with assert_raises(contains="symmetric, positive-definite"):
        fit_new(GaussianMixture[DType.float64](2, precisions_init=not_pd^), X)
    with assert_raises(contains="should be positive"):
        fit_new(GaussianMixture[DType.float64](2, covariance_type="diag", precisions_init=zeros[DType.float64](Shape(2, 2))), X)
    # Collapsed data without regularization raises the reference message.
    var collapsed = zeros[DType.float64](Shape(6, 2))
    with assert_raises(contains="ill-defined empirical covariance"):
        fit_new(GaussianMixture[DType.float64](2, reg_covar=0.0, random_state=0), collapsed)


def lifecycle() raises:
    var X = blobs[DType.float64]()
    # Repeat fits with an integer seed are identical; clone is unfitted.
    var model = GaussianMixture[DType.float64](3, covariance_type="diag", init_params="random", random_state=4)
    _ = model.fit(X)
    var first = model.means_.copy()
    var first_bound = model.lower_bound_
    _ = model.fit(X)
    for i in range(first.size):
        assert_equal(first.unsafe_get(i), model.means_.unsafe_get(i))
    assert_equal(first_bound, model.lower_bound_)
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.covariance_type, "diag")
    assert_equal(copy.random_state.value(), 4)
    assert_equal(copy.weights_.size, 0)
    _ = copy.fit(X)
    assert_equal(copy.lower_bound_, model.lower_bound_)
    # set_params changes configuration only.
    var params = model.get_params()
    params.n_components = 2
    params.covariance_type = "spherical"
    _ = model.set_params(params)
    assert_equal(model.n_components, 2)
    assert_true(model.__sklearn_is_fitted__(), "set_params keeps learned state")
    _ = model.fit(X)
    assert_equal(model.weights_.size, 2)
    # Borrowed RandomState advances across fits (instance semantics).
    var rng = RandomState(9)
    var borrowed = GaussianMixture[DType.float64](3, init_params="random")
    _ = borrowed.fit_with_rng(X, rng)
    var bound_a = borrowed.lower_bound_
    var fresh = RandomState(9)
    var other = GaussianMixture[DType.float64](3, init_params="random")
    _ = other.fit_with_rng(X, fresh)
    assert_equal(bound_a, other.lower_bound_)
    var before = rng.random_sample()
    var reference = RandomState(9)
    var replay = GaussianMixture[DType.float64](3, init_params="random")
    _ = replay.fit_with_rng(X, reference)
    assert_equal(before, reference.random_sample())
    # max_iter = 0: initialization only.
    var init_only = GaussianMixture[DType.float64](3, max_iter=0, random_state=0)
    _ = init_only.fit(X)
    assert_equal(init_only.n_iter_, 0)
    assert_true(isinf(init_only.lower_bound_) and init_only.lower_bound_ < 0)
    assert_equal(len(init_only.lower_bounds_), 0)
    assert_false(init_only.converged_)
    assert_equal(init_only.convergence_warning_, "")
    # Nonconvergence keeps the fit and reports the warning.
    var short = GaussianMixture[DType.float64](3, max_iter=1, tol=0.0, random_state=0)
    _ = short.fit(X)
    assert_false(short.converged_)
    assert_true(short.convergence_warning_.startswith("ConvergenceWarning"))
    assert_equal(short.n_iter_, 1)
    # Warm start resumes: each fit is one restart continuing the objective.
    var warm = GaussianMixture[DType.float64](3, max_iter=1, tol=0.0, warm_start=True, random_state=0)
    _ = warm.fit(X)
    var previous = warm.lower_bound_
    for _ in range(3):
        _ = warm.fit(X)
        assert_equal(warm.n_iter_, 1)
        assert_true(warm.lower_bound_ >= previous - 1e-12, "EM objective does not decrease")
        previous = warm.lower_bound_
    # All three init arrays: no responsibilities, deterministic without a seed.
    var fitted = GaussianMixture[DType.float64](3, random_state=0)
    _ = fitted.fit(X)
    var given = GaussianMixture[DType.float64](3, weights_init=fitted.weights_.copy(), means_init=fitted.means_.copy(), precisions_init=fitted.precisions_.copy())
    _ = given.fit(X)
    assert_true(abs(given.lower_bound_ - fitted.lower_bound_) < 1e-6)
    # n_init restarts keep the best objective.
    var restarts = GaussianMixture[DType.float64](3, n_init=3, init_params="random_from_data", random_state=2)
    _ = restarts.fit(X)
    var single = GaussianMixture[DType.float64](3, init_params="random_from_data", random_state=2)
    _ = single.fit(X)
    # The first restart consumes the same stream as the single-init fit.
    assert_true(restarts.lower_bound_ >= single.lower_bound_)
    # Strided input (negative column stride view) matches its compact copy.
    var wide = zeros[DType.float64](Shape(X.shape[0], 4))
    for i in range(X.shape[0]):
        wide.unsafe_set(i * 4 + 3, X.unsafe_get(i * 2))
        wide.unsafe_set(i * 4 + 1, X.unsafe_get(i * 2 + 1))
    var view = wide.view_with_layout(Shape(X.shape[0], 2), NDArrayStrides(4, -2), 3)
    var strided = GaussianMixture[DType.float64](3, random_state=0)
    _ = strided.fit(view)
    var compact = GaussianMixture[DType.float64](3, random_state=0)
    _ = compact.fit(X)
    for i in range(compact.means_.size):
        assert_true(abs(Float64(strided.means_.unsafe_get(i) - compact.means_.unsafe_get(i))) < 1e-12)
    # Learned arrays are owned: mutating the input after fit changes nothing.
    var owned = X.copy()
    var snapshot_model = GaussianMixture[DType.float64](3, random_state=0)
    _ = snapshot_model.fit(owned)
    var saved = snapshot_model.means_.unsafe_get(0)
    owned.unsafe_set(0, 1e6)
    assert_equal(snapshot_model.means_.unsafe_get(0), saved)


def main() raises:
    shapes[DType.float64]()
    shapes[DType.float32]()
    parameters_and_errors()
    lifecycle()
    print("PASS GaussianMixture contracts: shapes, dtypes, parameters, errors, lifecycle, warm start, cloning")
