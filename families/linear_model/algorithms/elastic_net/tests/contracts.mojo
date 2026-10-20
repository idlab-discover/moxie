"""Estimator lifecycle, weighted objective, warm starts and negative capabilities."""
from numojo import NDArray, zeros, Shape, NDArrayStrides
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.base import clone
from moxie.linear_model import ElasticNet, Lasso


def contracts[dtype: DType]() raises:
    var X = zeros[dtype](Shape(5, 2))
    var y = zeros[dtype](Shape(5))
    for i in range(5):
        X.unsafe_set(2 * i, Scalar[dtype](i))
        X.unsafe_set(2 * i + 1, Scalar[dtype](i % 2))
        y.unsafe_set(i, Scalar[dtype](2 * i + 1))
    var model = ElasticNet[dtype](alpha=0.1, tol=1e-7)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(X)
    model.fit(X, y)
    assert_true(model.converged_[0])
    assert_true(model.score(X, y) > 0.99)
    var copied = clone(model)
    assert_false(copied.__sklearn_is_fitted__())
    assert_equal(copied.alpha, 0.1)
    var saved = model.coef_.copy()
    model.set_params(warm_start=Optional(True))
    model.fit(X, y)
    assert_almost_equal(Float64(saved.unsafe_get(0)), Float64(model.coef_.unsafe_get(0)), atol=1e-6)
    var weights = zeros[dtype](Shape(5))
    var scaled = zeros[dtype](Shape(5))
    for i in range(5):
        weights.unsafe_set(i, Scalar[dtype](i + 1))
        scaled.unsafe_set(i, Scalar[dtype](10 * (i + 1)))
    var a = Lasso[dtype](alpha=0.2)
    var b = Lasso[dtype](alpha=0.2)
    a.fit(X, y, sample_weight=weights)
    b.fit(X, y, sample_weight=scaled)
    assert_almost_equal(Float64(a.coef_.unsafe_get(0)), Float64(b.coef_.unsafe_get(0)), atol=1e-6)
    # Scalar weights cancel under the reference's average-loss normalization.
    b.fit(X, y, sample_weight=4.0)
    a.fit(X, y)
    assert_equal(a.coef_.unsafe_get(0), b.coef_.unsafe_get(0))
    # One-column targets are raveled, multiple independent outputs stay matrices.
    var column = y.reshape(Shape(5, 1))
    a.fit(X, column)
    assert_equal(a.coef_.ndim, 1)
    assert_equal(a.predict(X).ndim, 1)
    var Y = zeros[dtype](Shape(5, 2))
    for i in range(5):
        Y.unsafe_set(2 * i, y.unsafe_get(i))
        Y.unsafe_set(2 * i + 1, -y.unsafe_get(i))
    a.fit(X, Y)
    assert_equal(a.coef_.shape[0], 2)
    assert_equal(len(a.n_iter_), 2)
    assert_equal(a.predict(X).shape[1], 2)
    var positive = ElasticNet[dtype](alpha=0.01, positive=True)
    positive.fit(X, Y)
    for i in range(positive.coef_.size):
        assert_true(positive.coef_.unsafe_get(i) >= 0)
    with assert_raises():
        a.fit(X, Y, sample_weight=zeros[dtype](Shape(5)))
    assert_equal(a.coef_.shape[0], 2)
    with assert_raises(contains="features"):
        _ = a.predict(zeros[dtype](Shape(2, 3)))
    with assert_raises(contains="warm_start"):
        model.fit(X, Y)
    with assert_raises(contains="NotImplementedError"):
        var random = Lasso[dtype](selection="random")
        random.fit(X, y)
    with assert_raises(contains="NotImplementedError"):
        var gram = Lasso[dtype](precompute=True)
        gram.fit(X, y)
    with assert_raises(contains="alpha"):
        var invalid = ElasticNet[dtype](alpha=-1)
        invalid.fit(X, y)
    # A reversed sample view must preserve the fit.
    var reversed_X = X.view_with_layout(Shape(5, 2), NDArrayStrides(-2, 1), 8)
    var reversed_y = y.view_with_layout(Shape(5), NDArrayStrides(-1), 4)
    b.fit(reversed_X, reversed_y)
    var ordinary = Lasso[dtype](alpha=0.2)
    ordinary.fit(X, y)
    assert_almost_equal(Float64(b.coef_.unsafe_get(0)), Float64(ordinary.coef_.unsafe_get(0)), atol=1e-6)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("ElasticNet/Lasso lifecycle, weights, shapes, strides and explicit unsupported modes passed")
