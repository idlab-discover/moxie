"""Estimator lifecycle, weighted objective, warm starts and negative capabilities."""
from numojo import NDArray, zeros, Shape, NDArrayStrides
from std.math import nan
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.base import clone
from moxie.linear_model import ElasticNet, Lasso, CoordinatePathResult, enet_path, lasso_path


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


def path_contracts[dtype: DType]() raises:
    var X = zeros[dtype](Shape(5, 2))
    var y = zeros[dtype](Shape(5))
    for i in range(5):
        X.unsafe_set(2 * i, Scalar[dtype](i))
        X.unsafe_set(2 * i + 1, Scalar[dtype](i % 2))
        y.unsafe_set(i, Scalar[dtype](2 * i + 1))

    var default_path: CoordinatePathResult[dtype] = enet_path(X, y)
    assert_equal(default_path.alphas.shape[0], 100)
    assert_false(Bool(default_path.n_iters))
    assert_equal(default_path.alphas.unsafe_get(0), 28.0)
    assert_equal(default_path.alphas.unsafe_get(99), 0.028)
    assert_equal(default_path.coefs.ndim, 2)
    assert_equal(default_path.coefs.shape[0], 2)
    assert_equal(default_path.coefs.shape[1], 100)
    assert_equal(default_path.dual_gaps.shape[0], 100)

    var alphas = zeros[DType.float64](Shape(3))
    alphas.unsafe_set(0, 0.25)
    alphas.unsafe_set(1, 2.0)
    alphas.unsafe_set(2, 0.5)
    var initial = zeros[dtype](Shape(2))
    initial.unsafe_set(0, Scalar[dtype](0.5))
    var path = lasso_path(
        X, y, alphas=alphas.view(), coef_init=initial.view(), return_n_iter=True
    )
    assert_equal(alphas.unsafe_get(0), 0.25)
    assert_equal(initial.unsafe_get(0), Scalar[dtype](0.5))
    assert_equal(path.alphas.unsafe_get(0), 2.0)
    assert_equal(path.alphas.unsafe_get(2), 0.25)
    assert_equal(len(path.n_iters.value()), 3)

    var reverse_X = X.view_with_layout(Shape(5, 2), NDArrayStrides(-2, 1), 8)
    var reverse_y = y.view_with_layout(Shape(5), NDArrayStrides(-1), 4)
    var initial_storage = zeros[dtype](Shape(2))
    initial_storage.unsafe_set(1, Scalar[dtype](0.5))
    var reverse_initial = initial_storage.view_with_layout(Shape(2), NDArrayStrides(-1), 1)
    var reversed_path = Lasso[dtype].path(
        reverse_X, reverse_y, alphas=alphas.view(), coef_init=reverse_initial^
    )
    for i in range(path.coefs.size):
        assert_almost_equal(
            Float64(path.coefs.unsafe_get(i)),
            Float64(reversed_path.coefs.unsafe_get(i)),
            atol=1e-5,
        )
    assert_equal(initial_storage.unsafe_get(1), Scalar[dtype](0.5))

    var model = ElasticNet[dtype](alpha=0.25, l1_ratio=0.5, fit_intercept=False)
    _ = model.fit(X, y)
    var single_alpha = zeros[DType.float64](Shape(1))
    single_alpha.unsafe_set(0, 0.25)
    var single = ElasticNet[dtype].path(X, y, alphas=single_alpha^)
    for j in range(2):
        assert_equal(single.coefs.unsafe_get(j), model.coef_.unsafe_get(j))
    var automatic_single = lasso_path(X, y, n_alphas=1)
    assert_equal(automatic_single.alphas.unsafe_get(0), 14.0)

    with assert_raises(contains="l1_ratio"):
        _ = enet_path(X, y, l1_ratio=0)
    with assert_raises(contains="alphas must be a nonempty vector"):
        _ = lasso_path(X, y, alphas=zeros[DType.float64](Shape(0)))
    var invalid = zeros[DType.float64](Shape(1))
    invalid.unsafe_set(0, nan[DType.float64]())
    with assert_raises(contains="alphas must be finite and nonnegative"):
        _ = lasso_path(X, y, alphas=invalid.view())
    with assert_raises(contains="alphas must be finite and nonnegative"):
        invalid.unsafe_set(0, -1)
        _ = lasso_path(X, y, alphas=invalid.view())
    with assert_raises(contains="eps and n_alphas"):
        _ = lasso_path(X, y, n_alphas=0)
    with assert_raises(contains="vector y"):
        _ = enet_path(X, y.reshape(Shape(5, 1)))
    with assert_raises(contains="sample counts"):
        _ = enet_path(X, zeros[dtype](Shape(3)))
    with assert_raises(contains="n_features"):
        _ = lasso_path(X, y, coef_init=zeros[dtype](Shape(3)))
    with assert_raises(contains="NaN or infinity"):
        X.unsafe_set(0, nan[dtype]())
        _ = lasso_path(X, y)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("ElasticNet/Lasso lifecycle, weights, shapes, strides and explicit unsupported modes passed")
    path_contracts[DType.float32]()
    path_contracts[DType.float64]()
    print("Dense coordinate path contracts passed in Float32 and Float64")
