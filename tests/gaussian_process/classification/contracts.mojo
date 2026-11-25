from numojo import NDArray, NDArrayStrides, Shape, zeros
from std.math import abs, nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true, TestSuite

from moxie.base import clone
from moxie.gaussian_process import GaussianProcessClassifier
from moxie.gaussian_process.kernels import CompoundKernel, Kernel, PairwiseKernel, RBF, WhiteKernel


def test_fit_and_clone() raises:
    var X = zeros[DType.float64](Shape(4, 1))
    X.unsafe_set(0, -2.0)
    X.unsafe_set(1, -1.0)
    X.unsafe_set(2, 1.0)
    X.unsafe_set(3, 2.0)
    var y = zeros[DType.int64](Shape(4))
    y.unsafe_set(0, 9)
    y.unsafe_set(1, 9)
    y.unsafe_set(2, -7)
    y.unsafe_set(3, -7)

    var model = GaussianProcessClassifier()
    with assert_raises():
        _ = model.predict(X)
    var prediction = model.set_params(copy_X_train=False).fit(X, y).predict(X)
    assert_equal(prediction.unsafe_get(0), 9)
    assert_equal(prediction.unsafe_get(3), -7)
    assert_equal(model.n_classes_(), 2)
    assert_equal(model.classes_().unsafe_get(0), -7)
    assert_equal(model.n_features_in_, 1)
    assert_equal(model.score(X, y), 1.0)
    assert_true(model.__sklearn_is_fitted__())
    assert_true(model.copy().__sklearn_is_fitted__())
    assert_false(clone(model).__sklearn_is_fitted__())
    assert_false(clone(model).get_params().copy_X_train)

    var proba = model.predict_proba(X)
    assert_equal(proba.shape[1], 2)
    for i in range(4):
        assert_equal(proba.unsafe_get(2 * i) + proba.unsafe_get(2 * i + 1), 1.0)
    assert_true(proba.unsafe_get(1) > 0.5)
    assert_true(proba.unsafe_get(7) < 0.5)
    var query = X.copy()
    X.unsafe_set(0, 10.0)
    assert_equal(model.predict(query).unsafe_get(0), 9)
    with assert_raises():
        _ = model.fit(query, zeros[DType.int64](Shape(4)))
    assert_equal(model.predict(query).unsafe_get(0), 9)
    with assert_raises():
        _ = model.set_params(max_iter_predict=0)
    assert_equal(model.get_params().max_iter_predict, 100)
    with assert_raises():
        _ = model.predict(zeros[DType.float64](Shape(1, 2)))
    with assert_raises():
        _ = model.log_marginal_likelihood(eval_gradient=True)

    y.unsafe_set(0, -10)
    y.unsafe_set(1, -10)
    _ = model.fit(query, y.reshape(Shape(4, 1)))
    assert_equal(model.classes_().unsafe_get(0), -10)
    assert_equal(model.predict(query).unsafe_get(0), -10)


def test_float32_strided_and_weighted() raises:
    var owner = zeros[DType.float32](Shape(4, 2))
    owner.unsafe_set(0, -2.0)
    owner.unsafe_set(2, -1.0)
    owner.unsafe_set(4, 1.0)
    owner.unsafe_set(6, 2.0)
    var X = owner.view_with_layout(Shape(4, 1), NDArrayStrides(2, 1), 0)
    var y = zeros[DType.float32](Shape(4))
    y.unsafe_set(2, 3.0)
    y.unsafe_set(3, 3.0)
    var model = GaussianProcessClassifier[DType.float32, DType.float32](
        RBF(1.0, "fixed"), optimizer=None
    )
    var prediction: NDArray[DType.float32] = model.fit(X, y).predict(X)
    assert_equal(prediction.unsafe_get(0), 0.0)
    assert_equal(prediction.unsafe_get(3), 3.0)

    var weights = zeros[DType.float64](Shape(4))
    weights.unsafe_set(0, 3.0)
    weights.unsafe_set(1, 1.0)
    y.unsafe_set(0, 3.0)
    assert_equal(model.score(X, y, weights^), 0.25)
    y.unsafe_set(0, 0.25)
    with assert_raises():
        _ = model.fit(X, y)
    owner.unsafe_set(0, nan[DType.float32]())
    with assert_raises():
        _ = model.predict(X)


def test_fit_errors_and_zero_latent_mean() raises:
    var X = zeros[DType.float64](Shape(3, 1))
    var y = zeros[DType.int64](Shape(3))
    y.unsafe_set(1, 1)
    y.unsafe_set(2, 2)
    var model = GaussianProcessClassifier()
    with assert_raises():
        _ = model.fit(X, y)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.fit(X, zeros[DType.int64](Shape(2)))
    with assert_raises():
        _ = model.fit(X, zeros[DType.int64](Shape(3, 2)))

    y.unsafe_set(2, 1)
    var components: List[Kernel] = [RBF().as_kernel()]
    var compound = CompoundKernel(components^)
    _ = model.set_params(kernel=Optional[Kernel](compound^))
    with assert_raises():
        _ = model.fit(X, y)

    var white = GaussianProcessClassifier(WhiteKernel(1.0, "fixed"))
    _ = white.fit(X, y)
    var latent = white.latent_mean_and_variance(X)
    assert_equal(latent.mean.unsafe_get(0), 0.0)
    assert_equal(latent.variance.unsafe_get(0), 1.0)
    assert_equal(white.predict(X).unsafe_get(0), 0)

    var zero = GaussianProcessClassifier(PairwiseKernel(metric="linear"), optimizer=None)
    _ = zero.fit(X, y)
    assert_equal(zero.latent_mean_and_variance(X).variance.unsafe_get(0), 0.0)
    var probability = zero.predict_proba(X).unsafe_get(1)
    assert_true(abs(probability - 0.5) < 1e-7)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
