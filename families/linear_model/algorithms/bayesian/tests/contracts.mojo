"""BayesianRidge lifecycle, configuration, shape and uncertainty contracts."""
from numojo import NDArray, NDArrayStrides, Shape, zeros
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.base import clone
from moxie.linear_model import BayesianRidge


def check[dtype: DType]() raises:
    var X = zeros[dtype](Shape(8, 2))
    var y = zeros[dtype](Shape(8))
    var weights = zeros[dtype](Shape(8))
    for i in range(8):
        X.unsafe_set(2 * i, Scalar[dtype](80 + i * 2))
        X.unsafe_set(2 * i + 1, Scalar[dtype](i % 3))
        y.unsafe_set(i, Scalar[dtype](5 + i * 3 + i % 2))
        weights.unsafe_set(i, 2)
    var model = BayesianRidge[dtype](compute_score=True, copy_X=False)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(X)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(X, return_std=True)
    _ = model.fit(X, y)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.coef_.ndim, 1)
    assert_equal(model.coef_.size, 2)
    assert_equal(model.intercept_.size, 1)
    assert_equal(model.sigma_.shape[0], 2)
    assert_equal(model.sigma_.shape[1], 2)
    assert_equal(model.scores_.size, model.n_iter_ + 1)
    assert_true(model.score(X, y) > 0.98)
    assert_equal(Float64(X.unsafe_get(0)), 80.0)
    var result = model.predict(X, return_std=True)
    assert_true(result.isa[BayesianRidge[dtype].UncertainPrediction]())
    var ordinary = model.predict(X, return_std=False)
    assert_true(ordinary.isa[NDArray[dtype]]())
    var Z = zeros[dtype](Shape(2, 2))
    Z.unsafe_set(2, model.X_offset_.unsafe_get(0))
    Z.unsafe_set(3, model.X_offset_.unsafe_get(1))
    var uncertainty = model.predict_std(Z)
    assert_true(uncertainty[1].unsafe_get(1) < uncertainty[1].unsafe_get(0))
    assert_almost_equal(Float64(uncertainty[1].unsafe_get(1)) ** 2, 1 / model.alpha_, atol=1e-5)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.coef_.size, 0)
    assert_true(fresh.get_params().compute_score)
    var before = model.predict(X)
    _ = model.fit(X, y)
    for i in range(8):
        assert_almost_equal(Float64(model.predict(X).unsafe_get(i)), Float64(before.unsafe_get(i)), atol=1e-6)
    with assert_raises(contains="single target"):
        _ = model.fit(X, zeros[dtype](Shape(8, 2)))
    assert_true(model.__sklearn_is_fitted__())
    assert_almost_equal(Float64(model.predict(X).unsafe_get(0)), Float64(before.unsafe_get(0)), atol=1e-6)
    with assert_raises():
        _ = model.fit(X, zeros[dtype](Shape(7)))
    with assert_raises():
        _ = model.predict(zeros[dtype](Shape(2, 3)))
    with assert_raises():
        _ = model.fit(X, y, sample_weight=0.0)
    with assert_raises():
        _ = model.fit(X, y, sample_weight=-1.0)
    var bad_y = y.copy()
    bad_y.unsafe_set(2, Scalar[dtype](Float64("nan")))
    with assert_raises():
        _ = model.fit(X, bad_y)
    var params = model.get_params()
    params.max_iter = 0
    _ = fresh.set_params(params)
    with assert_raises(contains="max_iter"):
        _ = fresh.fit(X, y)
    assert_false(fresh.__sklearn_is_fitted__())
    params.max_iter = 1
    params.compute_score = False
    _ = fresh.set_params(params).fit(X, y, sample_weight=2.0)
    assert_equal(fresh.n_iter_, 1)
    assert_equal(fresh.scores_.size, 0)
    _ = model.set_params(params).fit(X, y, sample_weight=weights)
    assert_almost_equal(model.alpha_, fresh.alpha_, atol=1e-6)
    assert_almost_equal(model.lambda_, fresh.lambda_, atol=1e-6)
    _ = model.set_params(max_iter=300, compute_score=True, alpha_init=Optional[Float64](0.5))
    _ = model.fit(X, y)
    assert_equal(model.alpha_init.value(), 0.5)
    var column_y = zeros[dtype](Shape(8, 1))
    for i in range(8):
        column_y.unsafe_set(i, y.unsafe_get(i))
    _ = model.fit(X, column_y)
    assert_equal(model.predict(X).ndim, 1)
    var reversed_X = X.view_with_layout(Shape(8, 2), NDArrayStrides(-2, 1), 14)
    var reversed_y = y.view_with_layout(Shape(8), NDArrayStrides(-1), 7)
    var ordered = model.predict_std(X)
    _ = model.fit(reversed_X, reversed_y)
    var reversed_predictions = model.predict_std(reversed_X)
    for i in range(8):
        assert_almost_equal(Float64(reversed_predictions[0].unsafe_get(i)), Float64(ordered[0].unsafe_get(7 - i)), atol=2e-5)
        assert_almost_equal(Float64(reversed_predictions[1].unsafe_get(i)), Float64(ordered[1].unsafe_get(7 - i)), atol=2e-5)
    _ = model.set_params(compute_score=False, alpha_init=Optional[Float64](0.0)).fit(X, y)
    assert_true(model.alpha_ > 0)
    _ = model.set_params(compute_score=True)
    with assert_raises(contains="positive initial"):
        _ = model.fit(X, y)
    _ = model.set_params(alpha_init=Optional[Float64](0.5))
    # A new fit may replace the feature count; no fitted buffers alias inputs.
    _ = model.fit(zeros[dtype](Shape(8, 3)), y)
    assert_equal(model.n_features_in_, 3)
    assert_equal(model.sigma_.shape[0], 3)
    assert_equal(model.coef_.size, 3)
    var old_intercept = model.intercept_.unsafe_get(0)
    y.unsafe_set(0, 999)
    assert_equal(model.intercept_.unsafe_get(0), old_intercept)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: BayesianRidge lifecycle, shapes, parameters, weights and uncertainty in both dtypes")
