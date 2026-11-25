from std.testing import TestSuite, assert_equal, assert_false, assert_raises, assert_true

from moxie.base import SupervisedEstimator, clone
from moxie.ensemble import HistGradientBoostingRegressor
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from numojo import NDArray, Shape, Strides, full


def fit_generic[E: SupervisedEstimator](mut model: E, X: E.Input, y: E.Target) raises:
    _ = model.fit(X, y)


def test_fit_clone_and_parameter_changes() raises:
    var X = full[DType.float64](Shape(8, 2), 0.0)
    var y = NDArray[DType.float64](Shape(8))
    for i in range(8):
        X.unsafe_set(2 * i, Float64(i))
        y.unsafe_set(i, Float64(i))
    var model = HistGradientBoostingRegressor(max_iter=4, min_samples_leaf=2, random_state=10)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(X)
    with assert_raises(contains="NotFittedError"):
        _ = model.staged_predict(X)
    fit_generic(model, X, y)
    assert_equal(model.n_iter_, 4)
    assert_equal(model.n_trees_per_iteration_, 1)
    assert_equal(model.n_features_in_, 2)
    assert_false(model.do_early_stopping_)
    assert_false(Bool(model.is_categorical_))
    assert_equal(model.train_score_.size, 0)
    assert_equal(model.validation_score_.size, 0)
    assert_true(model.score(X, y) > 0.0)

    var copied = clone(model)
    assert_equal(copied.get_params().max_iter, 4)
    assert_false(copied.__sklearn_is_fitted__())
    var params = model.get_params()
    params.max_iter = 1
    var before = model.predict(X)
    _ = model.set_params(params)
    var after = model.predict(X)
    assert_equal(before.unsafe_get(0), after.unsafe_get(0))
    var constant = full[DType.float64](Shape(8), 12.0)
    _ = model.fit(X, constant)
    assert_equal(model.n_iter_, 1)
    assert_equal(model.predict(X).unsafe_get(0), 12.0)


def test_strided_float32_and_weighted_fit() raises:
    var X = full[DType.float32](Shape(5, 2), 0.0)
    var target_storage = NDArray[DType.float64](Shape(10))
    var weight_storage = NDArray[DType.float64](Shape(10))
    for i in range(5):
        X.unsafe_set(i * 2 + 1, Float32(i))
        target_storage.unsafe_set(i * 2 + 1, Float64(i))
        weight_storage.unsafe_set(i * 2 + 1, Float64(i + 1))
    var features = X.view_with_layout(Shape(5, 1), Strides(2, 1), 1)
    var targets = target_storage.view_with_layout(Shape(5), Strides(2), 1)
    var weights = weight_storage.view_with_layout(Shape(5), Strides(2), 1)
    var model = HistGradientBoostingRegressor[DType.float32](
        max_iter=4, min_samples_leaf=1, random_state=5, early_stopping=True,
        validation_fraction=None, n_iter_no_change=2, tol=100,
    )
    _ = model.fit(features, targets, sample_weight=weights)
    assert_equal(model.n_iter_, 2)
    assert_true(model.do_early_stopping_)
    assert_equal(model.train_score_.size, 3)
    assert_equal(model.validation_score_.size, 0)
    var stages = model.staged_predict(features)
    var predictions = model.predict(features)
    assert_equal(len(stages), 2)
    for i in range(5):
        assert_equal(stages[1].unsafe_get(i), predictions.unsafe_get(i))
    assert_true(model.score(features, targets, weights^) > 0.0)


def test_invalid_inputs_preserve_fitted_state() raises:
    var X = full[DType.float64](Shape(4, 1), 0.0)
    var y = full[DType.float64](Shape(4), 5.0)
    var model = HistGradientBoostingRegressor(max_iter=1, random_state=3)
    _ = model.fit(X, y)
    var wrong_count = full[DType.float64](Shape(3), 0.0)
    var wrong_features = full[DType.float64](Shape(4, 2), 0.0)
    var invalid = full[DType.float64](Shape(4), Float64("nan"))
    with assert_raises(contains="ValueError"):
        _ = model.fit(X, wrong_count)
    with assert_raises(contains="ValueError"):
        _ = model.fit(X, invalid)
    with assert_raises(contains="ValueError"):
        _ = model.fit(X, y, X_val=X.view())
    with assert_raises(contains="ValueError"):
        _ = model.predict(wrong_features)
    var params = model.get_params()
    params.max_iter = 0
    _ = model.set_params(params)
    with assert_raises(contains="ValueError"):
        _ = model.fit(X, y)
    assert_equal(model.n_iter_, 1)
    assert_equal(model.predict(X).unsafe_get(0), 5.0)


def test_pipeline_and_unsupported_configuration() raises:
    var X = NDArray[DType.float64](Shape(8, 1))
    var y = NDArray[DType.float64](Shape(8))
    for i in range(8):
        X.unsafe_set(i, Float64(i))
        y.unsafe_set(i, Float64(i * i))

    var scaler = StandardScaler()
    _ = scaler.fit(X)
    var standalone = HistGradientBoostingRegressor(
        max_iter=3, min_samples_leaf=1, random_state=2, warm_start=True,
    )
    _ = standalone.fit(scaler.transform(X), y)
    var pipeline = make_pipeline(StandardScaler(), clone(standalone))
    _ = pipeline.fit(X, y)
    var expected = standalone.predict(scaler.transform(X))
    var prediction = pipeline.predict(X)
    for i in range(y.size):
        assert_equal(prediction.unsafe_get(i), expected.unsafe_get(i))

    _ = standalone.fit(scaler.transform(X), y)
    assert_equal(standalone.n_iter_, 3)
    var params = standalone.get_params()
    params.max_iter = 2
    _ = standalone.set_params(params)
    with assert_raises(contains="max_iter"):
        _ = standalone.fit(X, y)
    params.max_iter = 5
    _ = standalone.set_params(params)
    with assert_raises(contains="fitted feature count"):
        _ = standalone.fit(full[DType.float64](Shape(8, 2), 0.0), y)
    assert_equal(standalone.n_iter_, 3)
    assert_equal(standalone.predict(scaler.transform(X)).unsafe_get(0), expected.unsafe_get(0))
    _ = standalone.fit(scaler.transform(X), y)
    assert_equal(standalone.n_iter_, 5)
    params.warm_start = False
    params.max_iter = 1
    _ = standalone.set_params(params).fit(X, y)
    assert_equal(standalone.n_iter_, 1)
    var scorer = HistGradientBoostingRegressor(early_stopping=True, scoring=String("r2"))
    with assert_raises(contains="scoring=loss"):
        _ = scorer.fit(X, y)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
