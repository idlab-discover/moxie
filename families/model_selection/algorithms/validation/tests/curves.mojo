from evaluation import Adapter, Scorer, Dense
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import validation_curve, KFold, IndexSplit, ErrorPolicy
from numojo import Shape, Item, zeros
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.


def emit(name: String, train: Dense, test: Dense) raises:
    assert_equal(train.ndim, 2)
    assert_equal(test.ndim, 2)
    assert_equal(train.shape[0], test.shape[0])
    assert_equal(train.shape[1], test.shape[1])
    for parameter in range(train.shape[0]):
        for fold in range(train.shape[1]):
            print("curve", name, parameter, fold, train[Item(parameter, fold)], test[Item(parameter, fold)])


def main() raises:
    var X = zeros[DType.float64](Shape(8, 2))
    var y = Dense(Shape(8))
    var weights = Dense(Shape(8))
    for i in range(8):
        y[Item(i)] = Float64(i * i)
        weights[Item(i)] = Float64(i + 1)
    var splits = List[IndexSplit]()
    var cursor = KFold(4).split(8)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var model = DummyRegressor()
    _ = model.fit(X, y)
    var original = model.predict(X)
    var values: List[String] = ["median", "mean", "median", "invalid"]
    var result = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, param_name="strategy", param_range=values, error_score=ErrorPolicy(False, -123))
    assert_equal(result[0].shape[0], 4)
    assert_equal(result[0].shape[1], 4)
    for fold in range(4):
        assert_equal(result[0][Item(0, fold)], result[0][Item(2, fold)])
        assert_equal(result[1][Item(0, fold)], result[1][Item(2, fold)])
        assert_equal(result[0][Item(3, fold)], -123.0)
        assert_equal(result[1][Item(3, fold)], -123.0)
    emit("mixed", result[0], result[1])
    var weighted = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, weights.copy(), None, param_name="strategy", param_range=values, error_score=ErrorPolicy(False, -123))
    emit("weighted_fit", weighted[0], weighted[1])
    var weighted_score = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, weights.copy(), weights.copy(), param_name="strategy", param_range=values, error_score=ErrorPolicy(False, -123))
    emit("weighted_both", weighted_score[0], weighted_score[1])
    var invalid: List[String] = ["invalid", "also_invalid"]
    var failed = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, param_name="strategy", param_range=invalid, error_score=ErrorPolicy(False, -321))
    emit("all_failed", failed[0], failed[1])
    var nan_failed = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, param_name="strategy", param_range=invalid)
    for fold in range(4):
        assert_true(isnan(nan_failed[0][Item(0, fold)]))
        assert_true(isnan(nan_failed[1][Item(0, fold)]))
    emit("nan_failed", nan_failed[0], nan_failed[1])
    var valid: List[String] = ["mean"]
    var scoring_failed = validation_curve[Adapter](model, X, y, Adapter(), Scorer(True), splits, None, None, param_name="strategy", param_range=valid, error_score=ErrorPolicy(False, -456))
    emit("scoring_failed", scoring_failed[0], scoring_failed[1])
    with assert_raises():
        _ = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, param_name="strategy", param_range=values, error_score=ErrorPolicy(True, 0))
    with assert_raises():
        _ = validation_curve[Adapter](model, X, y, Adapter(), Scorer(True), splits, None, None, param_name="strategy", param_range=valid, error_score=ErrorPolicy(True, 0))
    with assert_raises():
        _ = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, param_name="unknown", param_range=valid)
    with assert_raises():
        var empty = List[String]()
        _ = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, param_name="strategy", param_range=empty)
    with assert_raises():
        var empty = List[IndexSplit]()
        _ = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), empty, None, None, param_name="strategy", param_range=valid)
    with assert_raises():
        var bad_weights = Dense(Shape(1))
        _ = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, bad_weights.copy(), None, param_name="strategy", param_range=valid, error_score=ErrorPolicy(True, 0))
    var repeated = validation_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, param_name="strategy", param_range=values, error_score=ErrorPolicy(False, -123))
    for i in range(4):
        for fold in range(4):
            assert_equal(repeated[0][Item(i, fold)], result[0][Item(i, fold)])
            assert_equal(repeated[1][Item(i, fold)], result[1][Item(i, fold)])
    assert_equal(model.strategy, "mean")
    assert_true(model.__sklearn_is_fitted__())
    var after = model.predict(X)
    for i in range(8):
        assert_equal(original[Item(i)], after[Item(i)])
    var unfitted = DummyRegressor()
    _ = validation_curve[Adapter](unfitted, X, y, Adapter(), Scorer(False), splits, None, None, param_name="strategy", param_range=valid)
    assert_false(unfitted.__sklearn_is_fitted__())
    print("PASS validation curve real estimators, ordering, weights, failures and repeat calls")
