from std.math import nan
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from moxie._labels import StringTarget
from moxie.base import WeightedSupervisedEstimator, clone
from moxie.ensemble import HistGradientBoostingClassifier
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from numojo import NDArray, Shape, Strides, full


def fit_weighted[
    E: WeightedSupervisedEstimator
](mut model: E, X: E.Input, y: E.Target, weights: E.SampleWeight,) raises:
    _ = model.fit(X, y, sample_weight=weights)


def test_binary_lifecycle_and_pipeline() raises:
    var X = NDArray[DType.float64](Shape(12, 1))
    var y = NDArray[DType.float64](Shape(12))
    for i in range(12):
        X.unsafe_set(i, Float64(i))
        y.unsafe_set(i, 8.0 if i >= 6 else -3.0)
    var model = HistGradientBoostingClassifier(
        max_iter=4, min_samples_leaf=2, random_state=10
    )
    with assert_raises(contains="NotFittedError"):
        _ = model.predict_proba(X)
    with assert_raises(contains="NotFittedError"):
        _ = model.staged_decision_function(X)
    _ = model.fit(X, y.reshape(Shape(12, 1)))
    assert_equal(model.n_iter_, 4)
    assert_equal(model.n_trees_per_iteration_, 1)
    assert_equal(model.n_features_in_, 1)
    assert_equal(model.get_classes().unsafe_get(0), -3.0)
    assert_equal(model.get_classes().unsafe_get(1), 8.0)
    assert_equal(model.predict_proba(X).ndim, 2)
    assert_equal(model.predict_proba(X).shape[1], 2)
    assert_equal(model.decision_function(X).ndim, 1)
    assert_equal(model.score(X, y), 1.0)
    var decision = model.decision_function(X)
    var stages = model.staged_decision_function(X)
    assert_true(stages[0].unsafe_get(0) != stages[3].unsafe_get(0))
    _ = model.fit(X, y)
    var repeated = model.decision_function(X)
    for i in range(y.size):
        assert_equal(stages[3].unsafe_get(i), decision.unsafe_get(i))
        assert_equal(repeated.unsafe_get(i), decision.unsafe_get(i))

    var copied = clone(model)
    assert_equal(copied.get_params().max_iter, 4)
    assert_false(copied.__sklearn_is_fitted__())
    var pipeline = make_pipeline(StandardScaler(), copied^)
    _ = pipeline.fit(X, y)
    assert_equal(pipeline.score(X, y), 1.0)

    var params = model.get_params()
    params.max_iter = 1
    _ = model.set_params(params)
    assert_equal(model.n_iter_, 4)
    var constant = full[DType.float64](Shape(12), 5.0)
    _ = model.fit(X, constant)
    assert_equal(model.n_iter_, 1)
    assert_equal(model.predict(X).unsafe_get(0), 5.0)
    assert_equal(model.classes_.size, 1)
    assert_equal(model.predict_proba(X).shape[1], 2)
    assert_false(Bool(model.is_categorical_))


def test_typed_targets_and_strided_weights() raises:
    var storage = full[DType.float32](Shape(9, 2), 0.0)
    var y = NDArray[DType.int64](Shape(9))
    var weights = full[DType.float64](Shape(18), 0.0)
    for i in range(9):
        storage.unsafe_set(2 * i + 1, Float32(i))
        y.unsafe_set(i, 9007199254740993 + Int64(i // 3))
        weights.unsafe_set(2 * i + 1, Float64(i + 1))
    var X = storage.view_with_layout(Shape(9, 1), Strides(2, 1), 1)
    var w = weights.view_with_layout(Shape(9), Strides(2), 1)
    var model = HistGradientBoostingClassifier[DType.float32, DType.int64](
        max_iter=4,
        min_samples_leaf=1,
        random_state=5,
        early_stopping=True,
        validation_fraction=None,
        n_iter_no_change=2,
        tol=100,
        class_weight=String("balanced"),
    )
    fit_weighted(model, X, y, w)
    assert_equal(model.n_iter_, 2)
    assert_true(model.do_early_stopping_)
    assert_equal(model.train_score_.size, 3)
    assert_equal(model.validation_score_.size, 0)
    assert_equal(model.n_trees_per_iteration_, 3)
    assert_equal(model.classes_.unsafe_get(0), Int64(9007199254740993))
    assert_equal(model.decision_function(X).shape[1], 3)
    var stages = model.staged_predict(X)
    assert_equal(len(stages), 2)
    var predictions = model.predict(X)
    for i in range(9):
        assert_equal(stages[1].unsafe_get(i), predictions.unsafe_get(i))
    assert_equal(model.score(X, y, w.view()), 1.0)


def test_string_labels_and_explicit_validation() raises:
    var X = NDArray[DType.float64](Shape(6, 1))
    for i in range(6):
        X.unsafe_set(i, Float64(i))
    var y = StringTarget(["owl", "owl", "owl", "ant", "ant", "ant"])
    var class_weights = Dict[Int64, Float64]()
    class_weights[0] = 2.0
    var model = HistGradientBoostingClassifier[string_labels=True](
        max_iter=4,
        min_samples_leaf=1,
        random_state=2,
        early_stopping=True,
        class_weight=class_weights^,
    )
    _ = model.fit(X, y, X_val=X.view(), y_val=y.copy())
    assert_equal(model.classes_[0], "ant")
    assert_equal(model.classes_[1], "owl")
    assert_equal(model.score(X, y), 1.0)
    assert_equal(model.validation_score_.size, 5)
    var unknown = StringTarget(["cat", "owl", "owl", "ant", "ant", "ant"])
    with assert_raises(contains="ValueError"):
        _ = model.fit(X, y, X_val=X.view(), y_val=unknown^)
    assert_equal(model.n_iter_, 4)
    assert_equal(model.predict(X).get(0), "owl")
    var params = model.get_params()
    params.warm_start = True
    params.max_iter = 6
    _ = model.set_params(params)
    var different = StringTarget(["owl", "owl", "owl", "ape", "ape", "ape"])
    with assert_raises(contains="fitted classes"):
        _ = model.fit(X, different)
    assert_equal(model.n_iter_, 4)
    assert_equal(model.classes_[0], "ant")
    _ = model.fit(X, y, X_val=X.view(), y_val=y.copy())
    assert_equal(model.n_iter_, 6)
    assert_equal(model.validation_score_.size, 7)


def test_invalid_targets_parameters_and_holdout() raises:
    var X = full[DType.float64](Shape(4, 1), 0.0)
    var y = NDArray[DType.float64](Shape(4))
    for i in range(4):
        y.unsafe_set(i, Float64(i))
    var model = HistGradientBoostingClassifier(max_iter=1, random_state=3)
    _ = model.fit(X, y)
    with assert_raises(contains="ValueError"):
        _ = model.fit(X, full[DType.float64](Shape(4), 0.5))
    with assert_raises(contains="ValueError"):
        _ = model.fit(X, y, X_val=X.view())
    with assert_raises(contains="ValueError"):
        _ = model.predict(full[DType.float64](Shape(4, 2), 0.0))
    var params = model.get_params()
    params.monotonic_cst = List[Int]([0])
    _ = model.set_params(params)
    with assert_raises(contains="multiclass"):
        _ = model.fit(X, y)
    params.monotonic_cst = None
    params.early_stopping = True
    _ = model.set_params(params)
    with assert_raises(contains="class"):
        _ = model.fit(X, y)
    params.scoring = String("accuracy")
    _ = model.set_params(params)
    with assert_raises(contains="scoring=loss"):
        _ = model.fit(X, y)
    assert_equal(model.n_iter_, 1)
    assert_equal(model.classes_.size, 4)
    params.early_stopping = False
    var wrong_keys = Dict[Int64, Float64]()
    wrong_keys[9] = 2.0
    params.class_weight = wrong_keys^
    _ = model.set_params(params)
    with assert_raises(contains="class_weight"):
        _ = model.fit(X, y)
    params.class_weight = NoneType()
    _ = model.set_params(params)
    with assert_raises(contains="sum to zero"):
        _ = model.fit(X, y, sample_weight=full[DType.float64](Shape(4), 0.0))
    assert_equal(model.n_iter_, 1)
    assert_equal(model.classes_.size, 4)


def test_categorical_float32_with_strided_input() raises:
    var storage = full[DType.float32](Shape(120, 2), 0)
    var y = NDArray[DType.int64](Shape(120))
    for i in range(120):
        var category = i % 4
        var value: Float32 = -80.5 if category == 1 else Float32(category)
        if category == 3:
            value = nan[DType.float32]()
        storage.unsafe_set(2 * i + 1, value)
        y.unsafe_set(i, Int64(1 if category == 1 or category == 3 else -1))
    var X = storage.view_with_layout(Shape(120, 1), Strides(2, 1), 1)
    var model = HistGradientBoostingClassifier[DType.float32, DType.int64](
        categorical_features=List[Bool]([True]), max_iter=3,
        min_samples_leaf=3, random_state=4,
    )
    _ = model.fit(X, y)
    assert_equal(model.is_categorical_.value(), List[Bool]([True]))
    assert_equal(model.score(X, y), 1.0)
    var query = NDArray[DType.float32](Shape(2, 1))
    query.unsafe_set(0, nan[DType.float32]())
    query.unsafe_set(1, 7000.25)
    var scores = model.decision_function(query)
    assert_equal(scores.unsafe_get(0), scores.unsafe_get(1))
    var stages = model.staged_decision_function(query)
    assert_equal(stages[2].unsafe_get(1), scores.unsafe_get(1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
