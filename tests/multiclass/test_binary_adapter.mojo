"""Native target bridge lifecycle over production classifiers."""

import numojo as nm
from moxie.base import (
    DecisionFunction,
    IncrementalClassifier,
    LogProbabilityPredictor,
    ProbabilityPredictor,
    clone,
)
from moxie.discriminant_analysis import LinearDiscriminantAnalysis
from moxie.multiclass import BinaryAdapter
from moxie.naive_bayes import GaussianNB
from moxie.svm import LinearSVC
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def check_labels[Label: Copyable & Comparable & Deinitable]() raises:
    comptime Adapter = BinaryAdapter[LinearSVC[Label]]
    comptime assert conforms_to(Adapter, DecisionFunction)
    comptime assert not conforms_to(Adapter, ProbabilityPredictor)
    comptime assert not conforms_to(Adapter, IncrementalClassifier)

    var X = nm.zeros[DType.float64](nm.Shape(4, 1))
    for i in range(4):
        X[nm.Item(i, 0)] = Float64(i) - 1.5

    var storage = nm.full[DType.int64](nm.Shape(9), 99)
    storage[nm.Item(8)] = 0
    storage[nm.Item(6)] = 0
    storage[nm.Item(4)] = 1
    storage[nm.Item(2)] = 1
    var y = storage.view_with_layout(nm.Shape(4), nm.NDArrayStrides(-2), 8)
    var model = BinaryAdapter(LinearSVC[Label](dual=False, random_state=42))
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(X)

    var prediction = model.fit(X, y).predict(X)
    assert_equal(prediction.ndim, 1)
    assert_equal(prediction.size, y.size)
    for i in range(y.size):
        assert_equal(prediction[nm.Item(i)], y[nm.Item(i)])
    var weights = nm.ones[DType.float64](nm.Shape(4))
    weights[nm.Item(0)] = 3.0
    y[nm.Item(0)] = 1
    assert_equal(model.score(X, y, weights^), 0.5)
    y[nm.Item(0)] = 0

    var fitted_copy = model.copy()
    var coefficient = model.estimator.coef_[nm.Item(0, 0)]
    fitted_copy.estimator.coef_[nm.Item(0, 0)] = 777.0
    assert_equal(model.estimator.coef_[nm.Item(0, 0)], coefficient)
    fitted_copy.estimator.coef_[nm.Item(0, 0)] = coefficient

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    assert_equal(params.random_state, 42)
    params.C = 0.25
    _ = fresh.set_params(params).fit(X, y)
    assert_equal(fresh.estimator.C, 0.25)
    assert_equal(model.estimator.C, 1.0)
    assert_true(fresh.decision_function(X)[nm.Item(0)] != model.decision_function(X)[nm.Item(0)])

    # Refit with signed codes. The fitted copy still owns its original model.
    y[nm.Item(0)] = -1
    y[nm.Item(1)] = -1
    assert_equal(model.fit(X, y).predict(X)[nm.Item(0)], -1)
    assert_equal(fitted_copy.predict(X)[nm.Item(0)], 0)

    var matrix_target = nm.zeros[DType.int64](nm.Shape(4, 1))
    with assert_raises(contains="rank-one"):
        _ = model.fit(X, matrix_target)
    var short_target = nm.zeros[DType.int64](nm.Shape(3))
    with assert_raises(contains="target length"):
        _ = model.fit(X, short_target)
    y[nm.Item(0)] = 0
    with assert_raises(contains="binary codes"):
        _ = model.fit(X, y)
    y[nm.Item(0)] = 2
    with assert_raises(contains="binary codes"):
        _ = model.fit(X, y)
    assert_equal(model.predict(X)[nm.Item(0)], -1)


def check_lda() raises:
    comptime LDA = LinearDiscriminantAnalysis[DType.float32]
    comptime Adapter = BinaryAdapter[LDA]
    comptime assert conforms_to(Adapter, DecisionFunction)
    comptime assert conforms_to(Adapter, LogProbabilityPredictor)

    var X = nm.zeros[DType.float32](nm.Shape(4, 1))
    for i in range(4):
        X[nm.Item(i, 0)] = Float32(i) - 1.5
    var y = nm.zeros[DType.int64](nm.Shape(4))
    y[nm.Item(2)] = 1
    y[nm.Item(3)] = 1
    var model = BinaryAdapter(LDA())
    with assert_raises(contains="NotFittedError"):
        _ = model.predict_proba(X)

    var predictions = model.fit(X, y).predict(X)
    assert_equal(predictions.ndim, 1)
    assert_equal(predictions.size, y.size)
    for i in range(y.size):
        assert_equal(predictions[nm.Item(i)], y[nm.Item(i)])
    var decision: nm.NDArray[DType.float32] = model.decision_function(X)
    assert_equal(decision.ndim, 1)
    var proba: nm.NDArray[DType.float32] = model.predict_proba(X)
    var log_proba = model.predict_log_proba(X)
    var native_log_proba = model.estimator.predict_log_proba(X)
    assert_equal(proba.ndim, 2)
    assert_equal(proba.shape[0], 4)
    assert_equal(proba.shape[1], 2)
    assert_true(proba[nm.Item(0, 0)] > proba[nm.Item(0, 1)])
    assert_true(proba[nm.Item(3, 1)] > proba[nm.Item(3, 0)])
    for i in range(y.size):
        for j in range(2):
            assert_equal(log_proba[nm.Item(i, j)], native_log_proba[nm.Item(i, j)])

    var fitted_copy = model.copy()
    var coefficient = model.estimator.coef_[nm.Item(0, 0)]
    fitted_copy.estimator.coef_[nm.Item(0, 0)] = 777.0
    assert_equal(model.estimator.coef_[nm.Item(0, 0)], coefficient)
    fitted_copy.estimator.coef_[nm.Item(0, 0)] = coefficient

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    params.store_covariance = True
    _ = fresh.set_params(params).fit(X, y)
    assert_true(Bool(fresh.estimator.covariance_))
    assert_false(Bool(model.estimator.covariance_))

    y[nm.Item(0)] = -1
    y[nm.Item(1)] = -1
    assert_equal(model.fit(X, y).predict(X)[nm.Item(0)], -1)
    assert_equal(fitted_copy.predict(X)[nm.Item(0)], 0)
    var weights = nm.ones[DType.float32](nm.Shape(4))
    weights[nm.Item(0)] = 3.0
    y[nm.Item(0)] = 1
    assert_equal(model.score(X, y, weights^), 0.5)
    y[nm.Item(0)] = 0
    with assert_raises(contains="binary codes"):
        _ = model.fit(X, y)
    assert_equal(model.predict(X)[nm.Item(0)], -1)


def check_gaussian_nb() raises:
    comptime Adapter = BinaryAdapter[GaussianNB[]]
    comptime assert conforms_to(Adapter, IncrementalClassifier)
    comptime assert conforms_to(Adapter, LogProbabilityPredictor)
    comptime assert not conforms_to(Adapter, DecisionFunction)

    var X = nm.zeros[DType.float64](nm.Shape(4, 1))
    for i in range(4):
        X[nm.Item(i, 0)] = Float64(i) - 1.5
    var y = nm.full[DType.int64](nm.Shape(4), -1)
    y[nm.Item(2)] = 1
    y[nm.Item(3)] = 1
    var storage = nm.full[DType.int64](nm.Shape(3), 99)
    storage[nm.Item(0)] = 1
    storage[nm.Item(2)] = -1
    var classes = storage.view_with_layout(nm.Shape(2), nm.NDArrayStrides(-2), 2)

    var model = BinaryAdapter(GaussianNB())
    with assert_raises(contains="NotFittedError"):
        _ = model.predict_proba(X)
    with assert_raises(contains="classes"):
        _ = model.partial_fit(X, y)
    var invalid_classes = nm.zeros[DType.int64](nm.Shape(2, 1))
    with assert_raises(contains="rank-one"):
        _ = model.partial_fit(X, y, invalid_classes^)
    assert_false(model.__sklearn_is_fitted__())

    var positive_rows: List[Int] = [2, 3]
    var negative_rows: List[Int] = [0, 1]
    _ = model.partial_fit(X[positive_rows], y[positive_rows], classes^)
    assert_equal(model.estimator.classes_[0], -1)
    assert_equal(model.estimator.classes_[1], 1)
    var fitted_copy = model.copy()
    _ = model.partial_fit(X[negative_rows], y[negative_rows])
    assert_equal(fitted_copy.estimator.class_count_.item(0), 0.0)
    assert_equal(model.estimator.class_count_.item(0), 2.0)
    assert_false(clone(model).__sklearn_is_fitted__())
    assert_equal(model.score(X, y), 1.0)

    var proba = model.predict_proba(X)
    var native_proba = model.estimator.predict_proba(X)
    assert_equal(proba.shape, nm.Shape(4, 2))
    for i in range(proba.size):
        assert_equal(proba.item(i), native_proba.item(i))

    y[nm.Item(0)] = 0
    y[nm.Item(1)] = 0
    with assert_raises(contains="previously unseen label"):
        _ = model.partial_fit(X, y)
    assert_equal(model.estimator.class_count_.item(0), 2.0)
    assert_equal(model.fit(X, y).predict(X).item(0), 0)
    assert_equal(model.estimator.class_count_.item(1), 2.0)


def main() raises:
    check_labels[Float64]()
    check_labels[Float32]()
    check_labels[Int64]()
    check_lda()
    check_gaussian_nb()
