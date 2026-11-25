"""Native target bridge lifecycle over production LinearSVC."""

import numojo as nm
from moxie.base import DecisionFunction, ProbabilityPredictor, clone
from moxie.multiclass import BinaryAdapter
from moxie.svm import LinearSVC
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def check_labels[Label: Copyable & Comparable & Deinitable]() raises:
    comptime Adapter = BinaryAdapter[LinearSVC[Label]]
    comptime assert conforms_to(Adapter, DecisionFunction)
    comptime assert not conforms_to(Adapter, ProbabilityPredictor)

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


def main() raises:
    check_labels[Float64]()
    check_labels[Float32]()
    check_labels[Int64]()
