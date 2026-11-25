"""Fitted copies and refits for the native dense decision-tree component."""
import numojo as nm
from moxie.base import clone
from moxie.multiclass import OneVsRestClassifier
from moxie.tree import DecisionTreeClassifier
from std.testing import assert_equal, assert_false, assert_raises


def main() raises:
    comptime Tree = DecisionTreeClassifier[DType.int64]
    var X = nm.zeros[DType.float32](nm.Shape(6, 1))
    var y = nm.zeros[DType.int64](nm.Shape(6))
    for i in range(6):
        X[nm.Item(i, 0)] = Float32(i)
        y[nm.Item(i)] = Int64(10 + i // 2)

    var model = OneVsRestClassifier[Tree, DType.float32](
        Tree(random_state=0), n_jobs=2
    )
    with assert_raises(contains="NotFittedError"):
        _ = model.predict_proba(X)
    assert_equal(model.fit(X, y).predict_proba(X)[nm.Item(0, 0)], 1.0)

    var snapshot = model.estimators_[0].estimator().copy()
    var original_value = snapshot.tree_.value.unsafe_get(0)
    snapshot.classes_[0][nm.Item(0)] = -1
    snapshot.tree_.value.unsafe_set(0, -1)
    assert_equal(model.estimators_[0].estimator().classes_[0][nm.Item(0)], 0)
    assert_equal(model.estimators_[0].estimator().tree_.value.unsafe_get(0), original_value)
    snapshot.classes_[0][nm.Item(0)] = 0
    snapshot.tree_.value.unsafe_set(0, original_value)

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    assert_equal(params.estimator.random_state.value(), 0)
    params.estimator.max_depth = 2
    _ = fresh.set_params(params).fit(X, y)
    assert_equal(fresh.estimators_[1].estimator().get_depth(), 2)
    assert_false(Bool(model.get_params().estimator.max_depth))

    for i in range(6):
        y[nm.Item(i)] = Int64(12 - i // 2)
    _ = model.fit(X, y)
    assert_equal(model.predict(X)[nm.Item(0)], 12)
    assert_equal(snapshot.predict(X)[nm.Item(0)], 1)
    with assert_raises(contains="ValueError"):
        _ = model.fit(X, nm.zeros[DType.int64](nm.Shape(5)))
    assert_equal(model.predict(X)[nm.Item(0)], 12)
