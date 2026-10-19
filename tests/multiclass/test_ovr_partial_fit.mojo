"""Dense incremental contracts; numerical parity uses the pinned sklearn."""
from std.math import nan, inf
from std.testing import assert_equal, assert_false, assert_true, assert_raises, TestSuite
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie.base import IncrementalClassifier, clone
from moxie.multiclass import OneVsRestClassifier
from ovr_incremental_fixtures import OnlineClassifier
from test_ovr import matrix, labels, assert_labels


def test_batches_and_classes() raises:
    comptime assert conforms_to(OneVsRestClassifier[OnlineClassifier], IncrementalClassifier)
    var m = OneVsRestClassifier(OnlineClassifier(), n_jobs=2)
    var X = matrix([[1.0], [2.0]])
    var classes = labels([7, -3, 2, 7])
    m.partial_fit(X, labels([7, 7]), classes.copy())
    classes[Item(0)] = 99
    assert_labels(m.classes_, [-3, 2, 7])
    assert_equal(len(m.estimators_), 3)
    assert_false(m.multilabel_())
    for k in range(3):
        assert_false(m.estimators_[k].is_constant())
        assert_equal(m.estimators_[k].estimator().positives, 2 if k == 2 else 0)
    m.partial_fit(X, labels([-3, 2]))
    m.partial_fit(X, labels([7, -3]), labels([2, 7, -3, 2]))
    for k in range(3):
        assert_equal(m.estimators_[k].estimator().updates, 3)
        assert_equal(m.estimators_[k].estimator().samples, 6)
    assert_equal(m.estimator.updates, 0)
    assert_equal(m.n_features_in_.value(), 1)
    var fresh = clone(m)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises():
        fresh.partial_fit(X, labels([7, 7]))
    # Replacing the prototype does not discard the fitted components.
    m.set_params(estimator=OnlineClassifier(shift=100.0))
    m.partial_fit(X, labels([2, 7]))
    assert_equal(m.estimators_[0].estimator().shift, 0.0)
    m.fit(X, labels([2, 7]))
    assert_equal(len(m.estimators_), 1)
    assert_equal(m.estimators_[0].estimator().updates, 1)
    assert_equal(m.estimators_[0].estimator().shift, 100.0)
    m.partial_fit(X, labels([7, 7]))
    assert_equal(m.estimators_[0].estimator().updates, 2)


def test_binary_and_single_class() raises:
    var X = matrix([[1.0], [2.0]])
    var m = OneVsRestClassifier(OnlineClassifier())
    m.partial_fit(X, labels([9, 9]), labels([9, 4]))
    assert_equal(len(m.estimators_), 1)
    assert_equal(m.estimators_[0].estimator().positives, 2)
    assert_equal(m.decision_function(X).ndim, 1)
    assert_equal(m.predict_proba(X).shape[1], 2)
    m.partial_fit(X, labels([4, 4]))
    assert_equal(m.estimators_[0].estimator().positives, 2)
    var one = OneVsRestClassifier(OnlineClassifier())
    one.partial_fit(X, labels([9, 9]), labels([9]))
    one.partial_fit(X, labels([9, 9]))
    assert_equal(one.estimators_[0].estimator().positives, 0)
    assert_equal(one.estimators_[0].estimator().updates, 2)
    assert_labels(one.predict(X), [9, 9])
    one.fit(X, labels([9, 9]))
    with assert_raises():
        one.partial_fit(X, labels([9, 9]))


def test_invalid_updates() raises:
    var X = matrix([[1.0], [2.0]])
    var y = labels([2, 7])
    var m = OneVsRestClassifier(OnlineClassifier())
    with assert_raises():
        m.partial_fit(X, y)
    with assert_raises():
        m.partial_fit(X, y, labels([]))
    with assert_raises():
        m.partial_fit(X, y, labels([2]))
    assert_false(m.__sklearn_is_fitted__())
    m.partial_fit(X, y, y.copy())
    with assert_raises():
        m.partial_fit(X, y, labels([2, 7, 9]))
    with assert_raises():
        m.partial_fit(X, labels([2, 9]))
    with assert_raises():
        m.partial_fit(X, labels([2]))
    with assert_raises():
        m.partial_fit(matrix([[1.0, 2.0], [3.0, 4.0]]), y)
    with assert_raises():
        m.partial_fit(X, labels([]))
    var indicator = nm.zeros[DType.int64](Shape(2, 2))
    indicator[Item(0, 0)] = 1
    indicator[Item(1, 1)] = 1
    with assert_raises():
        m.partial_fit(X, indicator)
    assert_equal(m.estimators_[0].estimator().updates, 1)
    m.fit(X, indicator)
    with assert_raises():
        m.partial_fit(X, y)
    var invalid = OneVsRestClassifier(OnlineClassifier(), n_jobs=0)
    with assert_raises():
        invalid.partial_fit(X, y, y.copy())
    invalid.set_params(n_jobs=Optional[Int](1), verbose=-1)
    with assert_raises():
        invalid.partial_fit(X, y, y.copy())



def test_invalid_float_targets_and_class_shapes() raises:
    var X = matrix([[1.0], [2.0]])
    var m = OneVsRestClassifier[OnlineClassifier, label=DType.float64](OnlineClassifier())
    var y = matrix([[0.0], [1.0]])
    var classes = labels([0, 1]).astype[DType.float64]()
    with assert_raises():
        m.partial_fit(X, y, y.copy())
    with assert_raises():
        m.partial_fit(X, matrix([[0.5], [1.0]]), classes.copy())
    with assert_raises():
        m.partial_fit(X, matrix([[nan[DType.float64]()], [1.0]]), classes.copy())
    with assert_raises():
        m.partial_fit(X, y, matrix([[0.0], [inf[DType.float64]()]]).reshape(Shape(2)))
    assert_false(m.__sklearn_is_fitted__())
    m.partial_fit(X, y, classes.copy())
    assert_equal(m.estimators_[0].estimator().positives, 1)

def test_multilabel_updates_and_transitions() raises:
    var X = matrix([[1.0], [2.0]])
    var y = matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]).astype[DType.int64]()
    var m = OneVsRestClassifier(OnlineClassifier(), n_jobs=2)
    with assert_raises():
        m.partial_fit(X, y)
    with assert_raises():
        m.partial_fit(X, y, labels([0, 1, 2]))
    assert_false(m.__sklearn_is_fitted__())
    m.fit(X, y)
    var empty_labels = nm.zeros[DType.int64](Shape(2, 3))
    m.partial_fit(X, empty_labels, labels([2, 0, 1, 2]))
    m.partial_fit(X, y)
    assert_true(m.multilabel_())
    assert_labels(m.classes_, [0, 1, 2])
    for k in range(3):
        assert_equal(m.estimators_[k].estimator().updates, 3)
        assert_equal(m.estimators_[k].estimator().positives, 2)
    with assert_raises():
        m.partial_fit(X, y, labels([0, 1, 3]))
    with assert_raises():
        m.partial_fit(X, nm.zeros[DType.int64](Shape(2, 2)))
    with assert_raises():
        m.partial_fit(X, nm.zeros[DType.int64](Shape(2, 4)))
    with assert_raises():
        m.partial_fit(X, nm.zeros[DType.int64](Shape(0, 3)))
    with assert_raises():
        m.partial_fit(X, matrix([[0.0, 2.0, 0.0], [1.0, 0.0, 1.0]]).astype[DType.int64]())
    with assert_raises():
        m.partial_fit(matrix([[1.0, 2.0], [3.0, 4.0]]), y)
    assert_equal(m.estimators_[0].estimator().updates, 3)
    m.partial_fit(X, labels([0, 2]))
    assert_true(m.multilabel_())
    assert_equal(len(m.estimators_), 3)
    var fresh = clone(m)
    assert_false(fresh.__sklearn_is_fitted__())
    m.fit(X, labels([4, 7]))
    assert_false(m.multilabel_())
    m.fit(X, empty_labels)
    with assert_raises():
        m.partial_fit(X, y)
    # A later constant also rejects atomically, after an earlier real component.
    y[Item(1, 2)] = 0
    m.fit(X, y)
    with assert_raises():
        m.partial_fit(X, empty_labels)
    assert_equal(m.estimators_[0].estimator().updates, 1)


def test_two_label_vector_transition() raises:
    var X = matrix([[1.0], [2.0]])
    var y = matrix([[0.0, 1.0], [1.0, 0.0]]).astype[DType.int64]()
    var m = OneVsRestClassifier(OnlineClassifier())
    m.fit(X, y)
    m.partial_fit(X, labels([0, 1]))
    assert_true(m.multilabel_())
    assert_equal(len(m.estimators_), 1)
    with assert_raises():
        _ = m.predict(X)
    m.partial_fit(X, y)
    assert_equal(len(m.estimators_), 1)
    m.fit(X, y)
    assert_equal(len(m.estimators_), 2)
    # The vector transition discards a second constant without updating it.
    y[Item(0, 1)] = 0
    m.fit(X, y)
    assert_true(m.estimators_[1].is_constant())
    m.partial_fit(X, labels([0, 1]))
    assert_equal(len(m.estimators_), 1)
    assert_equal(m.estimators_[0].estimator().updates, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
