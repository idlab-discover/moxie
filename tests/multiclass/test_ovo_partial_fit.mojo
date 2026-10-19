"""Native dense incremental OvO contracts using the shared online test double."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises, TestSuite
from std.math import nan, inf
from numojo import Shape, Item
from moxie.base import IncrementalClassifier, clone
from moxie.multiclass import OneVsOneClassifier
from ovr_incremental_fixtures import OnlineClassifier
from ovr_fixtures import MarginClassifier
from test_ovr import matrix, labels


def test_pair_lifecycle() raises:
    comptime assert conforms_to(OneVsOneClassifier[OnlineClassifier], IncrementalClassifier)
    comptime assert not conforms_to(OneVsOneClassifier[MarginClassifier], IncrementalClassifier)
    var X = matrix([[1.0], [2.0]])
    var m = OneVsOneClassifier(OnlineClassifier(), n_jobs=2)
    m.partial_fit(X, labels([7, 7]), labels([7, -3, 2, 7]))
    assert_equal(m.classes_[Item(0)], -3)
    assert_equal(len(m.estimators_), 3)
    assert_true(m.__sklearn_is_fitted__())
    assert_equal(m.n_features_in_.value(), 1)
    assert_false(Bool(m.pairwise_indices_))
    assert_equal(m.estimators_[0].updates, 0)
    assert_equal(m.estimators_[1].positives, 2)
    assert_equal(m.estimators_[2].positives, 2)
    with assert_raises():
        _ = m.predict(X)
    with assert_raises():
        _ = m.decision_function(X)
    m.partial_fit(X, labels([-3, -3]))
    assert_equal(m.estimators_[0].updates, 1)
    assert_equal(m.estimators_[0].positives, 0)
    assert_equal(m.estimators_[1].updates, 2)
    assert_equal(m.estimators_[2].updates, 1)
    _ = m.predict(X)
    _ = m.decision_function(X)
    var fresh = clone(m)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises():
        fresh.partial_fit(X, labels([2, 7]))
    m.fit(X, labels([2, 7]))
    assert_equal(len(m.estimators_), 1)
    assert_equal(m.estimators_[0].updates, 1)
    m.partial_fit(X, labels([7, 7]))
    assert_equal(m.estimators_[0].updates, 2)
    assert_equal(m.estimators_[0].samples, 4)
    assert_equal(m.decision_function(X).ndim, 1)
    m.set_params(estimator=OnlineClassifier(shift=99.0))
    m.partial_fit(X, labels([2, 7]))
    assert_equal(m.estimators_[0].shift, 0.0)
    m.fit(X, labels([2, 7]))
    assert_equal(m.estimators_[0].shift, 99.0)


def test_validation_and_failed_update_state() raises:
    var X = matrix([[1.0], [2.0]])
    var y = labels([2, 7])
    var m = OneVsOneClassifier(OnlineClassifier())
    with assert_raises():
        m.partial_fit(X, y)
    with assert_raises():
        m.partial_fit(X, y, labels([]))
    with assert_raises():
        m.partial_fit(X, labels([2, 2]), labels([2, 2]))
    with assert_raises():
        m.partial_fit(X, y, labels([2, 3]))
    assert_false(m.__sklearn_is_fitted__())
    m.partial_fit(X, y, y.copy())
    with assert_raises():
        m.partial_fit(X, y, labels([2, 7, 9]))
    with assert_raises():
        m.partial_fit(X, labels([2, 9]))
    with assert_raises():
        m.partial_fit(X, labels([2]))
    with assert_raises():
        m.partial_fit(X, labels([]))
    with assert_raises():
        m.partial_fit(matrix([[1.0, 2.0], [3.0, 4.0]]), y)
    with assert_raises():
        m.partial_fit(X.reshape(Shape(2)), y)
    with assert_raises():
        m.partial_fit(X, matrix([[0.0, 1.0], [1.0, 0.0]]).astype[DType.int64]())
    m.set_params(n_jobs=Optional[Optional[Int]](Optional[Int](0)))
    with assert_raises():
        m.partial_fit(X, y)
    assert_equal(m.estimators_[0].updates, 1)
    assert_equal(m.estimators_[0].samples, 2)
    assert_equal(m.n_features_in_.value(), 1)


def test_float_labels() raises:
    var X = matrix([[1.0], [2.0]])
    var y = matrix([[2.0], [7.0]])
    var m = OneVsOneClassifier[OnlineClassifier, label=DType.float64](OnlineClassifier())
    with assert_raises():
        m.partial_fit(X, y, matrix([[0.5], [1.0]]).reshape(Shape(2)))
    with assert_raises():
        m.partial_fit(X, y, matrix([[2.0], [inf[DType.float64]()]]).reshape(Shape(2)))
    with assert_raises():
        m.partial_fit(X, y, y.copy())
    m.partial_fit(X, y, y.reshape(Shape(2)))
    with assert_raises():
        m.partial_fit(X, matrix([[nan[DType.float64]()], [7.0]]))
    with assert_raises():
        m.partial_fit(X, matrix([[2.5], [7.0]]))
    assert_equal(m.estimators_[0].updates, 1)


def test_component_failure_preserves_state() raises:
    # Deliberately corrupt one test component's width to inject a worker error.
    for jobs in [1, 2]:
        var X = matrix([[1.0], [2.0], [3.0]])
        var y = labels([0, 1, 2])
        var m = OneVsOneClassifier(OnlineClassifier(), n_jobs=jobs)
        m.partial_fit(X, y, y.copy())
        m.estimators_[1].weights.append(0.0)
        with assert_raises():
            m.partial_fit(X, y)
        for k in range(3):
            assert_equal(m.estimators_[k].updates, 1)
            assert_equal(m.estimators_[k].samples, 2)
        assert_equal(m.estimators_[0].weights[0], 1.0)


def test_nonfinite_features_reach_component() raises:
    var m = OneVsOneClassifier(OnlineClassifier())
    m.partial_fit(matrix([[nan[DType.float64]()], [inf[DType.float64]()]]),
                  labels([0, 1]), labels([0, 1]))
    assert_equal(m.estimators_[0].updates, 1)
    assert_equal(m.estimators_[0].samples, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
