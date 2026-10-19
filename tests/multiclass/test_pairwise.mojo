"""Dense pairwise contracts; the recording component is a test double."""
from numojo import NDArray, Shape, Item, NDArrayStrides
import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises, TestSuite
from moxie.base import clone
from moxie.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier, PairwiseInput, pairwise_input, StringLabels, StringTarget
from pairwise_fixtures import PairwiseRecorder, PairwiseParams
from test_ovo import labels, assert_labels
from ovr_fixtures import MarginClassifier


def grid[dtype: DType = DType.float64](n: Int, p: Int) raises -> NDArray[dtype]:
    var result = nm.zeros[dtype](Shape(n, p))
    for i in range(n):
        for j in range(p):
            result[Item(i, j)] = Scalar[dtype](i * p + j + 1)
    return result^


def equal_array[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.ndim, b.ndim)
    assert_equal(a.size, b.size)
    for d in range(a.ndim):
        assert_equal(a.shape[d], b.shape[d])
    if a.ndim == 1:
        for i in range(a.size):
            assert_equal(a[Item(i)], b[Item(i)])
    else:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                assert_equal(a[Item(i, j)], b[Item(i, j)])


def test_subsets_and_parallel() raises:
    comptime for dtype in [DType.float32, DType.float64]:
        comptime E = PairwiseRecorder[dtype]
        var X = grid[dtype](6, 6)
        var y = labels([7, -3, 7, 2, -3, 7])
        var expected = List[List[Int]]()
        expected.append([1, 3, 4])
        expected.append([0, 1, 2, 4, 5])
        expected.append([0, 2, 3, 5])
        var serial = OneVsOneClassifier[E, dtype=dtype](E())
        serial.fit(X, y)
        for jobs in [1, 2, -1]:
            var m = OneVsOneClassifier[E, dtype=dtype](E(), n_jobs=jobs)
            m.fit(X, y)
            assert_equal(m.n_features_in_.value(), 6)
            assert_labels(m.classes_, [-3, 2, 7])
            assert_equal(len(m.pairwise_indices_.value()), 3)
            for k in range(3):
                ref indices = m.pairwise_indices_.value()[k]
                assert_labels(indices, expected[k])
                ref component = m.estimators_[k]
                assert_equal(component.seen.shape[0], len(expected[k]))
                assert_equal(component.seen.shape[1], len(expected[k]))
                assert_equal(component.updates, 1)
                for i in range(indices.size):
                    for j in range(indices.size):
                        assert_equal(component.seen[Item(i, j)], X[Item(expected[k][i], expected[k][j])])
            assert_labels(m.estimators_[0].targets, [0, 1, 0])
            assert_labels(m.estimators_[1].targets, [1, 0, 1, 0, 1])
            assert_labels(m.estimators_[2].targets, [1, 1, 0, 1])
            var Xt = grid[dtype](3, 6)
            equal_array(m.predict(Xt), serial.predict(Xt))
            equal_array(m.decision_function(Xt), serial.decision_function(Xt))
            # Gathered training arrays own their data.
            var previous = m.estimators_[0].seen.copy()
            X[Item(1, 1)] = 99
            equal_array(m.estimators_[0].seen, previous)
            X = grid[dtype](6, 6)


def test_strided_inputs() raises:
    comptime for dtype in [DType.float32, DType.float64]:
        comptime E = PairwiseRecorder[dtype]
        var storage = grid[dtype](12, 13)
        var X = storage.view_with_layout(Shape(6, 6), NDArrayStrides(-26, 2), 131)
        var m = OneVsOneClassifier[E, dtype=dtype](E(), n_jobs=2)
        m.fit(X, labels([7, -3, 7, 2, -3, 7]))
        ref c = m.estimators_[0]
        var rows: List[Int] = [1, 3, 4]
        for i in range(3):
            for j in range(3):
                assert_equal(c.seen[Item(i, j)], X[Item(rows[i], rows[j])])
        var Xt = storage.view_with_layout(Shape(3, 6), NDArrayStrides(26, -2), 11)
        equal_array(m.decision_function(Xt), m.decision_function(Xt.contiguous()))


def test_shapes_clone_refits_and_failure() raises:
    comptime E = PairwiseRecorder
    var m = OneVsOneClassifier(E(), n_jobs=2)
    var X = grid(6, 6)
    var y = labels([7, -3, 7, 2, -3, 7])
    with assert_raises(contains="NotFittedError"):
        _ = m.predict(X)
    with assert_raises(contains="X should be a square kernel matrix"):
        m.fit(grid(6, 5), y)
    assert_false(m.__sklearn_is_fitted__())
    m.fit(X, y)
    with assert_raises(contains="expecting 6 features"):
        _ = m.predict(grid(3, 5))
    with assert_raises(contains="Expected 2D"):
        m.fit(nm.zeros[DType.float64](Shape(6)), y)
    with assert_raises(contains="0 sample(s)"):
        _ = m.predict(nm.zeros[DType.float64](Shape(0, 6)))
    var old = m.decision_function(X)
    m.set_params(estimator=E(fail_rows=5))
    with assert_raises(contains="intentional pair failure"):
        m.fit(X, y)
    equal_array(m.decision_function(X), old)
    assert_labels(m.pairwise_indices_.value()[0], [1, 3, 4])
    m.set_params(estimator=E())
    var cloned = clone(m)
    assert_true(cloned.pairwise_input())
    assert_false(cloned.__sklearn_is_fitted__())
    assert_false(Bool(cloned.pairwise_indices_))
    assert_false(Bool(cloned.n_features_in_))
    assert_equal(len(cloned.estimators_), 0)
    assert_equal(cloned.n_jobs.value(), 2)
    # Tag/configuration changes must not affect inference on already fitted pairs.
    m.estimator.set_params(PairwiseParams(False, -1))
    assert_false(m.pairwise_input())
    equal_array(m.decision_function(X), old)
    m.fit(grid(6, 2), y)
    assert_false(Bool(m.pairwise_indices_))
    assert_equal(m.estimators_[0].seen.shape[1], 2)
    m.set_params(estimator=E())
    m.fit(grid(2, 2), labels([9, -1]))
    assert_labels(m.classes_, [-1, 9])
    assert_labels(m.pairwise_indices_.value()[0], [0, 1])
    assert_equal(len(m.estimators_), 1)
    assert_equal(m.decision_function(grid(1, 2)).ndim, 1)
    assert_equal(m.predict(grid(1, 2)).size, 1)


def test_string_probability_and_tags() raises:
    comptime E = PairwiseRecorder[probability=True]
    var m = OneVsOneClassifier[E, kind=StringLabels](E(), n_jobs=2)
    m.fit(grid(3, 3), StringTarget(["猫", "", "é"]))
    assert_equal(m.classes_[0], "")
    assert_equal(m.classes_[1], "é")
    assert_equal(m.classes_[2], "猫")
    assert_labels(m.pairwise_indices_.value()[0], [1, 2])
    assert_equal(m.predict(grid(1, 3)).rows(), 1)
    assert_true(pairwise_input(m))
    assert_false(pairwise_input(MarginClassifier()))
    assert_false(pairwise_input(OneVsOneClassifier(MarginClassifier())))
    var ovr = OneVsRestClassifier(E())
    assert_true(pairwise_input(ovr))
    ovr.fit(grid(3, 3), labels([0, 1, 2]))
    for i in range(len(ovr.estimators_)):
        equal_array(ovr.estimators_[i].estimator().seen, grid(3, 3))
    var ecoc = OutputCodeClassifier(E(), random_state=17)
    assert_false(pairwise_input(ecoc))
    ecoc.fit(grid(3, 3), labels([0, 1, 2]))
    _ = ecoc.predict(grid(2, 3))


def test_partial_fit_pairwise_audit() raises:
    var m = OneVsOneClassifier(PairwiseRecorder(), n_jobs=2)
    var X = grid(6, 6)
    var y = labels([7, -3, 7, 2, -3, 7])
    m.partial_fit(X, y, labels([-3, 2, 7]))
    assert_false(Bool(m.pairwise_indices_))
    assert_equal(m.estimators_[0].seen.shape[0], 3)
    assert_equal(m.estimators_[0].seen.shape[1], 6)
    m.partial_fit(grid(1, 6), labels([7]))
    assert_false(Bool(m.pairwise_indices_))
    _ = m.predict(grid(2, 6))
    m.fit(X, y)
    assert_true(Bool(m.pairwise_indices_))
    with assert_raises(contains="changed feature count"):
        m.partial_fit(X, y)
    assert_true(Bool(m.pairwise_indices_))
    # Two classes keep all columns, so fit -> partial_fit succeeds and clears indices.
    m.fit(X, labels([7, -3, 7, -3, -3, 7]))
    m.partial_fit(grid(1, 6), labels([7]))
    assert_false(Bool(m.pairwise_indices_))
    _ = m.predict(grid(2, 6))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
