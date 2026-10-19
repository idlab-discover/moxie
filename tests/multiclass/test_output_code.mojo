"""Output-code contracts; independent numerical oracle in output_code_parity.py."""
from numojo import NDArray, Shape, Item
import numojo as nm
from std.math import nan, inf
from std.testing import assert_equal, assert_true, assert_false, assert_almost_equal, assert_raises, TestSuite
from moxie.base import clone
from moxie.multiclass import OutputCodeClassifier, OutputCodeParams
from moxie._random import seed_global_random_state, check_random_state
from ovr_fixtures import MarginClassifier, MarginProbaClassifier, ProbaClassifier, ShiftParams, DecisionRegressor, _ScoreRegressor
from test_ovr import X4, labels, matrix, assert_labels
from test_parallel import FailingClassifier


def assert_book(a: NDArray[DType.float64], b: NDArray[DType.float64]) raises:
    assert_equal(a.shape[0], b.shape[0])
    assert_equal(a.shape[1], b.shape[1])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            assert_equal(a[Item(i, j)], b[Item(i, j)])


def test_seeded_book_and_constant_columns() raises:
    var m = OutputCodeClassifier(MarginClassifier(), random_state=0)
    assert_equal(m.code_size, 1.5)
    m.fit(X4(), labels([7, 3, 5, 5, 7, 3]))
    assert_labels(m.classes_, [3, 5, 7])
    assert_equal(len(m.estimators_), 4)
    assert_equal(m.n_features_in_.value(), 2)
    # np.random.RandomState(0).uniform(size=(3,4)) > .5, mapped to -1/+1.
    assert_book(m.code_book_, matrix([[1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, 1, 1]]))
    assert_true(m.estimators_[3].is_constant())
    assert_equal(m.estimators_[3].constant(), 1)
    assert_equal(m.estimators_[0].estimator().fits, 1)
    assert_false(m.estimator.__sklearn_is_fitted__())
    # Negative-target means prove that code bits are passed as signed labels.
    assert_equal(m.estimators_[0].estimator()._model.mean0[0], 2.5)


def test_probability_book_and_capabilities() raises:
    var m = OutputCodeClassifier(ProbaClassifier(), random_state=0)
    m.fit(X4(), labels([7, 3, 5, 5, 7, 3]))
    assert_book(m.code_book_, matrix([[1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 1, 1]]))
    assert_equal(m.predict(X4()).size, 6)


def test_single_class_and_constant_negative() raises:
    var m = OutputCodeClassifier(MarginClassifier(), random_state=1, code_size=3)
    m.fit(X4(), labels([9, 9, 9, 9, 9, 9]))
    assert_book(m.code_book_, matrix([[-1, 1, -1]]))
    assert_labels(m.predict(X4()), [9, 9, 9, 9, 9, 9])
    assert_equal(m.score(X4(), labels([9, 9, 9, 9, 9, 9])), 1.0)
    for component in m.estimators_:
        assert_true(component.is_constant())
    with assert_raises(contains="features"):
        _ = m.predict(matrix([[1, 2, 3]]))


def test_duplicate_codewords_tie() raises:
    # Seed 0, width 1 produces the same all-positive code for three classes.
    var m = OutputCodeClassifier(MarginClassifier(), random_state=0, code_size=0.34)
    m.fit(X4(), labels([7, 3, 5, 5, 7, 3]))
    assert_labels(m.predict(X4()), [3, 3, 3, 3, 3, 3])
    assert_equal(len(m.estimators_), 1)


def test_distinct_codewords_equidistant() raises:
    var m = OutputCodeClassifier(MarginClassifier(), random_state=17, code_size=2)
    var X = nm.zeros[DType.float64](Shape(3, 2))
    m.fit(X, labels([7, -2, 3]))
    assert_labels(m.predict(X), [-2, -2, -2])
    var distinct = False
    for k in range(6):
        distinct = distinct or m.code_book_[Item(0, k)] != m.code_book_[Item(1, k)]
    assert_true(distinct)


def test_clone_params_and_repeat_fit() raises:
    var m = OutputCodeClassifier(MarginClassifier(), random_state=17, code_size=2)
    var y = labels([0, 0, 1, 1, 2, 2])
    m.fit(X4(), y)
    var book = m.code_book_.copy()
    m.fit(X4(), y)
    assert_book(book, m.code_book_)
    var fresh = clone(m)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.random_state.value(), 17)
    assert_equal(fresh.code_size, 2.0)
    assert_equal(len(fresh.estimators_), 0)
    fresh.set_params(OutputCodeParams(ShiftParams(0.5), 1.0, Optional[Int](42), Optional[Int](2)))
    var params = fresh.get_params(False)
    assert_equal(params.estimator.shift, 0.5)
    assert_equal(params.n_jobs.value(), 2)
    fresh.set_params(code_size=3.0, random_state=Optional[Optional[Int]](Optional[Int](1)))
    fresh.set_params(estimator=MarginClassifier(shift=0.25))
    fresh.fit(X4(), labels([0, 1, 0, 1, 0, 1]))
    assert_equal(len(fresh.estimators_), 6)
    assert_equal(fresh.estimator.shift, 0.25)
    fresh.set_params(random_state=Optional[Optional[Int]](Optional[Int](None)),
                     n_jobs=Optional[Optional[Int]](Optional[Int](None)))
    assert_false(Bool(fresh.random_state))
    assert_false(Bool(fresh.n_jobs))
    m.fit(X4(), labels([4, 4, 4, 4, 4, 4]))
    assert_equal(len(m.estimators_), 2)
    assert_labels(m.classes_, [4])


def test_global_rng_advances_and_is_released() raises:
    seed_global_random_state(42)
    var m = OutputCodeClassifier(MarginClassifier())
    var y = labels([0, 0, 1, 1, 2, 2])
    m.fit(X4(), y)
    var first = m.code_book_.copy()
    m.fit(X4(), y)
    var different = False
    for i in range(3):
        for j in range(4):
            different = different or first[Item(i, j)] != m.code_book_[Item(i, j)]
    assert_true(different)
    seed_global_random_state(42)
    m.fit(X4(), y)
    assert_book(first, m.code_book_)
    # The fit must leave no outstanding lease.
    with check_random_state() as rng:
        assert_true(rng[].uniform() >= 0.0)


def test_parallel_matches_serial() raises:
    var y = labels([0, 0, 1, 1, 2, 2])
    var serial = OutputCodeClassifier(MarginProbaClassifier(), random_state=12, code_size=3)
    serial.fit(X4(), y)
    var predicted = serial.predict(X4())
    var settings: List[Int] = [2, -1, -2, 1000]
    for jobs in settings:
        var parallel = OutputCodeClassifier(MarginProbaClassifier(), random_state=12, code_size=3, n_jobs=jobs)
        parallel.fit(X4(), y)
        assert_book(serial.code_book_, parallel.code_book_)
        var result = parallel.predict(X4())
        for i in range(result.size):
            assert_equal(result[Item(i)], predicted[Item(i)])


def test_failed_fit_preserves_state() raises:
    var m = OutputCodeClassifier(FailingClassifier(), random_state=0, n_jobs=2)
    var y = labels([0, 0, 1, 1, 2, 2])
    m.fit(X4(), y)
    var old = m.code_book_.copy()
    var expected = m.predict(X4())
    m.estimator.min_rows = 1
    m.estimator.min_positives = 1
    with assert_raises():
        m.fit(X4(), y)
    assert_book(old, m.code_book_)
    var got = m.predict(X4())
    for i in range(got.size):
        assert_equal(got[Item(i)], expected[Item(i)])


def test_unfitted_and_invalid_parameters() raises:
    var m = OutputCodeClassifier(MarginClassifier())
    with assert_raises(contains="NotFittedError"):
        _ = m.predict(X4())
    var bad_sizes: List[Float64] = [0, -1, nan[DType.float64](), inf[DType.float64](), 0.01, 1e308]
    for size in bad_sizes:
        m.code_size = size
        with assert_raises(contains="ValueError"):
            m.fit(X4(), labels([0, 0, 1, 1, 2, 2]))
        assert_false(m.__sklearn_is_fitted__())
    m.code_size = 1.5
    for seed in [-1, 4294967296]:
        m.random_state = seed
        with assert_raises(contains="random_state"):
            m.fit(X4(), labels([0, 0, 1, 1, 2, 2]))
    m.random_state = 0
    m.n_jobs = 0
    with assert_raises(contains="n_jobs == 0"):
        m.fit(X4(), labels([0, 0, 1, 1, 2, 2]))


def test_targets_and_nonfinite_scores() raises:
    var m = OutputCodeClassifier[MarginClassifier, label=DType.float64](MarginClassifier(), random_state=0)
    var y = matrix([[7], [3], [5], [5], [7], [3]])
    m.fit(X4(), y)
    assert_equal(m.classes_[Item(0)], 3.0)
    with assert_raises(contains="inconsistent"):
        m.fit(X4(), matrix([[0], [1]]))
    with assert_raises(contains="1d"):
        m.fit(X4(), matrix([[0, 1], [1, 0]]))
    y[Item(0, 0)] = 0.5
    with assert_raises(contains="continuous"):
        m.fit(X4(), y)
    y[Item(0, 0)] = nan[DType.float64]()
    with assert_raises(contains="NaN"):
        m.fit(X4(), y)
    with assert_raises(contains="NaN"):
        _ = m.predict(matrix([[nan[DType.float64](), 0]]))
    with assert_raises(contains="nonempty"):
        _ = m.predict(nm.zeros[DType.float64](Shape(0, 2)))


def test_empty_and_mismatched_inputs() raises:
    var m = OutputCodeClassifier(MarginClassifier(), random_state=0)
    with assert_raises(contains="no class"):
        m.fit(X4(), nm.zeros[DType.int64](Shape(0)))
    with assert_raises(contains="nonempty"):
        m.fit(nm.zeros[DType.float64](Shape(6, 0)), labels([0, 0, 1, 1, 2, 2]))
    m.fit(X4(), labels([0, 0, 1, 1, 2, 2]))
    with assert_raises(contains="features"):
        _ = m.predict(matrix([[1, 2, 3]]))
    with assert_raises(contains="sample_weight"):
        _ = m.score(X4(), labels([0, 0, 1, 1, 2, 2]), nm.zeros[DType.float64](Shape(2)))


def test_weighted_score() raises:
    var m = OutputCodeClassifier(ProbaClassifier(), random_state=0, code_size=0.34)
    var y = labels([7, 3, 5, 5, 7, 3])
    m.fit(X4(), y)
    var weights = nm.zeros[DType.float64](Shape(6))
    for i in range(6):
        weights[Item(i)] = Float64(i + 1)
    assert_almost_equal(m.score(X4(), y, weights.copy()), 8.0 / 21.0, atol=1e-15)
    with assert_raises(contains="Weights sum"):
        _ = m.score(X4(), y, nm.zeros[DType.float64](Shape(6)))


def test_regressor_predict_precedence_and_parallel_lifecycle() raises:
    var y = labels([7, 3, 5, 5, 7, 3])
    var serial = OutputCodeClassifier(DecisionRegressor(), random_state=0)
    with assert_raises(contains="NotFittedError"):
        _ = serial.predict(X4())
    serial.fit(X4(), y)
    assert_book(serial.code_book_, matrix([[1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, 1, 1]]))
    assert_true(serial.estimators_[3].is_constant())
    # Same predict output as the margin classifier, but decision is shifted by 10.
    var classifier = OutputCodeClassifier(MarginClassifier(), random_state=0)
    classifier.fit(X4(), y)
    var expected = classifier.predict(X4())
    var predicted = serial.predict(X4())
    for i in range(predicted.size):
        assert_equal(predicted[Item(i)], expected[Item(i)])
    for jobs in [2, -1, 1000]:
        var parallel = OutputCodeClassifier(DecisionRegressor(), random_state=0, n_jobs=jobs)
        parallel.fit(X4(), y)
        assert_book(serial.code_book_, parallel.code_book_)
        var result = parallel.predict(X4())
        for i in range(result.size):
            assert_equal(result[Item(i)], predicted[Item(i)])
        parallel.fit(X4(), labels([9, 9, 9, 9, 9, 9]))
        assert_labels(parallel.predict(X4()), [9, 9, 9, 9, 9, 9])
    var fresh = clone(serial)
    assert_false(fresh.__sklearn_is_fitted__())
    fresh.set_params(estimator=DecisionRegressor(shift=0.25))
    fresh.fit(X4(), y)
    assert_equal(fresh.get_params().estimator.shift, 0.25)
    assert_equal(serial.score(X4(), y), classifier.score(X4(), y))


def test_regressor_float32_constants_and_ties() raises:
    var m = OutputCodeClassifier(_ScoreRegressor[dtype=DType.float32, decision=True](), random_state=17, code_size=2)
    var X = nm.zeros[DType.float64](Shape(3, 2))
    m.fit(X, labels([7, -2, 3]))
    assert_labels(m.predict(X), [-2, -2, -2])
    m.random_state = 1
    m.code_size = 3
    m.fit(X, labels([9, 9, 9]))
    assert_book(m.code_book_, matrix([[-1, 1, -1]]))
    assert_equal(m.estimators_[0].constant(), -1)
    assert_labels(m.predict(X), [9, 9, 9])
    m.random_state = 0
    m.code_size = 0.34
    m.fit(X, labels([7, -2, 3]))
    assert_labels(m.predict(X), [-2, -2, -2])


def test_regressor_column_prediction_rejected() raises:
    var m = OutputCodeClassifier(_ScoreRegressor[decision=True, column=True](), random_state=0)
    m.fit(X4(), labels([0, 0, 1, 1, 2, 2]))
    with assert_raises(contains="regressor predict must have shape"):
        _ = m.predict(X4())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
