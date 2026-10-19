"""Contract tests for `n_jobs`: joblib worker resolution and serial-equivalent parallel fits."""
from numojo import NDArray, Shape, Item
import numojo as nm
from std.sys import num_logical_cores
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)
from moxie.base import ClassifierMixin, DecisionFunction
from moxie.multiclass import OneVsOneClassifier, OneVsRestClassifier
from moxie._multiclass_binary import n_workers
from ovr_fixtures import Binary, Matrix, MarginClassifier

comptime N_CLASSES = 7
comptime N_SAMPLES = 210


def blobs() raises -> Tuple[Matrix, Binary]:
    """Deterministic, overlapping 7-class data with varied margins per sample."""
    var X = nm.zeros[DType.float64](Shape(N_SAMPLES, 3))
    var y = nm.zeros[DType.int64](Shape(N_SAMPLES))
    for i in range(N_SAMPLES):
        var c = (i * 5 + i // 7) % N_CLASSES
        y[Item(i)] = Int64(c)
        X[Item(i, 0)] = Float64(c) + Float64((i * 37) % 11) / 4.0
        X[Item(i, 1)] = Float64((c * 3) % N_CLASSES) - Float64((i * 13) % 7) / 3.0
        X[Item(i, 2)] = Float64((i * 29) % 17) / 5.0
    return (X^, y^)


def assert_same(got: Matrix, expected: Matrix) raises:
    """Bitwise equality: parallel fits run the same arithmetic on the same inputs."""
    assert_equal(got.ndim, expected.ndim)
    assert_equal(got.size, expected.size)
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            # Compare first: assert_equal formats its message on every call.
            if got[Item(i, j)] != expected[Item(i, j)]:
                assert_equal(got[Item(i, j)], expected[Item(i, j)], String("at (", i, ", ", j, ")"))


def job_settings() -> List[Optional[Int]]:
    return [Optional[Int](1), Optional[Int](2), Optional[Int](3), Optional[Int](-1),
            Optional[Int](-2), Optional[Int](1000), Optional[Int](-1000)]


# ----- a component that fails on chosen tasks ---------------------------------


@fieldwise_init
struct FailParams(Copyable, Movable):
    var min_rows: Int
    var min_positives: Int


struct FailingClassifier(ClassifierMixin, DecisionFunction, Copyable):
    """MarginClassifier that raises when a task is at least as large as configured.

    The message names the task's row and positive counts, so a test can tell
    which task's error was reported. A test double, not an algorithm.
    """

    comptime Params = FailParams
    comptime Input = Matrix
    comptime Target = Binary
    comptime Prediction = Binary
    comptime SampleWeight = Matrix
    comptime Decision = Matrix
    var min_rows: Int
    var min_positives: Int
    var _inner: MarginClassifier

    def __init__(out self, *, min_rows: Int = 1 << 30, min_positives: Int = 1 << 30):
        self.min_rows = min_rows
        self.min_positives = min_positives
        self._inner = MarginClassifier()

    def get_params(self, deep: Bool = True) raises -> FailParams:
        return FailParams(self.min_rows, self.min_positives)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: FailParams) raises -> ref[self] Self:
        self.min_rows = params.min_rows
        self.min_positives = params.min_positives
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(min_rows=self.min_rows, min_positives=self.min_positives)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._inner.__sklearn_is_fitted__()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Matrix, ref[y_origin] y: Binary) raises -> ref[self] Self:
        var positives = 0
        for i in range(y.size):
            positives += Int(y[Item(i)])
        if y.size >= self.min_rows or positives >= self.min_positives:
            raise Error(String("ValueError: planted failure rows=", y.size, " positives=", positives))
        self._inner.fit(X, y)
        return self

    def decision_function(self, X: Matrix) raises -> Matrix:
        return self._inner.decision_function(X)

    def predict(self, X: Matrix) raises -> Binary:
        return self._inner.predict(X)

    def score(self, X: Matrix, y: Binary, sample_weight: Optional[Matrix] = None) raises -> Float64:
        return self._inner.score(X, y, sample_weight)


def staircase(n_classes: Int) raises -> Tuple[Matrix, Binary]:
    """Class c holds c + 1 samples, so every task has a distinct size."""
    var n = n_classes * (n_classes + 1) // 2
    var X = nm.zeros[DType.float64](Shape(n, 2))
    var y = nm.zeros[DType.int64](Shape(n))
    var row = 0
    for c in range(n_classes):
        for r in range(c + 1):
            y[Item(row)] = Int64(c)
            X[Item(row, 0)] = Float64(c) + Float64(r) / 10.0
            X[Item(row, 1)] = Float64(r)
            row += 1
    return (X^, y^)


# ----- tests --------------------------------------------------------------------


def test_n_workers_follows_joblib() raises:
    var cores = num_logical_cores()
    assert_equal(n_workers(None, 10), 1)
    assert_equal(n_workers(Optional[Int](1), 10), 1)
    assert_equal(n_workers(Optional[Int](4), 10), 4)
    # More jobs than tasks never starts idle workers.
    assert_equal(n_workers(Optional[Int](1000), 10), 10)
    assert_equal(n_workers(Optional[Int](-1), 1 << 20), cores)
    assert_equal(n_workers(Optional[Int](-2), 1 << 20), max(1, cores - 1))
    assert_equal(n_workers(Optional[Int](-1000000), 10), 1)
    assert_equal(n_workers(Optional[Int](8), 0), 1)
    with assert_raises(contains="n_jobs == 0 in Parallel has no meaning"):
        _ = n_workers(Optional[Int](0), 10)


def test_ovr_parallel_fit_matches_serial() raises:
    var data = blobs()
    var serial = OneVsRestClassifier(MarginClassifier(shift=0.25))
    serial.fit(data[0], data[1])
    var expected = serial.decision_function(data[0])
    var expected_labels = serial.predict(data[0])
    for n_jobs in job_settings():
        var model = OneVsRestClassifier(MarginClassifier(shift=0.25), n_jobs=n_jobs)
        # Repeats catch scheduling-dependent slot or ordering mistakes.
        for _ in range(5):
            model.fit(data[0], data[1])
            assert_equal(len(model.estimators_), N_CLASSES)
            for k in range(N_CLASSES):
                assert_equal(model.estimators_[k].estimator().fits, 1)
            assert_same(model.decision_function(data[0]), expected)
            var predicted = model.predict(data[0])
            for i in range(N_SAMPLES):
                if predicted[Item(i)] != expected_labels[Item(i)]:
                    assert_equal(predicted[Item(i)], expected_labels[Item(i)], String("at ", i))


def test_ovo_parallel_fit_matches_serial() raises:
    var data = blobs()
    var serial = OneVsOneClassifier(MarginClassifier(shift=-0.5))
    serial.fit(data[0], data[1])
    var expected = serial.decision_function(data[0])
    for n_jobs in job_settings():
        var model = OneVsOneClassifier(MarginClassifier(shift=-0.5), n_jobs=n_jobs)
        for _ in range(5):
            model.fit(data[0], data[1])
            assert_equal(len(model.estimators_), N_CLASSES * (N_CLASSES - 1) // 2)
            for k in range(len(model.estimators_)):
                assert_equal(model.estimators_[k].fits, 1)
            assert_same(model.decision_function(data[0]), expected)


def test_ovr_rejects_n_jobs_zero() raises:
    var data = blobs()
    var model = OneVsRestClassifier(MarginClassifier(), n_jobs=0)
    with assert_raises(contains="n_jobs == 0 in Parallel has no meaning"):
        model.fit(data[0], data[1])
    # sklearn binarizes y, then joblib rejects n_jobs=0: before the X/y length check...
    var short = nm.zeros[DType.float64](Shape(3, 3))
    with assert_raises(contains="n_jobs == 0 in Parallel has no meaning"):
        model.fit(short, data[1])
    # ...even for a one-class target...
    with assert_raises(contains="n_jobs == 0 in Parallel has no meaning"):
        model.fit(data[0], nm.zeros[DType.int64](Shape(N_SAMPLES)))
    # ...but after target validation.
    var continuous = nm.zeros[DType.float64](Shape(N_SAMPLES))
    for i in range(N_SAMPLES):
        continuous[Item(i)] = Float64(i) + 0.5
    var floats = OneVsRestClassifier[MarginClassifier, label=DType.float64](MarginClassifier(), n_jobs=0)
    with assert_raises(contains="continuous"):
        floats.fit(data[0], continuous)
    assert_false(model.__sklearn_is_fitted__())


def test_ovr_parallel_error_is_lowest_task() raises:
    var data = staircase(5)
    # Columns 2, 3 and 4 fail; column 2 (3 positives) is reported, as in serial order.
    for n_jobs in [Optional[Int](None), Optional[Int](-1), Optional[Int](3)]:
        var model = OneVsRestClassifier(FailingClassifier(min_positives=3), n_jobs=n_jobs)
        with assert_raises(contains="planted failure rows=15 positives=3"):
            model.fit(data[0], data[1])
        assert_false(model.__sklearn_is_fitted__())


def test_ovo_parallel_error_is_lowest_task() raises:
    var data = staircase(5)
    # Pairs in fit order: (0,1) 3 rows, (0,2) 4, (0,3) 5, ... (1,2) 5 rows.
    # The first pair with >= 5 rows is (0,3): 5 rows, 4 positives.
    for n_jobs in [Optional[Int](None), Optional[Int](-1), Optional[Int](4)]:
        var model = OneVsOneClassifier(FailingClassifier(min_rows=5), n_jobs=n_jobs)
        with assert_raises(contains="planted failure rows=5 positives=4"):
            model.fit(data[0], data[1])
        assert_false(model.__sklearn_is_fitted__())


def test_failed_parallel_fit_preserves_state() raises:
    var small = staircase(3)
    var large = staircase(5)
    var ovr = OneVsRestClassifier(FailingClassifier(min_positives=4), n_jobs=-1)
    ovr.fit(small[0], small[1])
    with assert_raises(contains="planted failure"):
        ovr.fit(large[0], large[1])
    assert_equal(ovr.n_classes_(), 3)
    assert_equal(len(ovr.estimators_), 3)

    var ovo = OneVsOneClassifier(FailingClassifier(min_rows=6), n_jobs=-1)
    ovo.fit(small[0], small[1])
    with assert_raises(contains="planted failure"):
        ovo.fit(large[0], large[1])
    assert_equal(ovo.n_classes_(), 3)
    assert_equal(len(ovo.estimators_), 3)
    assert_true(ovo.__sklearn_is_fitted__())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
