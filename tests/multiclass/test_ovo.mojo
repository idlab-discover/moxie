"""Native contract tests for OneVsOneClassifier; values cross-checked by ovo_parity.py."""
from numojo import NDArray, Shape, Item
import numojo as nm
from std.math import inf, isnan, nan
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)
from moxie.base import ClassifierMixin, DecisionFunction, clone
from moxie.multiclass import OneVsOneClassifier, OneVsOneParams
from ovr_fixtures import (
    MarginClassifier,
    MarginProbaClassifier,
    ProbaClassifier,
    ShiftParams,
    _ScoreRegressor,
)


def matrix(rows: List[List[Float64]]) raises -> NDArray[DType.float64]:
    var result = nm.zeros[DType.float64](Shape(len(rows), len(rows[0])))
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            result[Item(i, j)] = rows[i][j]
    return result^


def labels(values: List[Int]) raises -> NDArray[DType.int64]:
    var result = nm.zeros[DType.int64](Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = Int64(values[i])
    return result^


def X4() raises -> NDArray[DType.float64]:
    return matrix([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [0.5, 2.5], [2.5, 0.0]])


def X8() raises -> NDArray[DType.float64]:
    return matrix([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0],
                   [0.5, 2.5], [2.5, 0.0], [1.5, 1.5], [3.0, 3.0]])


def assert_labels(got: NDArray[DType.int64], expected: List[Int]) raises:
    assert_equal(got.ndim, 1)
    assert_equal(got.size, len(expected))
    for i in range(len(expected)):
        assert_equal(Int(got[Item(i)]), expected[i])


struct PickyClassifier(ClassifierMixin, DecisionFunction, Copyable):
    """MarginClassifier that refuses training sets smaller than `min_samples`."""

    comptime Params = ShiftParams
    comptime Input = NDArray[DType.float64]
    comptime Target = NDArray[DType.int64]
    comptime Prediction = NDArray[DType.int64]
    comptime SampleWeight = NDArray[DType.float64]
    comptime Decision = NDArray[DType.float64]
    var min_samples: Float64
    var _inner: MarginClassifier

    def __init__(out self, *, min_samples: Float64):
        self.min_samples = min_samples
        self._inner = MarginClassifier()

    def get_params(self, deep: Bool = True) raises -> ShiftParams:
        return ShiftParams(self.min_samples)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: ShiftParams) raises -> ref[self] Self:
        self.min_samples = params.shift
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(min_samples=self.min_samples)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._inner.__sklearn_is_fitted__()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        if Float64(X.shape[0]) < self.min_samples:
            raise Error("ValueError: picky component refuses small training sets")
        self._inner.fit(X, y)
        return self

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        return self._inner.decision_function(X)

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        return self._inner.predict(X)

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return self._inner.score(X, y, sample_weight)


def test_multiclass_shapes_and_order() raises:
    var model = OneVsOneClassifier(MarginProbaClassifier())
    assert_false(model.__sklearn_is_fitted__())
    model.fit(X4(), labels([7, 3, 5, 5, 7, 3]))
    assert_true(model.__sklearn_is_fitted__())
    assert_labels(model.classes_, [3, 5, 7])
    assert_equal(model.n_classes_(), 3)
    assert_equal(len(model.estimators_), 3)
    assert_equal(model.n_features_in_.value(), 2)
    assert_false(Bool(model.pairwise_indices_))
    var decision = model.decision_function(X4())
    assert_equal(decision.shape[0], 6)
    assert_equal(decision.shape[1], 3)
    var predicted = model.predict(X4())
    for i in range(6):
        var best = 0
        var votes: Float64 = 0.0
        for k in range(3):
            # Each confidence term lies in (-1/3, 1/3), so rounding recovers the votes.
            var value = decision[Item(i, k)]
            var count = round(value)
            assert_true(abs(value - count) < 1.0 / 3.0)
            votes += count
            if value > decision[Item(i, best)]:
                best = k
        assert_equal(predicted[Item(i)], model.classes_[Item(best)])
        assert_equal(votes, 3.0)


def test_components_see_pair_subsets() raises:
    var X = X8()
    var y = labels([0, 1, 2, 3, 0, 1, 2, 3])
    var model = OneVsOneClassifier(MarginClassifier(shift=0.25))
    model.fit(X, y)
    assert_equal(len(model.estimators_), 6)
    var k = 0
    for i in range(4):
        for j in range(i + 1, 4):
            # Rows of classes i and j in sample order; i is the negative class.
            var rows = List[List[Float64]]()
            var targets = List[Int]()
            for s in range(8):
                var c = Int(y[Item(s)])
                if c == i or c == j:
                    rows.append([X[Item(s, 0)], X[Item(s, 1)]])
                    targets.append(1 if c == j else 0)
            var expected = MarginClassifier(shift=0.25)
            expected.fit(matrix(rows), labels(targets))
            assert_equal(model.estimators_[k].fits, 1)
            assert_equal(model.estimators_[k].shift, 0.25)
            var got = model.estimators_[k].decision_function(X)
            var want = expected.decision_function(X)
            for s in range(8):
                assert_equal(got[Item(s)], want[Item(s)])
            k += 1


def test_decision_is_votes_plus_scaled_confidence() raises:
    var X = X8()
    var model = OneVsOneClassifier(MarginClassifier())
    model.fit(X, labels([0, 1, 2, 0, 1, 2, 0, 1]))
    var decision = model.decision_function(X)
    var pairs: List[List[Int]] = [[0, 1], [0, 2], [1, 2]]
    for s in range(8):
        var votes: List[Float64] = [0.0, 0.0, 0.0]
        var sums: List[Float64] = [0.0, 0.0, 0.0]
        for k in range(3):
            var margin = model.estimators_[k].decision_function(X)[Item(s)]
            var i = pairs[k][0]
            var j = pairs[k][1]
            sums[i] -= margin
            sums[j] += margin
            if margin > 0.0:
                votes[j] += 1.0
            else:
                votes[i] += 1.0
        for c in range(3):
            assert_equal(decision[Item(s, c)], votes[c] + sums[c] / (3.0 * (abs(sums[c]) + 1.0)))


def test_binary_returns_positive_column() raises:
    var model = OneVsOneClassifier(MarginClassifier())
    model.fit(X4(), labels([4, 9, 4, 9, 4, 9]))
    assert_equal(len(model.estimators_), 1)
    assert_equal(model.n_classes_(), 2)
    var decision = model.decision_function(X4())
    assert_equal(decision.ndim, 1)
    var margin = model.estimators_[0].decision_function(X4())
    var predicted = model.predict(X4())
    for i in range(6):
        var vote = 1.0 if margin[Item(i)] > 0.0 else 0.0
        assert_equal(decision[Item(i)], vote + margin[Item(i)] / (3.0 * (abs(margin[Item(i)]) + 1.0)))
        assert_equal(Int(predicted[Item(i)]), 9 if decision[Item(i)] > 0.0 else 4)
    # sklearn #13617: two-class OvO agrees with its single component.
    var direct = model.estimators_[0].predict(X4())
    for i in range(6):
        assert_equal(Int(predicted[Item(i)]), 9 if direct[Item(i)] == 1 else 4)


def test_probability_component() raises:
    # Confidences fall back to P(positive); the binary threshold becomes 0.5.
    var model = OneVsOneClassifier(ProbaClassifier())
    model.fit(X4(), labels([0, 1, 1, 0, 0, 1]))
    var decision = model.decision_function(X4())
    var proba = model.estimators_[0].predict_proba(X4())
    var predicted = model.predict(X4())
    for i in range(6):
        var p = proba[Item(i, 1)]
        var vote = 1.0 if p > 0.5 else 0.0
        assert_equal(decision[Item(i)], vote + p / (3.0 * (p + 1.0)))
        assert_equal(Int(predicted[Item(i)]), 1 if decision[Item(i)] > 0.5 else 0)
    var direct = model.estimators_[0].predict(X4())
    for i in range(6):
        assert_equal(predicted[Item(i)], direct[Item(i)])
    var multi = OneVsOneClassifier(ProbaClassifier())
    multi.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    assert_equal(multi.decision_function(X4()).shape[1], 3)


def test_ties_choose_first_class() raises:
    # A point equidistant from every class mean: every margin is 0, so each
    # pair votes for its first class and confidences cancel.
    var X = matrix([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    var model = OneVsOneClassifier(MarginClassifier())
    model.fit(X, labels([0, 1, 2, 3]))
    var centre = matrix([[1.0, 1.0]])
    var decision = model.decision_function(centre)
    assert_equal(decision[Item(0, 0)], 3.0)
    assert_labels(model.predict(centre), [0])
    # Equal votes across classes: the confidence term breaks the tie.
    var cycle = OneVsOneClassifier(MarginClassifier())
    cycle.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    var d = cycle.decision_function(X4())
    var p = cycle.predict(X4())
    for i in range(6):
        var best = 0
        for k in range(1, 3):
            if d[Item(i, k)] > d[Item(i, best)]:
                best = k
        assert_equal(Int(p[Item(i)]), best)


def test_nan_features_follow_argmax() raises:
    var model = OneVsOneClassifier(MarginClassifier())
    model.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    var decision = model.decision_function(matrix([[nan[DType.float64](), 0.0]]))
    for k in range(3):
        assert_true(isnan(decision[Item(0, k)]))
    # np.argmax returns the first NaN.
    assert_labels(model.predict(matrix([[nan[DType.float64](), 0.0]])), [0])


def test_float_labels_and_column_target() raises:
    var model = OneVsOneClassifier[MarginClassifier, label=DType.float64](MarginClassifier())
    var y = nm.zeros[DType.float64](Shape(6, 1))
    var values: List[Float64] = [2.0, -1.0, 0.0, 0.0, 2.0, -1.0]
    for i in range(6):
        y[Item(i, 0)] = values[i]
    model.fit(X4(), y)
    assert_equal(model.classes_.size, 3)
    assert_equal(model.classes_[Item(0)], -1.0)
    assert_equal(model.classes_[Item(2)], 2.0)
    var predicted = model.predict(X4())
    assert_equal(predicted.ndim, 1)
    for i in range(6):
        var hit = False
        for k in range(3):
            hit = hit or predicted[Item(i)] == model.classes_[Item(k)]
        assert_true(hit)


def test_invalid_targets_and_inputs() raises:
    var model = OneVsOneClassifier[MarginClassifier, label=DType.float64](MarginClassifier())
    var continuous = nm.zeros[DType.float64](Shape(6))
    for i in range(6):
        continuous[Item(i)] = Float64(i) + 0.5
    with assert_raises(contains="Unknown label type: continuous"):
        model.fit(X4(), continuous)
    var missing = nm.zeros[DType.float64](Shape(6))
    missing[Item(2)] = nan[DType.float64]()
    with assert_raises(contains="Input y contains NaN."):
        model.fit(X4(), missing)
    var infinite = nm.zeros[DType.float64](Shape(6))
    infinite[Item(2)] = inf[DType.float64]()
    with assert_raises(contains="Input y contains infinity or a value too large for dtype('float64')."):
        model.fit(X4(), infinite)
    assert_false(model.__sklearn_is_fitted__())

    var ints = OneVsOneClassifier(MarginClassifier())
    with assert_raises(contains="can not be fit when only one class is present"):
        ints.fit(X4(), labels([5, 5, 5, 5, 5, 5]))
    var two_columns = nm.zeros[DType.int64](Shape(6, 2))
    with assert_raises(contains="y should be a 1d array, got an array of shape (6, 2) instead."):
        ints.fit(X4(), two_columns)
    with assert_raises(contains="inconsistent numbers of samples: [6, 3]"):
        ints.fit(X4(), labels([0, 1, 2]))
    with assert_raises(contains="Expected 2D array, got 1D array instead"):
        ints.fit(nm.zeros[DType.float64](Shape(6)), labels([0, 1, 2, 0, 1, 2]))
    with assert_raises(contains="Found array with dim 3, while dim <= 2 is required by OneVsOneClassifier."):
        ints.fit(nm.zeros[DType.float64](Shape(6, 2, 1)), labels([0, 1, 2, 0, 1, 2]))
    with assert_raises(contains="Found array with 0 sample(s) (shape=(0, 2)) while a minimum of 1"):
        ints.fit(nm.zeros[DType.float64](Shape(0, 2)), nm.zeros[DType.int64](Shape(0)))
    with assert_raises(contains="Found array with 0 feature(s) (shape=(6, 0)) while a minimum of 1"):
        ints.fit(nm.zeros[DType.float64](Shape(6, 0)), labels([0, 1, 2, 0, 1, 2]))
    var serial = OneVsOneClassifier(MarginClassifier(), n_jobs=0)
    with assert_raises(contains="n_jobs == 0 in Parallel has no meaning"):
        serial.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    assert_false(serial.__sklearn_is_fitted__())
    assert_false(ints.__sklearn_is_fitted__())

    with assert_raises(contains="NotFittedError: This OneVsOneClassifier instance is not fitted yet."):
        _ = ints.predict(X4())
    with assert_raises(contains="NotFittedError"):
        _ = ints.decision_function(X4())
    with assert_raises(contains="NotFittedError"):
        _ = ints.n_classes_()
    ints.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    with assert_raises(contains="X has 3 features, but OneVsOneClassifier is expecting 2 features as input."):
        _ = ints.predict(matrix([[1.0, 2.0, 3.0]]))
    with assert_raises(contains="Expected 2D array, got 1D array instead"):
        _ = ints.decision_function(nm.zeros[DType.float64](Shape(2)))
    with assert_raises(contains="Found array with 0 sample(s)"):
        _ = ints.predict(nm.zeros[DType.float64](Shape(0, 2)))


def test_failed_fit_preserves_state() raises:
    var model = OneVsOneClassifier(PickyClassifier(min_samples=4.0))
    model.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    # Pair (0, 1) fits on five rows; pair (0, 2) sees three and fails mid-fit.
    with assert_raises(contains="picky component"):
        model.fit(X8(), labels([0, 1, 2, 3, 0, 1, 1, 3]))
    assert_equal(model.classes_.size, 3)
    assert_equal(len(model.estimators_), 3)
    # Validation errors commit nothing either.
    with assert_raises(contains="only one class"):
        model.fit(X4(), labels([1, 1, 1, 1, 1, 1]))
    assert_equal(model.n_classes_(), 3)


def test_refit_replaces_state() raises:
    var model = OneVsOneClassifier(MarginClassifier())
    model.fit(X8(), labels([0, 1, 2, 3, 0, 1, 2, 3]))
    assert_equal(len(model.estimators_), 6)
    model.fit(X4(), labels([8, 9, 8, 9, 8, 9]))
    assert_labels(model.classes_, [8, 9])
    assert_equal(len(model.estimators_), 1)
    assert_equal(model.estimators_[0].fits, 1)


def test_clone_and_params() raises:
    var model = OneVsOneClassifier(MarginClassifier(shift=1.5), n_jobs=4)
    model.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(len(copy.estimators_), 0)
    assert_equal(copy.estimator.shift, 1.5)
    assert_equal(copy.n_jobs.value(), 4)
    var params = model.get_params()
    assert_equal(params.estimator.shift, 1.5)
    assert_equal(params.n_jobs.value(), 4)
    copy.set_params(OneVsOneParams[ShiftParams](ShiftParams(-2.0), None))
    assert_equal(copy.estimator.shift, -2.0)
    assert_false(Bool(copy.n_jobs))
    copy.set_params(n_jobs=Optional[Int](2))
    assert_equal(copy.n_jobs.value(), 2)
    copy.set_params(estimator=MarginClassifier(shift=9.0))
    assert_equal(copy.estimator.shift, 9.0)
    assert_equal(model.estimator.shift, 1.5)
    assert_true(model.__sklearn_is_fitted__())


def test_weighted_score() raises:
    var model = OneVsOneClassifier(MarginClassifier())
    var y = labels([0, 1, 2, 0, 1, 2])
    model.fit(X4(), y)
    var predicted = model.predict(X4())
    var weights = nm.zeros[DType.float64](Shape(6))
    var correct: Float64 = 0.0
    var total: Float64 = 0.0
    for i in range(6):
        weights[Item(i)] = Float64(i + 1)
        total += Float64(i + 1)
        if predicted[Item(i)] == y[Item(i)]:
            correct += Float64(i + 1)
    assert_almost_equal(model.score(X4(), y, weights.copy()), correct / total, atol=1e-15)
    with assert_raises(contains="sample_weight"):
        _ = model.score(X4(), y, nm.zeros[DType.float64](Shape(2)))
    with assert_raises(contains="Weights sum to zero"):
        _ = model.score(X4(), y, nm.zeros[DType.float64](Shape(6)))


def test_regressor_exact_votes_and_precedence() raises:
    # With means 0 and 2 the fixture predicts 4*x - 4.
    var X = matrix([[0.0], [2.0]])
    var y = labels([-3, 7])
    var Xt = matrix([[0.75], [1.0], [1.0625], [1.125], [1.1875], [1.25], [1.3125], [1.5]])
    var scores: List[Float64] = [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0]
    comptime for variant in range(3):
        comptime E = _ScoreRegressor[dtype=DType.float32 if variant == 2 else DType.float64, decision=variant == 1]
        var model = OneVsOneClassifier(E())
        with assert_raises(contains="NotFittedError"):
            _ = model.decision_function(Xt)
        model.fit(X, y)
        var decision = model.decision_function(Xt)
        assert_equal(decision.ndim, 1)
        for i in range(len(scores)):
            var c = scores[i]
            var vote: Float64 = 1.0 if c == 1.0 else 0.0
            assert_equal(decision[Item(i)], vote + c / (3.0 * (abs(c) + 1.0)))
        assert_labels(model.predict(Xt), [-3, -3, -3, -3, -3, 7, -3, -3])
        var copy = clone(model)
        assert_false(copy.__sklearn_is_fitted__())
        assert_false(copy.estimator.__sklearn_is_fitted__())
        model.set_params(estimator=E(shift=1.0))
        model.fit(X, y)
        assert_labels(model.predict(matrix([[1.0]])), [7])
        with assert_raises(contains="expecting 1 features"):
            _ = model.predict(X4())


def test_regressor_multiclass_votes_and_parallel_refit() raises:
    # Identical class centroids give exact zero predictions: pair votes are
    # [2, 1, 0], even for a regressor with a different decision_function.
    var X = matrix([[1.0], [1.0], [1.0]])
    var y = labels([7, -3, 2])
    comptime for variant in range(3):
        comptime E = _ScoreRegressor[dtype=DType.float32 if variant == 2 else DType.float64, decision=variant == 1]
        var model = OneVsOneClassifier(E(), n_jobs=2)
        model.fit(X, y)
        assert_labels(model.classes_, [-3, 2, 7])
        assert_equal(len(model.estimators_), 3)
        var d = model.decision_function(X)
        assert_equal(d.ndim, 2)
        for i in range(3):
            for c in range(3):
                assert_equal(d[Item(i, c)], Float64(2 - c))
        assert_labels(model.predict(X), [-3, -3, -3])
        assert_equal(model.score(X, y), 1.0 / 3.0)
        # Fractional predictions cast no votes; confidence alone breaks ties.
        model.set_params(estimator=E(shift=0.5))
        model.fit(X, y)
        d = model.decision_function(X)
        assert_equal(d[Item(0, 0)], -1.0 / 6.0)
        assert_equal(d[Item(0, 1)], 0.0)
        assert_equal(d[Item(0, 2)], 1.0 / 6.0)
        assert_labels(model.predict(X), [7, 7, 7])
        model.fit(matrix([[0.0], [2.0]]), labels([-3, 7]))
        assert_equal(len(model.estimators_), 1)
        assert_equal(model.decision_function(X).ndim, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
