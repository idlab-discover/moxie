"""Learning curves using the actual DummyRegressor."""
from evaluation import Adapter, Scorer, Dense
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie._random import RandomState
from moxie.model_selection import learning_curve, LearningCurveResult, KFold, IndexArray, IndexSplit, ErrorPolicy
from numojo import Shape, Item, Strides, zeros
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_false, assert_raises
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.


def fractions(values: List[Float64]) raises -> Dense:
    var result = Dense(Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = values[i]
    return result^


def counts(values: List[Int]) raises -> IndexArray:
    var result = IndexArray(Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = Int64(values[i])
    return result^


def emit(name: String, result: LearningCurveResult) raises:
    assert_equal(result.train_sizes.ndim, 1)
    assert_equal(result.train_scores.ndim, 2)
    assert_equal(result.train_scores.shape[0], result.train_sizes.size)
    assert_equal(result.test_scores.shape[0], result.train_sizes.size)
    assert_equal(result.test_scores.shape[1], result.train_scores.shape[1])
    assert_equal(Bool(result.fit_times), Bool(result.score_times))
    if result.fit_times:
        assert_equal(result.fit_times.value().shape[0], result.train_sizes.size)
        assert_equal(result.fit_times.value().shape[1], result.train_scores.shape[1])
        assert_equal(result.score_times.value().shape[0], result.train_sizes.size)
        assert_equal(result.score_times.value().shape[1], result.train_scores.shape[1])
    for size in range(result.train_sizes.size):
        for fold in range(result.train_scores.shape[1]):
            print("learning", name, size, fold, result.train_sizes[Item(size)], result.train_scores[Item(size, fold)], result.test_scores[Item(size, fold)])
            if result.fit_times:
                assert_true(result.fit_times.value()[Item(size, fold)] >= 0)
                assert_true(result.score_times.value()[Item(size, fold)] >= 0)


def main() raises:
    var X = zeros[DType.float64](Shape(8, 2))
    var y = Dense(Shape(8))
    var weights = Dense(Shape(8))
    for i in range(8):
        y[Item(i)] = Float64(i * i)
        weights[Item(i)] = Float64(i + 1)
    var splits = List[IndexSplit]()
    var cursor = KFold(4).split(8)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var model = DummyRegressor()
    _ = model.fit(X, y)
    var original = model.predict(X)
    var default_result = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None)
    assert_false(Bool(default_result.fit_times))
    emit("default", default_result)
    var sizes = counts([6, 2, 4, 2])
    var absolute = learning_curve[Adapter, DType.int64](model, X, y, Adapter(), Scorer(False), splits, None, None, train_sizes=sizes.copy(), return_times=True)
    emit("absolute", absolute)
    var fractions_input = fractions([1.0, 0.01, 0.5, 0.1, 0.99])
    var fractional = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, train_sizes=fractions_input.copy())
    emit("fractional", fractional)
    var shuffled = learning_curve[Adapter, DType.int64](model, X, y, Adapter(), Scorer(False), splits, None, None, train_sizes=sizes.copy(), shuffle=True, random_state=42)
    emit("shuffled", shuffled)
    var repeated = learning_curve[Adapter, DType.int64](model, X, y, Adapter(), Scorer(False), splits, None, None, train_sizes=sizes.copy(), shuffle=True, random_state=42)
    for i in range(3):
        for fold in range(4):
            assert_equal(repeated.train_scores[Item(i, fold)], shuffled.train_scores[Item(i, fold)])
            assert_equal(repeated.test_scores[Item(i, fold)], shuffled.test_scores[Item(i, fold)])
    var rng = RandomState(42)
    var mutable_first = learning_curve[Adapter, DType.int64](model, X, y, Adapter(), Scorer(False), splits, None, None, rng, train_sizes=sizes.copy(), shuffle=True)
    emit("mutable_first", mutable_first)
    var mutable_second = learning_curve[Adapter, DType.int64](model, X, y, Adapter(), Scorer(False), splits, None, None, rng, train_sizes=sizes.copy(), shuffle=True)
    emit("mutable_second", mutable_second)
    print("rng_next", rng.next_uint32())
    var untouched = RandomState(42)
    var expected_rng = untouched.copy()
    _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, untouched)
    assert_equal(untouched.next_uint32(), expected_rng.next_uint32())
    var weighted = learning_curve[Adapter, DType.int64](model, X, y, Adapter(), Scorer(False), splits, weights.copy(), None, train_sizes=sizes.copy(), shuffle=True, random_state=42)
    emit("weighted_fit", weighted)
    var weighted_both = learning_curve[Adapter, DType.int64](model, X, y, Adapter(), Scorer(False), splits, weights.copy(), weights.copy(), train_sizes=sizes.copy(), shuffle=True, random_state=42)
    emit("weighted_both", weighted_both)
    var scoring_failed = learning_curve[Adapter, DType.int64](model, X, y, Adapter(), Scorer(True), splits, None, None, train_sizes=sizes.copy(), error_score=ErrorPolicy(False, -123))
    emit("scoring_failed", scoring_failed)
    var uneven = List[IndexSplit]()
    uneven.append(IndexSplit(counts([7, 5, 3, 1, 0, 2]), counts([4, 6])))
    uneven.append(IndexSplit(counts([6, 4, 0]), counts([7, 5, 3, 2, 1])))
    uneven.append(IndexSplit(counts([1, 2, 3, 4, 5, 6, 7]), counts([0])))
    var irregular = learning_curve[Adapter, DType.int64](model, X, y, Adapter(), Scorer(False), uneven, None, None, train_sizes=sizes.copy())
    emit("uneven", irregular)
    var backing = fractions([1.0, 0.5, 0.01])
    var reversed_sizes = backing.view_with_layout(Shape(3), Strides(-1), 2)
    var strided = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, train_sizes=reversed_sizes.copy())
    emit("strided", strided)
    with assert_raises():
        _ = learning_curve[Adapter](DummyRegressor(strategy="invalid"), X, y, Adapter(), Scorer(False), splits, None, None, error_score=ErrorPolicy(False, -99))
    with assert_raises():
        _ = learning_curve[Adapter](DummyRegressor(strategy="invalid"), X, y, Adapter(), Scorer(False), splits, None, None)
    with assert_raises():
        _ = learning_curve[Adapter](DummyRegressor(strategy="invalid"), X, y, Adapter(), Scorer(False), splits, None, None, error_score=ErrorPolicy(True, 0))
    with assert_raises():
        _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(True), splits, None, None, error_score=ErrorPolicy(True, 0))
    for value in List[Int]([0, -1, 7]):
        with assert_raises():
            _ = learning_curve[Adapter, DType.int64](model, X, y, Adapter(), Scorer(False), splits, None, None, train_sizes=counts([value]))
    for value in List[Float64]([0.0, -0.1, 1.1, bitcast[DType.float64](UInt64(0x7ff8000000000000)), bitcast[DType.float64](UInt64(0x7ff0000000000000))]):
        with assert_raises():
            _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, train_sizes=fractions([value]))
    with assert_raises():
        _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, train_sizes=Dense(Shape(0)))
    with assert_raises():
        _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, train_sizes=Dense(Shape(1, 1)))
    with assert_raises():
        _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), List[IndexSplit](), None, None)
    with assert_raises():
        _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, exploit_incremental_learning=True)
    with assert_raises():
        _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, shuffle=True)
    with assert_raises():
        _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, shuffle=True, random_state=-1)
    with assert_raises():
        _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, shuffle=True, random_state=4294967296)
    var after = model.predict(X)
    for i in range(8):
        assert_equal(original[Item(i)], after[Item(i)])
        assert_equal(y[Item(i)], Float64(i * i))
        assert_equal(weights[Item(i)], Float64(i + 1))
    assert_equal(sizes[Item(0)], 6)
    assert_equal(fractions_input[Item(0)], 1.0)
    assert_equal(splits[0].train[Item(0)], 2)
    assert_equal(splits[0].test[Item(0)], 0)
    var unfitted = DummyRegressor()
    _ = learning_curve[Adapter](unfitted, X, y, Adapter(), Scorer(False), splits, None, None)
    assert_false(unfitted.__sklearn_is_fitted__())
    splits[0].test[Item(0)] = 8
    with assert_raises():
        _ = learning_curve[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None)
    print("PASS learning curves: real estimator, sizes, weights, RNG, errors, timing and isolation")
