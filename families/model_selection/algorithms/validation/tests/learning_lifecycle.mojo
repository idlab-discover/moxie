"""Capability probes for clone/order contracts; these are not estimators."""
from curve_lifecycle import CurveAdapter, CurveScorer
from lifecycle import Probe, Dense
from moxie.model_selection import learning_curve, LearningCurveResult, KFold, IndexArray, IndexSplit, ErrorPolicy
from numojo import Shape, Item, zeros
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_raises


def emit(name: String, result: LearningCurveResult) raises:
    for size in range(result.train_sizes.size):
        for fold in range(result.train_scores.shape[1]):
            print("probe", name, size, fold, result.train_scores[Item(size, fold)], result.test_scores[Item(size, fold)])


def main() raises:
    var X = zeros[DType.float64](Shape(8, 1))
    var y = Dense(Shape(8))
    for i in range(8):
        y[Item(i)] = Float64(i)
    var sizes = IndexArray(Shape(3))
    sizes[Item(0)] = 6
    sizes[Item(1)] = 1
    sizes[Item(2)] = 3
    var cursor = KFold(4).split(8)
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var original = Probe(7, 99)
    var result = learning_curve[CurveAdapter, DType.int64](original, X, y, CurveAdapter(True), CurveScorer(False), splits, 0, 10, train_sizes=sizes.copy())
    emit("order", result)
    var partial = learning_curve[CurveAdapter, DType.int64](original, X, y, CurveAdapter(False), CurveScorer(False), splits, 1, 10, train_sizes=sizes.copy(), error_score=ErrorPolicy(False, -999), return_times=True)
    emit("partial", partial)
    var scoring = learning_curve[CurveAdapter, DType.int64](original, X, y, CurveAdapter(False), CurveScorer(True), splits, 0, 10, train_sizes=sizes.copy(), error_score=ErrorPolicy(False, -456))
    emit("scoring", scoring)
    var nan_failed = learning_curve[CurveAdapter, DType.int64](original, X, y, CurveAdapter(False), CurveScorer(False), splits, 1, 10, train_sizes=sizes.copy())
    for size in range(3):
        for fold in range(4):
            assert_equal(partial.train_scores[Item(size, fold)], result.train_scores[Item(size, fold)] if fold == 0 else -999.0)
            assert_equal(partial.test_scores[Item(size, fold)], result.test_scores[Item(size, fold)] if fold == 0 else -999.0)
            assert_equal(scoring.train_scores[Item(size, fold)], result.train_scores[Item(size, fold)] if fold == 0 else -456.0)
            assert_equal(scoring.test_scores[Item(size, fold)], -456.0 if fold == 0 else result.test_scores[Item(size, fold)])
            if fold > 0:
                assert_equal(partial.score_times.value()[Item(size, fold)], 0.0)
                assert_true(isnan(nan_failed.train_scores[Item(size, fold)]))
                assert_true(isnan(nan_failed.test_scores[Item(size, fold)]))
    assert_equal(original.fit_count, 99)
    assert_equal(original.parameter, 7)
    with assert_raises():
        _ = learning_curve[CurveAdapter](original, X, y, CurveAdapter(False), CurveScorer(False), splits, 1, 10, error_score=ErrorPolicy(True, 0))
    # Invalid later folds must be caught before the first fit (trace stays empty).
    splits[3].train[Item(0)] = 8
    with assert_raises():
        _ = learning_curve[CurveAdapter](original, X, y, CurveAdapter(True), CurveScorer(False), splits, 0, 10)
    splits[3].train[Item(0)] = 0
    splits[0].train = IndexArray(Shape(0))
    with assert_raises():
        _ = learning_curve[CurveAdapter](original, X, y, CurveAdapter(True), CurveScorer(False), splits, 0, 10)
    print("PASS learning curve lifecycle, partial failures, early validation and failure timing")
