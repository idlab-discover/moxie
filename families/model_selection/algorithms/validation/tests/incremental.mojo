from lifecycle import Probe
from moxie.model_selection._validation import IncrementalCVAdapter, learning_curve
from moxie.model_selection import TypedScorer, IndexArray, IndexSplit, KFold, ErrorPolicy
from moxie.model_selection._split import _rows
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true, assert_raises

comptime Dense = NDArray[DType.float64]

struct IncrementalScorer(TypedScorer):
    comptime Estimator = Probe
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = Dense
    var fail: Bool
    def __init__(out self, fail: Bool = False):
        self.fail = fail
    def score(self, estimator: Probe, X: Dense, y: Dense, metadata: Dense) raises -> Float64:
        if self.fail:
            raise Error("score failure")
        assert_equal(metadata.size, X.shape[0])
        return Float64(estimator.parameter)

struct IncrementalAdapter(IncrementalCVAdapter):
    comptime Scorer = IncrementalScorer
    var fail: Bool
    def __init__(out self, fail: Bool = False):
        self.fail = fail
    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return X.shape[0]
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)
    def metadata_rows(self, metadata: Dense, rows: IndexArray) raises -> Dense:
        return _rows(metadata, rows)
    def fit(self, mut estimator: Probe, X: Dense, y: Dense, metadata: Dense) raises:
        raise Error("ordinary fit must not run")
    def partial_fit(self, mut estimator: Probe, X: Dense, y: Dense, metadata: Dense, full_target: Dense) raises:
        if self.fail:
            raise Error("partial fit failure")
        assert_equal(full_target.size, 8)
        assert_equal(metadata.size, y.size)
        for i in range(y.size):
            assert_equal(metadata[Item(i)], y[Item(i)] + 1)
            estimator.parameter += Int(y[Item(i)] + metadata[Item(i)])
        estimator.fit_count += 1


def main() raises:
    var X = Dense(Shape(8, 1))
    var y = Dense(Shape(8))
    var weights = Dense(Shape(8))
    for i in range(8):
        X[Item(i, 0)] = Float64(i)
        y[Item(i)] = Float64(i)
        weights[Item(i)] = Float64(i + 1)
    var cv = List[IndexSplit]()
    var cursor = KFold(4).split(8)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        cv.append(pair.value().copy())
    var sizes = IndexArray(Shape(3))
    sizes[Item(0)] = 1
    sizes[Item(1)] = 3
    sizes[Item(2)] = 6
    var original = Probe(0, 99)
    var result = learning_curve[IncrementalAdapter, DType.int64](original, X, y, IncrementalAdapter(), IncrementalScorer(), cv, weights, weights, train_sizes=sizes.copy(), exploit_incremental_learning=True, return_times=True)
    for fold in range(4):
        for size in range(3):
            var expected = 0
            for i in range(Int(sizes[Item(size)])):
                expected += 2 * Int(cv[fold].train[Item(i)]) + 1
            assert_equal(result.train_scores[Item(size, fold)], Float64(expected))
            assert_equal(result.test_scores[Item(size, fold)], Float64(expected))
            assert_true(result.fit_times.value()[Item(size, fold)] >= 0)
            assert_true(result.score_times.value()[Item(size, fold)] >= 0)
    assert_equal(original.fit_count, 99)
    assert_equal(original.parameter, 0)
    var errors = learning_curve[IncrementalAdapter, DType.int64](original, X, y, IncrementalAdapter(), IncrementalScorer(True), cv, weights, weights, train_sizes=sizes.copy(), exploit_incremental_learning=True, error_score=ErrorPolicy(False, -17))
    for i in range(3):
        for fold in range(4):
            assert_equal(errors.train_scores[Item(i, fold)], -17.0)
            assert_equal(errors.test_scores[Item(i, fold)], -17.0)
    with assert_raises():
        _ = learning_curve[IncrementalAdapter, DType.int64](original, X, y, IncrementalAdapter(True), IncrementalScorer(), cv, weights, weights, train_sizes=sizes.copy(), exploit_incremental_learning=True, error_score=ErrorPolicy(False, -17))
    print("PASS incremental clone, disjoint chunks, full target, cumulative scores, metadata and failures")
