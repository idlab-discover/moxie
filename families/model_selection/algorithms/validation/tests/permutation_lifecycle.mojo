"""Reference lifecycle probe: one clone per dataset, refits within that dataset."""
from lifecycle import Probe, ProbeAdapter, Dense
from moxie.model_selection import CVAdapter, TypedScorer, permutation_test_score, IndexArray, IndexSplit, KFold
from moxie._random import RandomState
from numojo import Shape, Item, zeros
from std.memory import bitcast
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_raises


@fieldwise_init
struct StateScore(TypedScorer):
    comptime Estimator = Probe
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = Int
    var nan: Bool
    def score(self, estimator: Probe, X: Dense, y: Dense, metadata: Int) raises -> Float64:
        if self.nan:
            return bitcast[DType.float64](UInt64(0x7ff8000000000000))
        return Float64(estimator.fit_count)


@fieldwise_init
struct StateAdapter(CVAdapter):
    comptime Scorer = StateScore
    var trace: Bool
    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return ProbeAdapter().n_samples(X, y)
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return ProbeAdapter().input_rows(X, rows, train)
    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return ProbeAdapter().target_rows(y, rows)
    def metadata_rows(self, metadata: Int, rows: IndexArray) raises -> Int:
        return metadata
    def fit(self, mut estimator: Probe, X: Dense, y: Dense, metadata: Int) raises:
        estimator.fit_count += 1
        assert_equal(metadata, 99)
        if self.trace:
            print("refit", estimator.fit_count)


def main() raises:
    var X = zeros[DType.float64](Shape(8, 1))
    var y = Dense(Shape(8))
    for i in range(8):
        y[Item(i)] = Float64(i)
    var splits = List[IndexSplit]()
    var cursor = KFold(4).split(8)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var model = Probe(7, 77)
    var result = permutation_test_score[StateAdapter](model, X, y, StateAdapter(True), StateScore(False), splits, 99, 0, n_permutations=3)
    assert_equal(result.score, 2.5)
    assert_equal(result.pvalue, 1.0)
    for i in range(3):
        assert_equal(result.permutation_scores[Item(i)], 2.5)
    assert_equal(model.fit_count, 77)
    var nan = permutation_test_score[StateAdapter](model, X, y, StateAdapter(False), StateScore(True), splits, 99, 0, n_permutations=3)
    assert_true(isnan(nan.score))
    assert_equal(nan.pvalue, 0.25)
    for i in range(3):
        assert_true(isnan(nan.permutation_scores[Item(i)]))
    var rng = RandomState(42)
    var before = rng.copy()
    splits[3].train[Item(0)] = 99
    with assert_raises():
        _ = permutation_test_score[StateAdapter](model, X, y, StateAdapter(True), StateScore(False), splits, 99, 0, rng, n_permutations=3)
    assert_equal(rng.next_uint32(), before.next_uint32())
    print("PASS permutation clone/refit lifecycle, ties, NaNs and validation before RNG use")
