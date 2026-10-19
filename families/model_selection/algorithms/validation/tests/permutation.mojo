from evaluation import Adapter, Scorer, Dense
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import permutation_test_score, PermutationTestResult, PermutationCV, SplitLabels, KFold, StratifiedKFold, IndexSplit
from moxie._random import RandomState
from numojo import Shape, Item, zeros
from std.testing import assert_equal, assert_true, assert_false, assert_raises
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.


struct StratifiedCV(PermutationCV):
    comptime Adapter = Adapter
    var calls: Int
    def __init__(out self):
        self.calls = 0
    def split(mut self, X: Dense, y: Dense, groups: Optional[SplitLabels]) raises -> List[IndexSplit]:
        self.calls += 1
        var cursor = StratifiedKFold(2).split(y.size, y=SplitLabels(y))
        var splits = List[IndexSplit]()
        while True:
            var pair = cursor.next()
            if not pair:
                break
            splits.append(pair.value().copy())
        return splits^


def emit(name: String, result: PermutationTestResult) raises:
    print("permutation", name, result.score, result.pvalue)
    for i in range(result.permutation_scores.size):
        print("null", name, i, result.permutation_scores[Item(i)])


def main() raises:
    var X = zeros[DType.float64](Shape(8, 2))
    var y = Dense(Shape(8))
    var weights = Dense(Shape(8))
    var labels = List[String]()
    for i in range(8):
        y[Item(i)] = Float64(i * i)
        weights[Item(i)] = Float64(i + 1)
        labels.append("z" if i % 3 == 0 else "a")
    var groups = SplitLabels(labels)
    var splits = List[IndexSplit]()
    var cursor = KFold(4).split(8)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var model = DummyRegressor()
    _ = model.fit(X, y)
    var before = model.predict(X)
    var plain = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, n_permutations=7)
    emit("plain", plain)
    var grouped = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, groups=groups.copy(), n_permutations=7, random_state=42)
    emit("grouped", grouped)
    var weighted = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, weights.copy(), None, n_permutations=7, random_state=42)
    emit("weighted", weighted)
    var both = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, weights.copy(), weights.copy(), n_permutations=7, random_state=42)
    emit("both", both)
    var rng = RandomState(42)
    var first = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, rng, groups=groups.copy(), n_permutations=7)
    emit("mutable_first", first)
    var second = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, rng, groups=groups.copy(), n_permutations=7)
    emit("mutable_second", second)
    print("permutation_rng", rng.next_uint32())
    var repeated = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, n_permutations=7)
    for i in range(7):
        assert_equal(repeated.permutation_scores[Item(i)], plain.permutation_scores[Item(i)])
    assert_equal(repeated.pvalue, plain.pvalue)
    var singleton = Dense(Shape(8))
    for i in range(8):
        singleton[Item(i)] = Float64(i) / 2.0
    var tied = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, groups=SplitLabels(singleton), n_permutations=1)
    assert_equal(tied.score, tied.permutation_scores[Item(0)])
    assert_equal(tied.pvalue, 1.0)
    emit("singletons", tied)
    var class_y = Dense(Shape(8))
    for i in range(8):
        class_y[Item(i)] = Float64(i % 2)
    var stratified = StratifiedCV()
    var dynamic = permutation_test_score[StratifiedCV](model, X, class_y, Adapter(), Scorer(False), stratified, None, None, n_permutations=7)
    assert_equal(stratified.calls, 8)
    emit("stratified", dynamic)
    for count in List[Int]([0, -1]):
        with assert_raises():
            _ = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, n_permutations=count)
    with assert_raises():
        _ = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, random_state=-1)
    with assert_raises():
        _ = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), splits, None, None, groups=SplitLabels(List[String](["a"])))
    with assert_raises():
        _ = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(True), splits, None, None)
    with assert_raises():
        _ = permutation_test_score[Adapter](DummyRegressor(strategy="invalid"), X, y, Adapter(), Scorer(False), splits, None, None)
    with assert_raises():
        _ = permutation_test_score[Adapter](model, X, y, Adapter(), Scorer(False), List[IndexSplit](), None, None)
    var after = model.predict(X)
    for i in range(8):
        assert_equal(before[Item(i)], after[Item(i)])
        assert_equal(y[Item(i)], Float64(i * i))
        assert_equal(weights[Item(i)], Float64(i + 1))
    var unfitted = DummyRegressor()
    _ = permutation_test_score[Adapter](unfitted, X, y, Adapter(), Scorer(False), splits, None, None, n_permutations=1)
    assert_false(unfitted.__sklearn_is_fitted__())
    print("PASS permutation real-estimator scores, groups, weights, dynamic CV, isolation and errors")
