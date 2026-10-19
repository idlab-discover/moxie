from moxie._nb_gaussian import GaussianNB
from moxie._naive_bayes_dense import NumericDomain
from moxie._labels import encode_column
from moxie.model_selection._validation import IncrementalCVAdapter, learning_curve
from moxie.model_selection import TypedScorer, IndexArray, IndexSplit, KFold
from moxie.model_selection._split import _rows
from moxie._random import RandomState
from numojo import NDArray, Shape, Item
from std.testing import assert_equal

comptime Dense = NDArray[DType.float64]
comptime Model = GaussianNB[NumericDomain[DType.int64]]

struct Accuracy(TypedScorer):
    def __init__(out self):
        pass
    comptime Estimator = Model
    comptime Input = Dense
    comptime Target = IndexArray
    comptime Metadata = NoneType
    def score(self, estimator: Model, X: Dense, y: IndexArray, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)

struct GaussianAdapter(IncrementalCVAdapter):
    def __init__(out self):
        pass
    comptime Scorer = Accuracy
    def n_samples(self, X: Dense, y: IndexArray) raises -> Int:
        assert_equal(X.shape[0], y.size)
        return y.size
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: IndexArray, rows: IndexArray) raises -> IndexArray:
        return _rows(y, rows)
    def metadata_rows(self, metadata: NoneType, rows: IndexArray) -> NoneType:
        return None
    def fit(self, mut estimator: Model, X: Dense, y: IndexArray, metadata: NoneType) raises:
        _ = estimator.fit(X, y)
    def partial_fit(self, mut estimator: Model, X: Dense, y: IndexArray, metadata: NoneType, full_target: IndexArray) raises:
        # GaussianNB accepts duplicate class labels and owns their unique encoding.
        _ = estimator.partial_fit(X, y, classes=full_target.copy())


def main() raises:
    var X = Dense(Shape(12, 2))
    var y = IndexArray(Shape(12))
    for i in range(12):
        X[Item(i, 0)] = Float64((i * 7) % 11)
        X[Item(i, 1)] = Float64(i % 3) + Float64(i) * 0.17
        y[Item(i)] = Int64(i % 2)
    var cv = List[IndexSplit]()
    var cursor = KFold(3).split(12)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        cv.append(pair.value().copy())
    var sizes = IndexArray(Shape(3))
    sizes[Item(0)] = 2
    sizes[Item(1)] = 5
    sizes[Item(2)] = 8
    var model = Model()
    for mode in range(2):
        var rng = RandomState(42)
        var result = learning_curve[GaussianAdapter, DType.int64](model, X, y, GaussianAdapter(), Accuracy(), cv, None, None, rng, train_sizes=sizes.copy(), exploit_incremental_learning=True, shuffle=mode == 1)
        for size in range(3):
            for fold in range(3):
                print("gaussian_incremental", mode, size, fold, result.train_scores[Item(size, fold)], result.test_scores[Item(size, fold)])
        print("gaussian_rng", mode, rng.next_uint32())
    assert_equal(model.__sklearn_is_fitted__(), False)
