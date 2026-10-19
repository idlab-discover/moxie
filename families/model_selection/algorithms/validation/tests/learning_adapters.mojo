"""CV-only adapters: Float32, optional targets and pairwise prefix slicing."""
from lifecycle import Probe
from moxie.model_selection import CVAdapter, TypedScorer, learning_curve, IndexArray, IndexSplit, KFold
from numojo import NDArray, Shape, Item
from std.testing import assert_equal

comptime Dense32 = NDArray[DType.float32]


struct KernelScorer(TypedScorer):
    comptime Estimator = Probe
    comptime Input = Dense32
    comptime Target = Optional[Dense32]
    comptime Metadata = Int
    def __init__(out self):
        pass
    def score(self, estimator: Probe, X: Dense32, y: Optional[Dense32], metadata: Int) raises -> Float64:
        assert_equal(estimator.fit_count, 1)
        assert_equal(X.shape[1], estimator.parameter)
        if y:
            assert_equal(y.value().shape[0], X.shape[0])
            assert_equal(y.value().ndim, 2)
            assert_equal(y.value().shape[1], 2)
            for i in range(X.shape[0]):
                assert_equal(y.value()[Item(i, 0)] + y.value()[Item(i, 1)], 0.0)
        var total: Float64 = Float64(metadata)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                total += Float64(X[Item(i, j)])
        return total


struct KernelAdapter(CVAdapter):
    comptime Scorer = KernelScorer
    def __init__(out self):
        pass
    def n_samples(self, X: Dense32, y: Optional[Dense32]) raises -> Int:
        assert_equal(X.shape[0], X.shape[1])
        return X.shape[0]
    def input_rows(self, X: Dense32, rows: IndexArray, train: IndexArray) raises -> Dense32:
        var result = Dense32(Shape(rows.size, train.size))
        for i in range(rows.size):
            for j in range(train.size):
                result[Item(i, j)] = X[Item(Int(rows[Item(i)]), Int(train[Item(j)]))]
        return result^
    def target_rows(self, y: Optional[Dense32], rows: IndexArray) raises -> Optional[Dense32]:
        if not y:
            return None
        var result = Dense32(Shape(rows.size, 2))
        for i in range(rows.size):
            for j in range(2):
                result[Item(i, j)] = y.value()[Item(Int(rows[Item(i)]), j)]
        return result^
    def metadata_rows(self, metadata: Int, rows: IndexArray) raises -> Int:
        # Non-sample-aligned fit options stay unchanged.
        return metadata
    def fit(self, mut estimator: Probe, X: Dense32, y: Optional[Dense32], metadata: Int) raises:
        assert_equal(estimator.fit_count, 0)
        assert_equal(metadata, 99)
        assert_equal(X.shape[0], X.shape[1])
        if y:
            assert_equal(y.value().shape[0], X.shape[0])
        estimator.parameter = X.shape[0]
        estimator.fit_count = 1


def main() raises:
    var X = Dense32(Shape(8, 8))
    var y = Dense32(Shape(8, 2))
    for i in range(8):
        y[Item(i, 0)] = Float32(i)
        y[Item(i, 1)] = Float32(-i)
        for j in range(8):
            X[Item(i, j)] = Float32(100 * i + j)
    var sizes = IndexArray(Shape(2))
    sizes[Item(0)] = 2
    sizes[Item(1)] = 6
    var splits = List[IndexSplit]()
    var cursor = KFold(4).split(8)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var model = Probe(7, 99)
    var unsupervised = learning_curve[KernelAdapter, DType.int64](model, X, None, KernelAdapter(), KernelScorer(), splits, 99, 10, train_sizes=sizes.copy())
    var multioutput = learning_curve[KernelAdapter, DType.int64](model, X, y.copy(), KernelAdapter(), KernelScorer(), splits, 99, 10, train_sizes=sizes.copy())
    for size in range(2):
        for fold in range(4):
            var count = Int(sizes[Item(size)])
            var expected_train: Float64 = 10
            var expected_test: Float64 = 10
            for i in range(count):
                for j in range(count):
                    expected_train += Float64(100 * Int(splits[fold].train[Item(i)]) + Int(splits[fold].train[Item(j)]))
            for i in range(2):
                for j in range(count):
                    expected_test += Float64(100 * Int(splits[fold].test[Item(i)]) + Int(splits[fold].train[Item(j)]))
            assert_equal(unsupervised.train_scores[Item(size, fold)], expected_train)
            assert_equal(unsupervised.test_scores[Item(size, fold)], expected_test)
            assert_equal(multioutput.train_scores[Item(size, fold)], expected_train)
            assert_equal(multioutput.test_scores[Item(size, fold)], expected_test)
    assert_equal(model.fit_count, 99)
    assert_equal(model.parameter, 7)
    print("PASS CV-only learning adapter: Float32 pairwise kernels, optional/multioutput targets")
