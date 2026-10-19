from moxie.base import BaseEstimator
from moxie.model_selection import TypedScorer, PredictionAdapter, cross_validate, cross_val_predict, IndexArray, IndexSplit, KFold, ErrorPolicy
from moxie.model_selection._split import _rows
from numojo import NDArray, Shape, Item, zeros
from std.testing import assert_equal, assert_true, assert_false, assert_raises

comptime Dense = NDArray[DType.float64]

@fieldwise_init
struct Probe(BaseEstimator):
    comptime Params = Int
    var parameter: Int
    var fit_count: Int
    def get_params(self, deep: Bool = True) raises -> Int:
        return self.parameter
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        self.parameter = params
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self(self.parameter, 0)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fit_count != 0

@fieldwise_init
struct ProbeScorer(TypedScorer):
    comptime Estimator = Probe
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = Int
    var offset: Int
    def score(self, estimator: Probe, X: Dense, y: Dense, metadata: Int) raises -> Float64:
        assert_equal(estimator.fit_count, 1)
        return Float64(estimator.parameter + self.offset + metadata)

struct ProbeAdapter(PredictionAdapter):
    comptime Scorer = ProbeScorer
    comptime PredictionDType = DType.float32
    def __init__(out self):
        pass
    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return X.shape[0]
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)
    def metadata_rows(self, metadata: Int, rows: IndexArray) raises -> Int:
        return metadata
    def fit(self, mut estimator: Probe, X: Dense, y: Dense, metadata: Int) raises:
        assert_equal(estimator.fit_count, 0)
        if metadata == 1 and y[Item(0)] == 0:
            raise Error("intentional partial fit failure")
        estimator.fit_count += 1
    def predict(self, estimator: Probe, X: Dense) raises -> NDArray[DType.float32]:
        assert_equal(estimator.fit_count, 1)
        var result = NDArray[DType.float32](Shape(X.shape[0], 2))
        for i in range(X.shape[0]):
            result[Item(i, 0)] = Float32(X[Item(i, 0)])
            result[Item(i, 1)] = Float32(estimator.parameter)
        return result^


def main() raises:
    var X = Dense(Shape(8, 1))
    var y = Dense(Shape(8))
    for i in range(8):
        X[Item(i, 0)] = Float64(i)
        y[Item(i)] = Float64(i)
    var model = Probe(7, 99)
    var cursor = KFold(4).split(8)
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var scorers: List[ProbeScorer] = [ProbeScorer(1), ProbeScorer(3)]
    var names: List[String] = ["a", "b"]
    var result = cross_validate[ProbeAdapter](model, X, y, ProbeAdapter(), scorers, names, splits, 0, 10, return_estimator=True)
    assert_equal(model.fit_count, 99)
    assert_false(Bool(result.indices))
    assert_false(Bool(result.scores.train_scores))
    for i in range(4):
        assert_equal(result.estimators.value()[i].fit_count, 1)
        assert_equal(result.scores.test_scores[Item(0, i, 0)], 18.0)
        assert_equal(result.scores.test_scores[Item(0, i, 1)], 20.0)
    result.estimators.value()[0].parameter = 77
    assert_equal(result.estimators.value()[1].parameter, 7)
    assert_equal(model.parameter, 7)
    var partial = cross_validate[ProbeAdapter](model, X, y, ProbeAdapter(), scorers, names, splits, 1, 0, error_score=ErrorPolicy(False, -123))
    assert_false(partial.status[0].failed)
    for i in range(1, 4):
        assert_true(partial.status[i].failed)
        assert_equal(partial.status[i].score_time, 0.0)
        assert_equal(partial.scores.test_scores[Item(0, i, 0)], -123.0)
    var output = cross_val_predict[ProbeAdapter](model, X, y, ProbeAdapter(), splits, 0)
    assert_equal(output.ndim, 2)
    for i in range(8):
        assert_equal(output[Item(i, 0)], Float32(i))
        assert_equal(output[Item(i, 1)], 7.0)
    with assert_raises():
        var empty = List[IndexSplit]()
        _ = cross_validate[ProbeAdapter](model, X, y, ProbeAdapter(), scorers, names, empty, 0, 0)
    with assert_raises():
        var duplicate: List[String] = ["a", "a"]
        _ = cross_validate[ProbeAdapter](model, X, y, ProbeAdapter(), scorers, duplicate, splits, 0, 0)
    print("PASS isolated clones, one fit per metric set, partial failures and matrix predictions")
