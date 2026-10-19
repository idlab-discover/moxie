"""Capability probes for decision shapes and class ordering, not estimators."""
from moxie.base import BaseEstimator
from moxie._labels import encode_column
from moxie.model_selection import ClassPredictionAdapter, TypedScorer, ClassResponse, CVClassPrediction, cross_val_predict, encode_prediction_target, IndexArray, IndexSplit, KFold
from moxie.model_selection._validation import _align_class_response
from moxie.model_selection._split import _rows
from numojo import NDArray, Shape, Item, zeros
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises

comptime Dense = NDArray[DType.float64]


struct DecisionModel(BaseEstimator):
    comptime Params = Int
    var mode: Int
    var fits: Int
    var classes: IndexArray
    def __init__(out self, mode: Int = 0) raises:
        self.mode = mode
        self.fits = 0
        self.classes = IndexArray(Shape(0))
    def get_params(self, deep: Bool = True) raises -> Int: return self.mode
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        self.mode = params
        return self
    def __sklearn_clone__(self) raises -> Self: return Self(self.mode)
    def __sklearn_is_fitted__(self) -> Bool: return self.fits > 0


struct DecisionScorer(TypedScorer):
    comptime Estimator = DecisionModel
    comptime Input = Dense
    comptime Target = IndexArray
    comptime Metadata = Int
    def __init__(out self): pass
    def score(self, estimator: DecisionModel, X: Dense, y: IndexArray, metadata: Int) raises -> Float64:
        raise Error("scoring must not be used by cross_val_predict")


struct DecisionAdapter[dtype: DType](ClassPredictionAdapter):
    comptime Scorer = DecisionScorer
    comptime ResponseDType = Self.dtype
    def __init__(out self): pass
    def n_samples(self, X: Dense, y: IndexArray) raises -> Int:
        assert_equal(X.shape[0], y.shape[0])
        return y.shape[0]
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: IndexArray, rows: IndexArray) raises -> IndexArray:
        return _rows(y, rows)
    def metadata_rows(self, metadata: Int, rows: IndexArray) raises -> Int: return metadata
    def fit(self, mut estimator: DecisionModel, X: Dense, y: IndexArray, metadata: Int) raises:
        raise Error("encoded fit must be used for class responses")
    def encode_target(self, y: IndexArray) raises -> IndexArray:
        return encode_prediction_target(y)
    def fit_encoded(self, mut estimator: DecisionModel, X: Dense, y: IndexArray, metadata: Int) raises:
        assert_equal(estimator.fits, 0)
        assert_equal(metadata, 99)
        if estimator.mode == 2: raise Error("intentional fit failure")
        estimator.classes = encode_column(y).classes.copy()
        estimator.fits += 1
    def response(self, estimator: DecisionModel, X: Dense, method: String) raises -> ClassResponse[Self.dtype]:
        assert_equal(method, "decision_function")
        assert_equal(estimator.fits, 1)
        var k = estimator.classes.size
        var columns = k * (k - 1) // 2 if estimator.mode == 1 else k
        var values = NDArray[Self.dtype](Shape(X.shape[0]) if k <= 2 else Shape(X.shape[0], columns))
        for i in range(X.shape[0]):
            if k <= 2:
                values[Item(i)] = (X[Item(i, 0)] + Float64(estimator.classes[Item(k - 1)])).cast[Self.dtype]()
            else:
                for j in range(columns):
                    var label = Float64(j) if estimator.mode == 1 else Float64(estimator.classes[Item(j)])
                    values[Item(i, j)] = (X[Item(i, 0)] * 10.0 + label).cast[Self.dtype]()
        var arrays = List[NDArray[Self.dtype]]()
        arrays.append(values^)
        var classes = List[IndexArray]()
        classes.append(estimator.classes.copy())
        return ClassResponse[Self.dtype](arrays^, classes^, False)



struct IndicatorAdapter(ClassPredictionAdapter):
    comptime Scorer = DecisionScorer
    comptime ResponseDType = DType.float64
    def __init__(out self): pass
    def n_samples(self, X: Dense, y: IndexArray) raises -> Int:
        return DecisionAdapter[DType.float64]().n_samples(X, y)
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: IndexArray, rows: IndexArray) raises -> IndexArray:
        return _rows(y, rows)
    def metadata_rows(self, metadata: Int, rows: IndexArray) raises -> Int: return metadata
    def fit(self, mut estimator: DecisionModel, X: Dense, y: IndexArray, metadata: Int) raises:
        raise Error("expected encoded fitting")
    def encode_target(self, y: IndexArray) raises -> IndexArray:
        return encode_prediction_target(y)
    def fit_encoded(self, mut estimator: DecisionModel, X: Dense, y: IndexArray, metadata: Int) raises:
        assert_equal(estimator.fits, 0)
        assert_equal(y.ndim, 2)
        estimator.classes = IndexArray(Shape(y.shape[1]))
        for i in range(y.shape[1]): estimator.classes[Item(i)] = Int64(i)
        estimator.fits = 1
    def response(self, estimator: DecisionModel, X: Dense, method: String) raises -> ClassResponse[DType.float64]:
        return DecisionAdapter[DType.float64]().response(estimator, X, method)

def emit[dtype: DType](name: String, result: CVClassPrediction[dtype]) raises:
    assert_equal(len(result.values), 1)
    var values = result.array()
    for i in range(values.shape[0]):
        if values.ndim == 1:
            print("decision", name, i, -1, values[Item(i)])
        else:
            for j in range(values.shape[1]):
                print("decision", name, i, j, values[Item(i, j)])


def main() raises:
    var X = Dense(Shape(12, 1))
    var y = IndexArray(Shape(12))
    var full_y = IndexArray(Shape(12))
    var binary_y = IndexArray(Shape(12))
    for i in range(12):
        X[Item(i, 0)] = Float64(i)
        y[Item(i)] = Int64((i // 2 + 1) * 10)
        full_y[Item(i)] = Int64(i % 4 + 10)
        binary_y[Item(i)] = Int64(i % 2 + 10)
    var splits = List[IndexSplit]()
    var cursor = KFold(3).split(12)
    while True:
        var pair = cursor.next()
        if not pair: break
        splits.append(pair.value().copy())
    var model = DecisionModel()
    model.fits = 77
    var missing = cross_val_predict[DecisionAdapter[DType.float64]](model, X, y, DecisionAdapter[DType.float64](), splits, 99, method="decision_function")
    emit("missing64", missing)
    var missing32 = cross_val_predict[DecisionAdapter[DType.float32]](model, X, y, DecisionAdapter[DType.float32](), splits, 99, method="decision_function")
    emit("missing32", missing32)
    var binary = cross_val_predict[DecisionAdapter[DType.float64]](model, X, binary_y, DecisionAdapter[DType.float64](), splits, 99, method="decision_function")
    emit("binary", binary)
    var ovo = cross_val_predict[DecisionAdapter[DType.float64]](DecisionModel(1), X, full_y, DecisionAdapter[DType.float64](), splits, 99, method="decision_function")
    emit("ovo", ovo)
    with assert_raises():
        _ = cross_val_predict[DecisionAdapter[DType.float64]](DecisionModel(1), X, y, DecisionAdapter[DType.float64](), splits, 99, method="decision_function")
    with assert_raises():
        _ = cross_val_predict[DecisionAdapter[DType.float64]](DecisionModel(2), X, y, DecisionAdapter[DType.float64](), splits, 99, method="decision_function")
    for i in range(12): binary_y[Item(i)] = Int64(i // 6)
    var halves = List[IndexSplit]()
    var half_cursor = KFold(2).split(12)
    while True:
        var pair = half_cursor.next()
        if not pair: break
        halves.append(pair.value().copy())
    with assert_raises():
        _ = cross_val_predict[DecisionAdapter[DType.float64]](model, X, binary_y, DecisionAdapter[DType.float64](), halves, 99, method="decision_function")
    assert_equal(model.fits, 77)
    var indicators = IndexArray(Shape(12, 3))
    for i in range(12):
        for j in range(3): indicators[Item(i, j)] = Int64((i + j) % 2)
    var indicator_result = cross_val_predict[IndicatorAdapter](model, X, indicators, IndicatorAdapter(), splits, 99, method="decision_function")
    emit("indicator", indicator_result)
    # Exact Int64 label identity beyond Float64's integer precision.
    var large = IndexArray(Shape(3))
    large[Item(0)] = 9007199254740993
    large[Item(1)] = 9007199254740992
    large[Item(2)] = 9007199254740993
    var encoded = encode_prediction_target(large)
    assert_equal(encoded[Item(0)], 1)
    assert_equal(encoded[Item(1)], 0)
    assert_equal(encoded[Item(2)], 1)
    var classes = IndexArray(Shape(1))
    classes[Item(0)] = 1
    var response = zeros[DType.float32](Shape(2, 1))
    var logs = _align_class_response(classes, response, 2, "predict_log_proba")
    assert_equal(logs[Item(0, 0)], bitcast[DType.float32](UInt32(0xff7fffff)))
    assert_equal(logs[Item(0, 1)], 0.0)
    classes[Item(0)] = 2
    with assert_raises(): _ = _align_class_response(classes, response, 2, "predict_proba")
    var bad = IndexArray(Shape(2))
    bad[Item(0)] = 1
    bad[Item(1)] = 1
    with assert_raises(): _ = _align_class_response(bad, response, 2, "predict_proba")
    bad[Item(1)] = 0
    with assert_raises(): _ = _align_class_response(bad, response, 2, "predict_proba")
    splits[2].test[Item(0)] = 99
    with assert_raises():
        _ = cross_val_predict[DecisionAdapter[DType.float64]](model, X, y, DecisionAdapter[DType.float64](), splits, 99, method="decision_function")
    print("PASS decision shape/alignment, Float32 sentinel, large labels, cloning and errors")
