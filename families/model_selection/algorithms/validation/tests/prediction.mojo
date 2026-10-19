"""Actual DummyClassifier predictions and class responses, including String labels."""
from moxie.dummy import DummyClassifier as _DummyClassifier, Label
from moxie._labels import StringTarget
from moxie.model_selection import TypedScorer, StringPredictionAdapter, ClassPredictionAdapter, ClassResponse, CVClassPrediction, cross_val_predict, encode_prediction_target, KFold, IndexArray, IndexSplit
from moxie.model_selection._split import _rows
from numojo import NDArray, Shape, Item, zeros
from std.testing import assert_equal, assert_true, assert_false, assert_raises
comptime DummyClassifier = _DummyClassifier[multioutput=True]  # Row-wise label targets, as before the Moxie merge.

comptime Dense = NDArray[DType.float64]
comptime Labels = List[List[Label]]


def labels(y: StringTarget) raises -> Labels:
    var result = Labels()
    for i in range(y.rows()):
        var row = List[Label]()
        for j in range(y.columns()):
            row.append(Label(y.get(i, j)))
        result.append(row^)
    return result^


def string_rows(y: StringTarget, rows: IndexArray) raises -> StringTarget:
    var values = List[String]()
    for i in range(rows.size):
        for j in range(y.columns()):
            values.append(y.get(Int(rows[Item(i)]), j))
    if y.rank() == 1:
        return StringTarget(values)
    return StringTarget(values, rows.size, y.columns())


struct Accuracy(TypedScorer):
    comptime Estimator = DummyClassifier
    comptime Input = Dense
    comptime Target = StringTarget
    comptime Metadata = Optional[Dense]
    def __init__(out self): pass
    def score(self, estimator: DummyClassifier, X: Dense, y: StringTarget, metadata: Optional[Dense]) raises -> Float64:
        return estimator.score(X, labels(y), metadata)


@fieldwise_init
struct StringAdapter(StringPredictionAdapter):
    var matrix: Bool
    comptime Scorer = Accuracy
    def n_samples(self, X: Dense, y: StringTarget) raises -> Int:
        if X.shape[0] != y.rows():
            raise Error("ValueError: inconsistent target rows")
        return y.rows()
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: StringTarget, rows: IndexArray) raises -> StringTarget:
        return string_rows(y, rows)
    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if metadata:
            return _rows(metadata.value(), rows)
        return None
    def fit(self, mut estimator: DummyClassifier, X: Dense, y: StringTarget, metadata: Optional[Dense]) raises:
        if metadata:
            _ = estimator.fit(X, labels(y), sample_weight=metadata.value())
        else:
            _ = estimator.fit(X, labels(y))
    def predict(self, estimator: DummyClassifier, X: Dense) raises -> StringTarget:
        var prediction = estimator.predict(X)
        var values = List[String]()
        for i in range(len(prediction)):
            for j in range(len(prediction[i])):
                values.append(prediction[i][j].text.copy())
        if self.matrix:
            return StringTarget(values, X.shape[0], estimator.n_outputs_.value())
        return StringTarget(values)


struct ResponseAdapter(ClassPredictionAdapter):
    comptime ResponseDType = DType.float64
    def __init__(out self): pass
    comptime Scorer = Accuracy
    def n_samples(self, X: Dense, y: StringTarget) raises -> Int:
        if X.shape[0] != y.rows():
            raise Error("ValueError: inconsistent target rows")
        return y.rows()
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: StringTarget, rows: IndexArray) raises -> StringTarget:
        return string_rows(y, rows)
    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if metadata:
            return _rows(metadata.value(), rows)
        return None
    def fit(self, mut estimator: DummyClassifier, X: Dense, y: StringTarget, metadata: Optional[Dense]) raises:
        if metadata:
            _ = estimator.fit(X, labels(y), sample_weight=metadata.value())
        else:
            _ = estimator.fit(X, labels(y))
    def encode_target(self, y: StringTarget) raises -> IndexArray:
        return encode_prediction_target(y)
    def fit_encoded(self, mut estimator: DummyClassifier, X: Dense, y: IndexArray, metadata: Optional[Dense]) raises:
        var target = Labels()
        for i in range(y.shape[0]):
            var row = List[Label]()
            for j in range(1 if y.ndim == 1 else y.shape[1]):
                row.append(Label(Int(y[Item(i)] if y.ndim == 1 else y[Item(i, j)])))
            target.append(row^)
        if metadata:
            _ = estimator.fit(X, target, sample_weight=metadata.value())
        else:
            _ = estimator.fit(X, target)
    def response(self, estimator: DummyClassifier, X: Dense, method: String) raises -> ClassResponse[DType.float64]:
        var values: List[Dense]
        if method == "predict_proba":
            values = estimator.predict_proba(X)
        elif method == "predict_log_proba":
            values = estimator.predict_log_proba(X)
        else:
            raise Error("ValueError: DummyClassifier has no decision_function")
        var classes = List[IndexArray]()
        for output in range(len(estimator.classes_)):
            var codes = IndexArray(Shape(len(estimator.classes_[output])))
            for i in range(codes.size):
                codes[Item(i)] = Int64(estimator.classes_[output][i].integer)
            classes.append(codes^)
        return ClassResponse[DType.float64](values^, classes^, estimator.n_outputs_.value() > 1)


def emit(name: String, result: CVClassPrediction[DType.float64]) raises:
    print("container", name, Int(result.is_list), len(result.values))
    for output in range(len(result.values)):
        for row in range(result.values[output].shape[0]):
            for col in range(result.values[output].shape[1]):
                print("response", name, output, row, col, result.values[output][Item(row, col)])


def main() raises:
    var X = zeros[DType.float64](Shape(8, 2))
    var words: List[String] = ["é", "z", "a", "a", "z", "z", "z", "z"]
    var y = StringTarget(words)
    var weights = Dense(Shape(8))
    for i in range(8): weights[Item(i)] = Float64(i + 1)
    var splits = List[IndexSplit]()
    var cursor = KFold(4).split(8)
    while True:
        var pair = cursor.next()
        if not pair: break
        # Deliberately reverse held-out order: output must still follow input rows.
        var a = pair.value().test[Item(0)]
        pair.value().test[Item(0)] = pair.value().test[Item(1)]
        pair.value().test[Item(1)] = a
        splits.append(pair.value().copy())
    var model = DummyClassifier()
    StringAdapter(False).fit(model, X, y, None)
    var predicted = cross_val_predict[StringAdapter](model, X, y, StringAdapter(False), splits, None)
    assert_equal(predicted.rank(), 1)
    assert_equal(predicted.rows(), 8)
    for i in range(8): print("string", "single", i, 0, predicted.get(i))
    for method in List[String](["predict_proba", "predict_log_proba"]):
        var result = cross_val_predict[ResponseAdapter](model, X, y, ResponseAdapter(), splits, None, method=method)
        emit(method, result)
        assert_false(result.is_list)
        var snapshot = result.array()
        assert_equal(snapshot.shape[1], 3)
        var repeated = cross_val_predict[ResponseAdapter](model, X, y, ResponseAdapter(), splits, None, method=method)
        for i in range(8):
            for j in range(3):
                assert_equal(result.values[0][Item(i, j)], repeated.values[0][Item(i, j)])
    var weighted = cross_val_predict[ResponseAdapter](model, X, y, ResponseAdapter(), splits, weights.copy(), method="predict_proba")
    emit("weighted", weighted)
    var multiwords = List[String]()
    for i in range(8):
        multiwords.append(words[i].copy())
        multiwords.append("yes" if i < 2 else "no")
    var multi_y = StringTarget(multiwords, 8, 2)
    var multi = cross_val_predict[StringAdapter](model, X, multi_y, StringAdapter(True), splits, None)
    assert_equal(multi.rank(), 2)
    assert_equal(multi.columns(), 2)
    for i in range(8):
        for j in range(2): print("string", "multi", i, j, multi.get(i, j))
    var multi_result = cross_val_predict[ResponseAdapter](model, X, multi_y, ResponseAdapter(), splits, None, method="predict_proba")
    assert_true(multi_result.is_list)
    emit("multi", multi_result)
    with assert_raises(): _ = multi_result.array()
    var multi_log = cross_val_predict[ResponseAdapter](model, X, multi_y, ResponseAdapter(), splits, None, method="predict_log_proba")
    emit("multi_log", multi_log)
    var binary_y = StringTarget(List[String](["z", "z", "z", "z", "a", "a", "a", "a"]))
    var binary_splits = List[IndexSplit]()
    var binary_cursor = KFold(2).split(8)
    while True:
        var pair = binary_cursor.next()
        if not pair: break
        binary_splits.append(pair.value().copy())
    var binary = cross_val_predict[ResponseAdapter](model, X, binary_y, ResponseAdapter(), binary_splits, None, method="predict_proba")
    emit("binary_missing", binary)
    var binary_log = cross_val_predict[ResponseAdapter](model, X, binary_y, ResponseAdapter(), binary_splits, None, method="predict_log_proba")
    emit("binary_missing_log", binary_log)
    assert_equal(model.classes_[0][0].kind, 2)
    assert_equal(model.classes_[0][0].text, "a")
    assert_equal(model.n_outputs_.value(), 1)
    assert_equal(y.get(0), "é")
    var unfitted = DummyClassifier()
    _ = cross_val_predict[ResponseAdapter](unfitted, X, y, ResponseAdapter(), splits, None, method="predict_proba")
    assert_false(unfitted.__sklearn_is_fitted__())
    with assert_raises():
        _ = cross_val_predict[ResponseAdapter](model, X, y, ResponseAdapter(), splits, None, method="bad")
    with assert_raises():
        _ = cross_val_predict[StringAdapter](model, X, y, StringAdapter(False), splits, None, method="predict_proba")
    with assert_raises():
        _ = cross_val_predict[ResponseAdapter](model, X, y, ResponseAdapter(), splits, None, method="decision_function")
    splits[1].test[Item(0)] = splits[0].test[Item(0)]
    with assert_raises():
        _ = cross_val_predict[StringAdapter](model, X, y, StringAdapter(False), splits, None)
    with assert_raises():
        _ = cross_val_predict[ResponseAdapter](model, X, y, ResponseAdapter(), splits, None, method="predict_proba")
    print("PASS real String/class responses, multioutput, missing classes, weights, partitions and isolation")
