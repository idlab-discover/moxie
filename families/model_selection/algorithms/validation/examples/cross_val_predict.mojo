"""Out-of-fold probabilities in sorted global class order, even with rare labels."""
from moxie.dummy import DummyClassifier as _DummyClassifier, Label
from moxie.model_selection import ClassPredictionAdapter, TypedScorer, ClassResponse, cross_val_predict, encode_prediction_target, IndexArray, IndexSplit, KFold
from moxie.model_selection._split import _rows
from numojo import NDArray, Shape, Item, zeros
comptime DummyClassifier = _DummyClassifier[multioutput=True]  # Row-wise label targets, as before the Moxie merge.

comptime Dense = NDArray[DType.float64]


def labels(y: IndexArray) raises -> List[List[Label]]:
    var result = List[List[Label]]()
    for i in range(y.size): result.append(List[Label]([Label(Int(y[Item(i)]))]))
    return result^


struct Accuracy(TypedScorer):
    comptime Estimator = DummyClassifier
    comptime Input = Dense
    comptime Target = IndexArray
    comptime Metadata = Optional[Dense]
    def __init__(out self): pass
    def score(self, estimator: DummyClassifier, X: Dense, y: IndexArray, metadata: Optional[Dense]) raises -> Float64:
        return estimator.score(X, labels(y), metadata)


struct ClassifierAdapter(ClassPredictionAdapter):
    comptime Scorer = Accuracy
    comptime ResponseDType = DType.float64
    def __init__(out self): pass
    def n_samples(self, X: Dense, y: IndexArray) raises -> Int:
        if y.ndim != 1 or y.size != X.shape[0]:
            raise Error("ValueError: expected sample-aligned label vector")
        return y.size
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: IndexArray, rows: IndexArray) raises -> IndexArray:
        return _rows(y, rows)
    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if metadata: return _rows(metadata.value(), rows)
        return None
    def fit(self, mut estimator: DummyClassifier, X: Dense, y: IndexArray, metadata: Optional[Dense]) raises:
        if metadata:
            _ = estimator.fit(X, labels(y), sample_weight=metadata.value())
        else:
            _ = estimator.fit(X, labels(y))
    def encode_target(self, y: IndexArray) raises -> IndexArray:
        return encode_prediction_target(y)
    def fit_encoded(self, mut estimator: DummyClassifier, X: Dense, y: IndexArray, metadata: Optional[Dense]) raises:
        self.fit(estimator, X, y, metadata)
    def response(self, estimator: DummyClassifier, X: Dense, method: String) raises -> ClassResponse[DType.float64]:
        var values: List[Dense]
        if method == "predict_proba":
            values = estimator.predict_proba(X)
        elif method == "predict_log_proba":
            values = estimator.predict_log_proba(X)
        else:
            raise Error("ValueError: DummyClassifier does not support decision_function")
        var classes = List[IndexArray]()
        var codes = IndexArray(Shape(len(estimator.classes_[0])))
        for i in range(codes.size): codes[Item(i)] = Int64(estimator.classes_[0][i].integer)
        classes.append(codes^)
        return ClassResponse[DType.float64](values^, classes^, False)


def main() raises:
    var X = zeros[DType.float64](Shape(8, 1))
    var y = IndexArray(Shape(8))
    for i in range(8): y[Item(i)] = Int64(30 if i == 0 else (10 if i < 3 else 20))
    var splits = List[IndexSplit]()
    var cursor = KFold(4).split(8)
    while True:
        var pair = cursor.next()
        if not pair: break
        splits.append(pair.value().copy())
    var result = cross_val_predict[ClassifierAdapter](
        DummyClassifier(), X, y, ClassifierAdapter(), splits, None,
        method="predict_proba",
    )
    var probabilities = result.array()
    print("Probability columns: labels 10, 20, 30")
    for i in range(8):
        print("row", i, probabilities[Item(i, 0)], probabilities[Item(i, 1)], probabilities[Item(i, 2)])
