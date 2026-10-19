from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import SearchScorer, SearchAdapter, IndexArray, IndexSplit, KFold
from moxie.model_selection._split import _rows
from numojo import NDArray, Shape, Item
from std.math import abs
from std.memory import bitcast
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.

comptime Dense = NDArray[DType.float64]

@fieldwise_init
struct Scorer(SearchScorer):
    comptime Estimator = DummyRegressor
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = Optional[Dense]
    var mode: Int
    def score(self, estimator: DummyRegressor, X: Dense, y: Dense, metadata: Self.Metadata) raises -> Float64:
        if self.mode == 4:
            return estimator.score(X, y, metadata)
        if self.mode == 2:
            raise Error("intentional scorer failure")
        if self.mode == 3:
            return bitcast[DType.float64](UInt64(0x7ff8000000000000))
        var prediction = estimator.predict(X)
        var total: Float64 = 0
        var mass: Float64 = 0
        for i in range(y.shape[0]):
            var w: Float64 = 1
            if metadata:
                w = metadata.value()[Item(i)]
            var delta = prediction[Item(i)] - y[Item(i)]
            total += w * (delta * delta if self.mode == 0 else abs(delta))
            mass += w
        return -total / mass

struct Adapter(SearchAdapter):
    comptime Scorer = Scorer
    comptime Value = Float64
    def __init__(out self):
        pass
    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise Error("ValueError: invalid dense shapes")
        return y.shape[0]
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)
    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if metadata:
            return _rows(metadata.value(), rows)
        return None
    def fit(self, mut estimator: DummyRegressor, X: Dense, y: Dense, metadata: Optional[Dense]) raises:
        # Native lifecycle fixture: every CV fit succeeds, full-data refit fails.
        if estimator.strategy == "constant" and estimator.constant.value()[Item(0)] == 99 and X.shape[0] == 10:
            raise Error("intentional refit failure")
        if metadata:
            _ = estimator.fit(X, y, sample_weight=metadata.value())
        else:
            _ = estimator.fit(X, y)
    def configure(self, mut estimator: DummyRegressor, params: Dict[String, Float64]) raises:
        for key in params.keys():
            if key != "constant" and key != "invalid":
                raise Error("ValueError: unknown parameter")
        if "constant" in params:
            var value = Dense(Shape(1))
            value[Item(0)] = params["constant"]
            _ = estimator.set_params(strategy=String("constant"), constant=Optional[Dense](value^))
        if "invalid" in params:
            _ = estimator.set_params(strategy=String("invalid"))
    def default_scorer(self) raises -> Scorer:
        return Scorer(4)
    def default_cv(self, X: Dense, y: Dense) raises -> List[IndexSplit]:
        var cursor = KFold().split(self.n_samples(X, y))
        var splits = List[IndexSplit]()
        while True:
            var pair = cursor.next()
            if not pair:
                break
            splits.append(pair.value().copy())
        return splits^


def data() raises -> Tuple[Dense, Dense]:
    var X = Dense(Shape(10, 2))
    var y = Dense(Shape(10))
    for i in range(10):
        X[Item(i, 0)] = Float64(i)
        X[Item(i, 1)] = Float64(10 - i)
        y[Item(i)] = Float64(i)
    return (X^, y^)


def grid(values: List[Float64]) -> Dict[String, List[Float64]]:
    var result = Dict[String, List[Float64]]()
    result["constant"] = values.copy()
    return result^


def main():
    pass
