"""Compare mean and median baselines on the same four folds."""
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import TypedScorer, CVAdapter, ParameterAdapter, KFold, IndexArray, IndexSplit, validation_curve
from moxie.model_selection._split import _rows
from numojo import NDArray, Shape, Item, zeros
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.

comptime Dense = NDArray[DType.float64]

struct R2Scorer(TypedScorer):
    comptime Estimator = DummyRegressor
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = Optional[Dense]
    def __init__(out self):
        pass
    def score(self, estimator: DummyRegressor, X: Dense, y: Dense, metadata: Optional[Dense]) raises -> Float64:
        return estimator.score(X, y, metadata)


struct RegressionAdapter(CVAdapter, ParameterAdapter):
    comptime Scorer = R2Scorer
    comptime Value = String
    def __init__(out self):
        pass
    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise Error("ValueError: expected aligned dense X and y")
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
        if metadata:
            _ = estimator.fit(X, y, sample_weight=metadata.value())
        else:
            _ = estimator.fit(X, y)
    def configure(self, mut estimator: DummyRegressor, params: Dict[String, String]) raises:
        for key in params:
            if key != "strategy":
                raise Error("ValueError: this adapter accepts only strategy")
        _ = estimator.set_params(strategy=params["strategy"].copy())


def main() raises:
    var X = zeros[DType.float64](Shape(8, 1))
    var y = Dense(Shape(8))
    for i in range(8):
        y[Item(i)] = Float64(i * i)
    var splits = List[IndexSplit]()
    var cursor = KFold(4).split(8)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var strategies: List[String] = ["mean", "median"]
    var scores = validation_curve[RegressionAdapter](
        DummyRegressor(), X, y, RegressionAdapter(), R2Scorer(), splits,
        None, None, param_name="strategy", param_range=strategies,
    )
    for strategy in range(len(strategies)):
        for fold in range(len(splits)):
            print(strategies[strategy], "fold", fold, "train R2", scores[0][Item(strategy, fold)], "test R2", scores[1][Item(strategy, fold)])
