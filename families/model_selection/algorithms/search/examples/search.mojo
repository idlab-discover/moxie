"""Tune a real DummyRegressor with the native typed search API."""
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid, SearchAdapter, SearchScorer, IndexArray, IndexSplit, KFold
from moxie.model_selection._split import _rows
from numojo import NDArray, Shape, Item
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.

comptime Dense = NDArray[DType.float64]

struct R2(SearchScorer):
    comptime Estimator = DummyRegressor
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = Optional[Dense]
    def __init__(out self):
        pass
    def score(self, estimator: DummyRegressor, X: Dense, y: Dense, metadata: Self.Metadata) raises -> Float64:
        return estimator.score(X, y, metadata)

struct RegressionAdapter(SearchAdapter):
    comptime Scorer = R2
    comptime Value = Float64
    def __init__(out self):
        pass
    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise Error("ValueError: expected aligned matrix and vector")
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
    def configure(self, mut estimator: DummyRegressor, params: Dict[String, Float64]) raises:
        for key in params.keys():
            if key != "constant":
                raise Error("ValueError: this adapter tunes only constant")
        if "constant" in params:
            var constant = Dense(Shape(1))
            constant[Item(0)] = params["constant"]
            _ = estimator.set_params(constant=Optional[Dense](constant^))
    def default_scorer(self) raises -> R2:
        return R2()
    def default_cv(self, X: Dense, y: Dense) raises -> List[IndexSplit]:
        var cursor = KFold().split(self.n_samples(X, y))
        var folds = List[IndexSplit]()
        while True:
            var pair = cursor.next()
            if not pair:
                break
            folds.append(pair.value().copy())
        return folds^


def main() raises:
    var X = Dense(Shape(10, 1))
    var y = Dense(Shape(10))
    for i in range(10):
        X[Item(i, 0)] = Float64(i)
        y[Item(i)] = Float64(i)
    var grid = Dict[String, List[Float64]]()
    grid["constant"] = [0.0, 2.0, 4.0, 5.0, 9.0]
    var model = DummyRegressor(strategy="constant")
    var search = GridSearchCV[RegressionAdapter](model, grid, RegressionAdapter())
    search.fit(X, y, None, None)
    print("grid winner:", search.best_params_()["constant"], "mean CV R2:", search.best_score_())
    var random = RandomizedSearchCV[RegressionAdapter](model, ParameterGrid[Float64](grid), RegressionAdapter(), n_iter=3, random_state=42)
    random.fit(X, y, None, None)
    print("random winner:", random.best_params_()["constant"], "mean CV R2:", random.best_score_())
    print("refitted prediction:", random.predict(X)[Item(0)])
