"""Nested CV: cross_val_score(GridSearchCV(SVC)) with concrete scorer/adapter types."""
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true
from moxie.svm import SVC
from moxie.base import clone
from moxie.preprocessing import StandardScaler
from moxie.pipeline import Pipeline
from moxie.model_selection import (
    RandomizedSearchCV, ParameterGrid, GridSearchCV, IndexArray, IndexSplit, KFold, SearchAdapter, SearchScorer,
    CVAdapter, TypedScorer, cross_val_score,
)
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float64]
comptime Labels = List[Float64]


def dataset() raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(30, 3))
    var y = Labels()
    for i in range(30):
        X[Item(i, 0)] = Float64(i % 3) + Float64(i) / 30
        X[Item(i, 1)] = Float64((i * 7) % 5)
        X[Item(i, 2)] = 10.0 * Float64((i * 11) % 4) - Float64(i % 3)
        y.append(Float64(i % 3))
    return (X^, y^)


def folds(n: Int) raises -> List[IndexSplit]:
    var out = List[IndexSplit]()
    var cursor = KFold(3).split(n)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        out.append(pair.value().copy())
    return out^


def take(y: Labels, rows: IndexArray) raises -> Labels:
    var out = Labels()
    for i in range(rows.size):
        out.append(y[Int(rows.item(i))])
    return out^


struct Accuracy(SearchScorer):
    comptime Estimator = SVC[]
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: SVC[], X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct Inner(SearchAdapter):
    comptime Scorer = Accuracy
    comptime Value = Float64

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        return len(y)

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return take(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: SVC[], X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(X, y)

    def configure(self, mut estimator: SVC[], params: Dict[String, Float64]) raises:
        var config = estimator.get_params()
        for key in params.keys():
            if key == "C":
                config.C = params[key]
            else:
                raise Error("ValueError: Invalid parameter " + key)
        _ = estimator.set_params(config)

    def default_scorer(self) raises -> Accuracy:
        return Accuracy()

    def default_cv(self, X: Dense, y: Labels) raises -> List[IndexSplit]:
        return folds(len(y))


comptime Search = GridSearchCV[Inner]


struct OuterAccuracy(TypedScorer):
    comptime Estimator = Search
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: Search, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct Outer(CVAdapter):
    comptime Scorer = OuterAccuracy

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        return len(y)

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return take(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: Search, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(X, y)


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var y = data[1].copy()
    var grid = Dict[String, List[Float64]]()
    grid["C"] = [0.1, 1.0, 10.0]
    var search = Search(SVC(), grid, Inner())
    var scores = cross_val_score[Outer](search, X, y, Outer(), OuterAccuracy(), folds(30), None, None)
    var expected: List[Float64] = [0.3, 0.4, 0.3]
    for k in range(scores.size):
        assert_equal(scores.item(k), expected[k])
        print("nested_cv", scores.item(k))
    assert_true(not search.__sklearn_is_fitted__())
    var cv = folds(30)
    for k in range(3):
        var fitted = clone(search)
        _ = fitted.fit(_rows(X, cv[k].train), take(y, cv[k].train))
        assert_equal(fitted.score(_rows(X, cv[k].test), take(y, cv[k].test)), scores.item(k))
        print("nested_best", k, fitted.best_params_()["C"])
    var pipeline = Pipeline(StandardScaler(), clone(search))
    var predicted = pipeline.fit(X, y).predict(X)
    var scaler = StandardScaler()
    var transformed = scaler.fit_transform(X)
    var manual = clone(search)
    _ = manual.fit(transformed, y, None, None)
    var wanted = manual.predict(transformed)
    for i in range(len(y)):
        assert_equal(predicted[i], wanted[i])
        print("search_final_prediction", predicted[i])
    assert_equal(pipeline.score(X, y), manual.score(transformed, y))
    print("search_final_score", pipeline.score(X, y))
    assert_true(not clone(pipeline).__sklearn_is_fitted__())
    var random = RandomizedSearchCV[Inner](SVC(), ParameterGrid[Float64](grid), Inner(), n_iter=3, random_state=7)
    var random_pipeline = Pipeline(StandardScaler(), random^)
    var random_prediction = random_pipeline.fit(X, y).predict(X)
    for i in range(len(y)):
        assert_equal(random_prediction[i], predicted[i])
    print("PASS nested")
