"""Forests inside clone, Pipeline, cross_val_score and GridSearchCV.

Each composed result is checked against the same steps fitted by hand, so the
composition adds no behavior of its own (sklearn's Pipeline/CV/search contract).
Integer seeds make every refit reproducible, as in scikit-learn.
"""
import numojo as nm
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true
from moxie.base import clone
from moxie.ensemble import RandomForestClassifier, ExtraTreesRegressor
from moxie.preprocessing import StandardScaler
from moxie.pipeline import make_pipeline
from moxie.model_selection import (
    GridSearchCV, IndexArray, IndexSplit, SearchAdapter, SearchScorer, StratifiedKFold,
    KFold, cross_val_score,
)
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float32]
comptime Values = NDArray[DType.float64]
comptime Forest = RandomForestClassifier[]


def dataset() raises -> Tuple[Dense, Values]:
    var X = Dense(Shape(60, 3))
    var y = Values(Shape(60))
    for i in range(60):
        var label = i % 3
        X[Item(i, 0)] = Float32(label) * 1.5 + Float32((i * 7) % 5) / 3
        X[Item(i, 1)] = Float32((i * 5) % 11) / 4 - Float32(label)
        X[Item(i, 2)] = 100.0 + Float32((i * 3) % 7) * 10
        y[Item(i)] = Float64(label * 2 - 1)
    return (X^, y^)


def codes(y: Values) raises -> IndexArray:
    var out = IndexArray(Shape(y.size))
    for i in range(y.size):
        out[Item(i)] = Int64(y.item(i))
    return out^


def stratified(y: Values) raises -> List[IndexSplit]:
    var cursor = StratifiedKFold(3).split(y.size, codes(y))
    var out = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        out.append(pair.value().copy())
    return out^


def assert_same(a: Values, b: Values) raises:
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_equal(a.item(i), b.item(i))


struct Accuracy(SearchScorer):
    comptime Estimator = Forest
    comptime Input = Dense
    comptime Target = Values
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def score(self, estimator: Forest, X: Dense, y: Values, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct DepthSearch(SearchAdapter):
    """Tunes max_depth; sklearn's GridSearchCV(forest, {"max_depth": [...]})."""
    comptime Scorer = Accuracy
    comptime Value = Int

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def n_samples(self, X: Dense, y: Values) raises -> Int:
        return y.size

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Values, rows: IndexArray) raises -> Values:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: Forest, X: Dense, y: Values, metadata: NoneType) raises:
        _ = estimator.fit(X, y)

    def configure(self, mut estimator: Forest, params: Dict[String, Int]) raises:
        var config = estimator.get_params()
        for key in params.keys():
            if key != "max_depth":
                raise Error("ValueError: Invalid parameter " + key)
            config.max_depth = params[key]
        _ = estimator.set_params(config)

    def default_scorer(self) raises -> Accuracy:
        return Accuracy()

    def default_cv(self, X: Dense, y: Values) raises -> List[IndexSplit]:
        return stratified(y)


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var y = data[1].copy()

    # clone copies configuration, not the fitted members; refitting reproduces them.
    var model = Forest(15, max_depth=3, random_state=0, oob_score=True)
    _ = model.fit(X, y)
    var copied = clone(model)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_equal(len(copied), 0)
    assert_equal(copied.get_params().max_depth.value(), 3)
    _ = copied.fit(X, y)
    assert_same(model.predict(X), copied.predict(X))
    assert_equal(model.oob_score_, copied.oob_score_)
    assert_same(model.feature_importances_, copied.feature_importances_)

    # Pipeline: Float32 scaling then a forest equals the manual composition.
    var pipe = make_pipeline(StandardScaler[DType.float32](), Forest(10, random_state=0))
    _ = pipe.fit(X, y)
    var scaler = StandardScaler[DType.float32]()
    _ = scaler.fit(X)
    var Xs = scaler.transform(X)
    var manual = Forest(10, random_state=0)
    _ = manual.fit(Xs, y)
    assert_same(pipe.predict(X), manual.predict(Xs))
    assert_equal(pipe.score(X, y), manual.score(Xs, y))
    var pipe_proba = pipe.predict_proba(X)
    var manual_proba = manual.predict_proba(Xs)
    assert_same(pipe_proba, manual_proba)

    # A regression forest in a pipeline as well.
    var target = Values(Shape(60))
    for i in range(60):
        target[Item(i)] = Float64(X.item(i, 0)) * 2 - Float64(X.item(i, 1))
    var rpipe = make_pipeline(StandardScaler[DType.float32](), ExtraTreesRegressor(8, random_state=3))
    _ = rpipe.fit(X, target)
    var rmanual = ExtraTreesRegressor(8, random_state=3)
    _ = rmanual.fit(Xs, target)
    assert_same(rpipe.predict(X), rmanual.predict(Xs))

    # cross_val_score clones the forest per stratified fold; scores equal manual fits.
    var splits = stratified(y)
    var template = Forest(10, max_depth=2, random_state=0)
    var scores = cross_val_score[DepthSearch](template, X, y, DepthSearch(), Accuracy(), splits, None, None)
    assert_equal(scores.size, 3)
    for k in range(3):
        var fold = clone(template)
        _ = fold.fit(_rows(X, splits[k].train), _rows(y, splits[k].train))
        assert_equal(scores.item(k), fold.score(_rows(X, splits[k].test), _rows(y, splits[k].test)))

    # GridSearchCV over max_depth refits the best configuration on all rows.
    var depths: List[Int] = [1, 2, 4]
    var grid = Dict[String, List[Int]]()
    grid["max_depth"] = depths.copy()
    var search = GridSearchCV[DepthSearch](Forest(10, random_state=0), grid, DepthSearch(), cv=stratified(y))
    search.fit(X, y, None, None)
    var means = search.cv_results_().mean_test_score.copy()
    var best = 0
    for i in range(3):
        var candidate = Forest(10, max_depth=depths[i], random_state=0)
        var by_hand = cross_val_score[DepthSearch](candidate, X, y, DepthSearch(), Accuracy(), splits, None, None)
        var mean = (by_hand.item(0) + by_hand.item(1) + by_hand.item(2)) / 3
        assert_equal(means.item(i), mean)
        if mean > means.item(best):
            best = i
    assert_equal(search.best_params_()["max_depth"], depths[best])
    var refit = Forest(10, max_depth=depths[best], random_state=0)
    _ = refit.fit(X, y)
    assert_same(search.best_estimator_().predict(X), refit.predict(X))
    print("PASS: forest clone, Pipeline, cross_val_score and GridSearchCV composition")
