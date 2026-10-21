"""Trees inside clone, Pipeline, cross_val_score and GridSearchCV.

Each composed result is checked against the same steps fitted by hand, so the
composition adds no behavior of its own (sklearn's Pipeline/CV/search contract).
"""
import numojo as nm
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true
from moxie.base import clone
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier
from moxie.preprocessing import StandardScaler
from moxie.pipeline import make_pipeline
from moxie.model_selection import (
    GridSearchCV, IndexArray, IndexSplit, SearchAdapter, SearchScorer, StratifiedKFold,
    cross_val_score,
)
from moxie.model_selection._validation import CVAdapter
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float32]
comptime Labels = NDArray[DType.int64]
comptime Tree = DecisionTreeClassifier[DType.int64]


def dataset() raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(60, 3))
    var y = Labels(Shape(60))
    for i in range(60):
        var label = i % 3
        X[Item(i, 0)] = Float32(label) * 1.5 + Float32((i * 7) % 5) / 3
        X[Item(i, 1)] = Float32((i * 5) % 11) / 4 - Float32(label)
        X[Item(i, 2)] = 100.0 + Float32((i * 3) % 7) * 10
        y[Item(i)] = Int64(label * 2 - 1)
    return (X^, y^)


def stratified(y: Labels) raises -> List[IndexSplit]:
    var cursor = StratifiedKFold(3).split(y.size, y.copy())
    var out = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        out.append(pair.value().copy())
    return out^


def assert_same(a: Labels, b: Labels) raises:
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_equal(a.item(i), b.item(i))


struct Accuracy(SearchScorer):
    comptime Estimator = Tree
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def score(self, estimator: Tree, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct DepthSearch(SearchAdapter):
    """Tunes max_depth; sklearn's GridSearchCV(tree, {"max_depth": [...]})."""
    comptime Scorer = Accuracy
    comptime Value = Int

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        return y.size

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: Tree, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(X, y)

    def configure(self, mut estimator: Tree, params: Dict[String, Int]) raises:
        var config = estimator.get_params()
        for key in params.keys():
            if key != "max_depth":
                raise Error("ValueError: Invalid parameter " + key)
            config.max_depth = params[key]
        _ = estimator.set_params(config)

    def default_scorer(self) raises -> Accuracy:
        return Accuracy()

    def default_cv(self, X: Dense, y: Labels) raises -> List[IndexSplit]:
        return stratified(y)


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var y = data[1].copy()

    # clone copies configuration, not the fitted tree; refitting reproduces it.
    var model = Tree(max_depth=3, random_state=0)
    _ = model.fit(X, y)
    var copied = clone(model)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_equal(copied.get_params().max_depth.value(), 3)
    _ = copied.fit(X, y)
    assert_same(model.predict(X), copied.predict(X))
    assert_equal(model.tree_.node_count, copied.tree_.node_count)

    # Pipeline: Float32 scaling then a tree equals the manual composition.
    var pipe = make_pipeline(StandardScaler[DType.float32](), Tree(random_state=0))
    _ = pipe.fit(X, y)
    var scaler = StandardScaler[DType.float32]()
    _ = scaler.fit(X)
    var Xs = scaler.transform(X)
    var manual = Tree(random_state=0)
    _ = manual.fit(Xs, y)
    assert_same(pipe.predict(X), manual.predict(Xs))
    assert_equal(pipe.score(X, y), manual.score(Xs, y))
    var pipe_proba = pipe.predict_proba(X)
    var manual_proba = manual.predict_proba(Xs)
    assert_equal(pipe_proba[0].item(5, 1), manual_proba[0].item(5, 1))

    # cross_val_score clones the tree per stratified fold; scores equal manual fits.
    var splits = stratified(y)
    var template = Tree(max_depth=2, random_state=0)
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
    var search = GridSearchCV[DepthSearch](Tree(random_state=0), grid, DepthSearch(), cv=stratified(y))
    search.fit(X, y, None, None)
    var means = search.cv_results_().mean_test_score.copy()
    var best = 0
    for i in range(3):
        var candidate = Tree(max_depth=depths[i], random_state=0)
        var by_hand = cross_val_score[DepthSearch](candidate, X, y, DepthSearch(), Accuracy(), splits, None, None)
        var mean = (by_hand.item(0) + by_hand.item(1) + by_hand.item(2)) / 3
        assert_equal(means.item(i), mean)
        if mean > means.item(best):
            best = i
    assert_equal(search.best_params_()["max_depth"], depths[best])
    var refit = Tree(max_depth=depths[best], random_state=0)
    _ = refit.fit(X, y)
    assert_same(search.best_estimator_().predict(X), refit.predict(X))
    print("PASS: tree clone, Pipeline, cross_val_score and GridSearchCV composition")
