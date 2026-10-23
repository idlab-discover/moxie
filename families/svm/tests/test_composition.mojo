"""Kernel SVMs inside clone, Pipeline, cross_val_score and GridSearchCV.

Each composed result is checked against the same steps fitted by hand, so the
composition adds no behavior of its own (sklearn's Pipeline/CV/search contract).
LIBSVM fitting is deterministic without probability, so the comparisons are exact.
"""
import numojo as nm
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true
from moxie.base import clone
from moxie.svm import SVC, SVR
from moxie.preprocessing import StandardScaler
from moxie.pipeline import make_pipeline
from moxie.model_selection import (
    GridSearchCV, IndexArray, IndexSplit, SearchAdapter, SearchScorer, StratifiedKFold,
    cross_val_score,
)
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float64]
comptime Labels = List[Float64]


def dataset() raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(60, 3))
    var y = Labels()
    for i in range(60):
        var label = i % 3
        X[Item(i, 0)] = Float64(label) * 1.2 + Float64((i * 7) % 5) / 3
        X[Item(i, 1)] = Float64((i * 5) % 11) / 4 - Float64(label)
        X[Item(i, 2)] = 100.0 + Float64((i * 3) % 7) * 10
        y.append(Float64(label * 2 - 1))
    return (X^, y^)


def take(y: Labels, rows: IndexArray) raises -> Labels:
    var out = Labels()
    for i in range(rows.size):
        out.append(y[Int(rows.item(i))])
    return out^


def stratified(y: Labels) raises -> List[IndexSplit]:
    var codes = IndexArray(Shape(len(y)))
    for i in range(len(y)):
        codes[Item(i)] = Int64(y[i])
    var cursor = StratifiedKFold(3).split(len(y), codes^)
    var out = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        out.append(pair.value().copy())
    return out^


def assert_same(a: Labels, b: Labels) raises:
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def assert_same(a: Dense, b: Dense) raises:
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_equal(a.item(i), b.item(i))


struct Accuracy(SearchScorer):
    comptime Estimator = SVC[]
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def score(self, estimator: SVC[], X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct CSearch(SearchAdapter):
    """Tunes C; sklearn's GridSearchCV(SVC(), {"C": [...]})."""
    comptime Scorer = Accuracy
    comptime Value = Float64

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
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
            if key != "C":
                raise Error("ValueError: Invalid parameter " + key)
            config.C = params[key]
        _ = estimator.set_params(config)

    def default_scorer(self) raises -> Accuracy:
        return Accuracy()

    def default_cv(self, X: Dense, y: Labels) raises -> List[IndexSplit]:
        return stratified(y)


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var y = data[1].copy()

    # clone copies configuration, not the fitted members; refitting reproduces them.
    var model = SVC(C=3.0, kernel="poly", degree=2, gamma="auto", class_weight="balanced")
    _ = model.fit(X, y)
    var copied = clone(model)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_equal(copied.get_params().C, 3.0)
    assert_equal(copied.get_params().kernel, "poly")
    _ = copied.fit(X, y)
    assert_same(model.predict(X), copied.predict(X))
    assert_same(model.dual_coef_, copied.dual_coef_)

    # Pipeline: scaling then SVC equals the manual composition.
    var pipe = make_pipeline(StandardScaler(), SVC(C=2.0))
    _ = pipe.fit(X, y)
    var scaler = StandardScaler()
    _ = scaler.fit(X)
    var Xs = scaler.transform(X)
    var manual = SVC(C=2.0)
    _ = manual.fit(Xs, y)
    assert_same(pipe.predict(X), manual.predict(Xs))
    assert_same(pipe.decision_function(X), manual.decision_function(Xs))
    assert_equal(pipe.score(X, y), manual.score(Xs, y))

    # A regression SVR in a pipeline as well.
    var target = Dense(Shape(60))
    for i in range(60):
        target[Item(i)] = X.item(i, 0) * 2 - X.item(i, 1)
    var rpipe = make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05))
    _ = rpipe.fit(X, target)
    var rmanual = SVR(C=10.0, epsilon=0.05)
    _ = rmanual.fit(Xs, target)
    assert_same(rpipe.predict(X), rmanual.predict(Xs))

    # cross_val_score clones the SVC per stratified fold; scores equal manual fits.
    var splits = stratified(y)
    var template = SVC(C=0.5, kernel="linear")
    var scores = cross_val_score_svc(template, X, y, splits)
    assert_equal(scores.size, 3)
    for k in range(3):
        var fold = clone(template)
        _ = fold.fit(_rows(X, splits[k].train), take(y, splits[k].train))
        assert_equal(scores.item(k), fold.score(_rows(X, splits[k].test), take(y, splits[k].test)))

    # GridSearchCV over C refits the best configuration on all rows.
    var cs: List[Float64] = [0.01, 1.0, 100.0]
    var grid = Dict[String, List[Float64]]()
    grid["C"] = cs.copy()
    var search = GridSearchCV[CSearch](SVC(), grid, CSearch(), cv=stratified(y))
    search.fit(X, y, None, None)
    var means = search.cv_results_().mean_test_score.copy()
    var best = 0
    for i in range(3):
        var by_hand = cross_val_score_svc(SVC(C=cs[i]), X, y, splits)
        var mean = (by_hand.item(0) + by_hand.item(1) + by_hand.item(2)) / 3
        assert_equal(means.item(i), mean)
        if mean > means.item(best):
            best = i
    assert_equal(search.best_params_()["C"], cs[best])
    var refit = SVC(C=cs[best])
    _ = refit.fit(X, y)
    assert_same(search.best_estimator_().predict(X), refit.predict(X))
    print("PASS: SVC/SVR clone, Pipeline, cross_val_score and GridSearchCV composition")


def cross_val_score_svc(template: SVC[], X: Dense, y: Labels, splits: List[IndexSplit]) raises -> Dense:
    return cross_val_score[CSearch](template, X, y, CSearch(), Accuracy(), splits, None, None)
