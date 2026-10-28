"""Multiclass wrappers around RidgeClassifier inside clone, Pipeline, CV and search."""
import numojo as nm
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true
from std.utils import Variant
from moxie.base import ClassifierMixin, Predictor, clone
from moxie.linear_model import RidgeClassifier
from moxie.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from moxie.preprocessing import StandardScaler
from moxie.pipeline import make_pipeline
from moxie.model_selection import (
    GridSearchCV, IndexArray, IndexSplit, SearchAdapter, SearchScorer, StratifiedKFold,
    cross_val_score,
)
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._validation import CVAdapter
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float64]
comptime Labels = NDArray[DType.int64]
comptime OvR = OneVsRestClassifier[RidgeClassifier[]]
comptime OvO = OneVsOneClassifier[RidgeClassifier[]]


def dataset() raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(45, 3))
    var y = Labels(Shape(45))
    for i in range(45):
        var label = i % 3
        X[Item(i, 0)] = Float64(label) * 1.5 + Float64((i * 7) % 5) / 3
        X[Item(i, 1)] = Float64((i * 5) % 11) / 4 - Float64(label)
        X[Item(i, 2)] = 100.0 + Float64((i * 3) % 7) * 10
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


def assert_same(a: Dense, b: Dense) raises:
    assert_equal(a.shape[0], b.shape[0])
    assert_equal(a.shape[1], b.shape[1])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            assert_equal(a.item(i, j), b.item(i, j))


struct Accuracy[E: ClassifierMixin & Predictor & Deinitable](SearchScorer):
    comptime Estimator = Self.E
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def score(self, estimator: Self.E, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return Acc[Self.E]().score(estimator, X, y, None)


struct Folds[E: ClassifierMixin & Predictor & Deinitable](SearchAdapter):
    """Tunes OvO(RidgeClassifier) `estimator__alpha`."""
    comptime Scorer = Accuracy[Self.E]
    comptime Value = Float64

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

    def fit(self, mut estimator: Self.E, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(rebind[Self.E.Input](X), rebind[Self.E.Target](y))

    def configure(self, mut estimator: Self.E, params: Dict[String, Float64]) raises:
        # sklearn's nested "estimator__alpha" through the typed deep Params record.
        var config = rebind_var[OvO.Params](estimator.get_params(deep=True))
        for key in params.keys():
            if key != "estimator__alpha":
                raise Error("ValueError: Invalid parameter " + key)
            config.estimator.alpha = Variant[Float64, List[Float64]](params[key])
        _ = estimator.set_params(rebind[Self.E.Params](config))

    def default_scorer(self) raises -> Accuracy[Self.E]:
        return Accuracy[Self.E]()

    def default_cv(self, X: Dense, y: Labels) raises -> List[IndexSplit]:
        return stratified(y)


# Adapters generic over the estimator serve every wrapper; concrete adapter
# types are covered by test_concrete_adapters.mojo.
struct Acc[E: ClassifierMixin & Predictor & Deinitable](TypedScorer):
    comptime Estimator = Self.E
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType
    def __init__(out self):
        pass
    def score(self, estimator: Self.E, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        var predicted = rebind_var[Labels](estimator.predict(rebind[Self.E.Input](X)))
        var hits = 0
        for i in range(y.size):
            if predicted.unsafe_get(i) == y.unsafe_get(i):
                hits += 1
        return Float64(hits) / Float64(y.size)

struct Rows[E: ClassifierMixin & Predictor & Deinitable](CVAdapter):
    comptime Scorer = Acc[Self.E]
    def __init__(out self):
        pass
    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        return y.size
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return _rows(y, rows)
    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None
    def fit(self, mut estimator: Self.E, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(rebind[Self.E.Input](X), rebind[Self.E.Target](y))

def cv_scores[E: ClassifierMixin & Predictor & Deinitable](estimator: E, X: Dense, y: Labels, splits: List[IndexSplit]) raises -> Dense:
    return cross_val_score[Rows[E]](estimator, X, y, Rows[E](), Acc[E](), splits, None, None)


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var y = data[1].copy()

    # clone keeps nested configuration and drops fitted state.
    var model = OvR(RidgeClassifier(0.5))
    _ = model.fit(X, y)
    var copied = clone(model)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(copied.get_params().estimator.alpha[Float64], 0.5)
    _ = copied.fit(X, y)
    assert_same(model.decision_function(X), copied.decision_function(X))
    var code = OutputCodeClassifier(RidgeClassifier(), code_size=2.0, random_state=4)
    _ = code.fit(X, y)
    var code_copy = clone(code)
    assert_true(not code_copy.__sklearn_is_fitted__())
    _ = code_copy.fit(X, y)
    assert_same(code.code_book_, code_copy.code_book_)

    # Pipeline final step: scaling then OvR equals the manual composition.
    var pipe = make_pipeline(StandardScaler(), OvR(RidgeClassifier()))
    _ = pipe.fit(X, y)
    var scaler = StandardScaler()
    _ = scaler.fit(X)
    var Xs = scaler.transform(X)
    var manual = OvR(RidgeClassifier())
    _ = manual.fit(Xs, y)
    assert_same(pipe.decision_function(X), manual.decision_function(Xs))
    assert_equal(pipe.score(X, y), manual.score(Xs, y))

    # cross_val_score clones OvO per stratified fold; scores equal manual fits.
    var splits = stratified(y)
    var template = OvO(RidgeClassifier(2.0))
    var scores = cv_scores(template, X, y, splits)
    assert_equal(scores.size, 3)
    for k in range(3):
        var fold = clone(template)
        _ = fold.fit(_rows(X, splits[k].train), _rows(y, splits[k].train))
        assert_equal(scores.item(k), fold.score(_rows(X, splits[k].test), _rows(y, splits[k].test)))

    # GridSearchCV tunes the nested estimator__alpha and refits the best wrapper.
    var grid = Dict[String, List[Float64]]()
    grid["estimator__alpha"] = [0.01, 10.0, 1000.0]
    var search = GridSearchCV[Folds[OvO]](OvO(RidgeClassifier()), grid, Folds[OvO](), cv=stratified(y))
    search.fit(X, y, None, None)
    var means = search.cv_results_().mean_test_score.copy()
    var best = 0
    for i in range(3):
        var candidate = OvO(RidgeClassifier([0.01, 10.0, 1000.0][i]))
        var by_hand = cv_scores(candidate, X, y, splits)
        var mean = (by_hand.item(0) + by_hand.item(1) + by_hand.item(2)) / 3
        assert_equal(means.item(i), mean)
        if mean > means.item(best):
            best = i
    assert_equal(search.best_params_()["estimator__alpha"], [0.01, 10.0, 1000.0][best])
    var refit = OvO(RidgeClassifier([0.01, 10.0, 1000.0][best]))
    _ = refit.fit(X, y)
    assert_same(search.best_estimator_().decision_function(X), refit.decision_function(X))
    print("PASS: multiclass clone, Pipeline, cross_val_score and GridSearchCV composition")
