"""Multioutput wrappers around Ridge inside clone, Pipeline, cross_val_score and search."""
import numojo as nm
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true
from std.utils import Variant
from moxie.base import RegressorMixin, Predictor, clone
from moxie.linear_model import Ridge
from moxie.multioutput import MultiOutputRegressor, RegressorChain
from moxie.preprocessing import StandardScaler
from moxie.pipeline import make_pipeline
from moxie.model_selection import (
    GridSearchCV, IndexArray, IndexSplit, KFold, SearchAdapter, SearchScorer, cross_val_score,
)
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._validation import CVAdapter
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float64]
comptime MOR = MultiOutputRegressor[Ridge[]]
comptime Chain = RegressorChain[Ridge[]]


def dataset() raises -> Tuple[Dense, Dense]:
    var X = Dense(Shape(30, 3))
    var Y = Dense(Shape(30, 2))
    for i in range(30):
        X[Item(i, 0)] = Float64((i * 7) % 5) / 3 + Float64(i) / 10
        X[Item(i, 1)] = Float64((i * 5) % 11) / 4
        X[Item(i, 2)] = 100.0 + Float64((i * 3) % 7) * 10
        Y[Item(i, 0)] = 2 * X.item(i, 0) - X.item(i, 1) + Float64(i % 3) / 2
        Y[Item(i, 1)] = X.item(i, 1) + 0.05 * X.item(i, 2) - Float64(i % 4) / 3
    return (X^, Y^)


def folds(n: Int) raises -> List[IndexSplit]:
    var cursor = KFold(3).split(n)
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


# Adapters stay generic over the estimator: Mojo 1.0 crashes when a CV adapter
# names a concrete wrapper type (see tests/multiclass/test_composition.mojo).
struct R2[E: RegressorMixin & Predictor & Deinitable](TypedScorer):
    comptime Estimator = Self.E
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = NoneType
    def __init__(out self):
        pass
    def score(self, estimator: Self.E, X: Dense, y: Dense, metadata: NoneType) raises -> Float64:
        # Both wrappers weight scores with Float64 vectors; a named Optional keeps
        # the generic call from abandoning a non-Deinitable temporary.
        var unweighted: Optional[NDArray[DType.float64]] = None
        return estimator.score(rebind[Self.E.Input](X), rebind[Self.E.Target](y), rebind[Optional[Self.E.SampleWeight]](unweighted))


struct Rows[E: RegressorMixin & Predictor & Deinitable](CVAdapter):
    comptime Scorer = R2[Self.E]
    def __init__(out self):
        pass
    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return y.shape[0]
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)
    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None
    def fit(self, mut estimator: Self.E, X: Dense, y: Dense, metadata: NoneType) raises:
        _ = estimator.fit(rebind[Self.E.Input](X), rebind[Self.E.Target](y))


struct SearchR2[E: RegressorMixin & Predictor & Deinitable](SearchScorer):
    comptime Estimator = Self.E
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = NoneType
    def __init__(out self):
        pass
    def __init__(out self, *, copy: Self):
        pass
    def score(self, estimator: Self.E, X: Dense, y: Dense, metadata: NoneType) raises -> Float64:
        return R2[Self.E]().score(estimator, X, y, None)


struct Search[E: RegressorMixin & Predictor & Deinitable](SearchAdapter):
    """Tunes MultiOutputRegressor(Ridge) `estimator__alpha` through the typed record."""
    comptime Scorer = SearchR2[Self.E]
    comptime Value = Float64
    def __init__(out self):
        pass
    def __init__(out self, *, copy: Self):
        pass
    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return y.shape[0]
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)
    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None
    def fit(self, mut estimator: Self.E, X: Dense, y: Dense, metadata: NoneType) raises:
        _ = estimator.fit(rebind[Self.E.Input](X), rebind[Self.E.Target](y))
    def configure(self, mut estimator: Self.E, params: Dict[String, Float64]) raises:
        var config = rebind_var[MOR.Params](estimator.get_params(deep=True))
        for key in params.keys():
            if key != "estimator__alpha":
                raise Error("ValueError: Invalid parameter " + key)
            config.estimator.alpha = Variant[Float64, List[Float64]](params[key])
        _ = estimator.set_params(rebind[Self.E.Params](config))
    def default_scorer(self) raises -> SearchR2[Self.E]:
        return SearchR2[Self.E]()
    def default_cv(self, X: Dense, y: Dense) raises -> List[IndexSplit]:
        return folds(y.shape[0])


def cv_scores[E: RegressorMixin & Predictor & Deinitable](estimator: E, X: Dense, y: Dense, splits: List[IndexSplit]) raises -> Dense:
    return cross_val_score[Rows[E]](estimator, X, y, Rows[E](), R2[E](), splits, None, None)


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var Y = data[1].copy()

    # Pipeline final step: scaling then the wrapper equals the manual composition.
    var scaler = StandardScaler()
    _ = scaler.fit(X)
    var Xs = scaler.transform(X)
    var pipe = make_pipeline(StandardScaler(), MOR(Ridge(0.5)))
    _ = pipe.fit(X, Y)
    var manual = MOR(Ridge(0.5))
    _ = manual.fit(Xs, Y)
    assert_same(pipe.predict(X), manual.predict(Xs))
    assert_equal(pipe.score(X, Y), manual.score(Xs, Y))
    var chain_pipe = make_pipeline(StandardScaler(), Chain(Ridge(0.5), order=List[Int]([1, 0])))
    _ = chain_pipe.fit(X, Y)
    var chain = Chain(Ridge(0.5), order=List[Int]([1, 0]))
    _ = chain.fit(Xs, Y)
    assert_same(chain_pipe.predict(X), chain.predict(Xs))

    # cross_val_score clones the wrapper per fold; scores equal manual fits.
    var splits = folds(30)
    var template = MOR(Ridge(2.0))
    var scores = cv_scores(template, X, Y, splits)
    assert_equal(scores.size, 3)
    for k in range(3):
        var fold = clone(template)
        _ = fold.fit(_rows(X, splits[k].train), _rows(Y, splits[k].train))
        assert_equal(scores.item(k), fold.score(_rows(X, splits[k].test), _rows(Y, splits[k].test)))
    var chain_scores = cv_scores(Chain(Ridge(2.0), order=List[Int]([1, 0])), X, Y, splits)
    for k in range(3):
        var fold = Chain(Ridge(2.0), order=List[Int]([1, 0]))
        _ = fold.fit(_rows(X, splits[k].train), _rows(Y, splits[k].train))
        assert_equal(chain_scores.item(k), fold.score(_rows(X, splits[k].test), _rows(Y, splits[k].test)))

    # GridSearchCV tunes the nested estimator__alpha and refits the best wrapper.
    var grid = Dict[String, List[Float64]]()
    grid["estimator__alpha"] = [0.01, 10.0, 1000.0]
    var search = GridSearchCV[Search[MOR]](MOR(Ridge()), grid, Search[MOR](), cv=folds(30))
    search.fit(X, Y, None, None)
    var means = search.cv_results_().mean_test_score.copy()
    var best = 0
    for i in range(3):
        var by_hand = cv_scores(MOR(Ridge([0.01, 10.0, 1000.0][i])), X, Y, splits)
        var mean = (by_hand.item(0) + by_hand.item(1) + by_hand.item(2)) / 3
        assert_equal(means.item(i), mean)
        if mean > means.item(best):
            best = i
    assert_equal(search.best_params_()["estimator__alpha"], [0.01, 10.0, 1000.0][best])
    var refit = MOR(Ridge([0.01, 10.0, 1000.0][best]))
    _ = refit.fit(X, Y)
    assert_same(search.best_estimator_().predict(X), refit.predict(X))
    print("PASS: multioutput clone, Pipeline, cross_val_score and GridSearchCV composition")
