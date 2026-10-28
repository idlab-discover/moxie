"""A Pipeline as the estimator of cross_val_score and GridSearchCV.

sklearn users cross-validate and tune whole pipelines
(compose/plot_digits_pipe.py, model_selection/plot_grid_search_*.py). Each
composed result is checked against the same steps fitted by hand, and the
fixture's scores against recorded sklearn reference values.
SVC fitting is deterministic, so all comparisons are exact.
"""
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true
from moxie.base import clone
from moxie.decomposition import PCA
from moxie.svm import SVC
from moxie.preprocessing import StandardScaler
from moxie.pipeline import Pipeline
from moxie.model_selection import (
    GridSearchCV, IndexArray, IndexSplit, KFold, SearchAdapter, SearchScorer,
    cross_val_score,
)
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float64]
comptime Labels = List[Float64]
comptime ScaledSVC = Pipeline[StandardScaler[], SVC[]]
comptime ReducedSVC = Pipeline[PCA[], SVC[]]


def dataset() raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(30, 3))
    var y = Labels()
    for i in range(30):
        X[Item(i, 0)] = Float64(i % 3) + Float64(i) / 30
        X[Item(i, 1)] = Float64((i * 7) % 5)
        X[Item(i, 2)] = 10.0 * Float64((i * 11) % 4) - Float64(i % 3)
        y.append(Float64(i % 3))
    return (X^, y^)


def folds() raises -> List[IndexSplit]:
    var out = List[IndexSplit]()
    var cursor = KFold(3).split(30)
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


struct ScaledAccuracy(SearchScorer):
    comptime Estimator = ScaledSVC
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def score(self, estimator: ScaledSVC, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct ScaledSearch(SearchAdapter):
    """Mirror sklearn GridSearchCV(make_pipeline(StandardScaler(), SVC()), {"svc__C": [...]})."""
    comptime Scorer = ScaledAccuracy
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

    def fit(self, mut estimator: ScaledSVC, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(X, y)

    def configure(self, mut estimator: ScaledSVC, params: Dict[String, Float64]) raises:
        var config = estimator.get_params()
        for key in params.keys():
            if key != "svc__C":
                raise Error("ValueError: Invalid parameter " + key)
            config.estimator.C = params[key]
        _ = estimator.set_params(config)

    def default_scorer(self) raises -> ScaledAccuracy:
        return ScaledAccuracy()

    def default_cv(self, X: Dense, y: Labels) raises -> List[IndexSplit]:
        return folds()


struct ReducedAccuracy(SearchScorer):
    comptime Estimator = ReducedSVC
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def score(self, estimator: ReducedSVC, X: Dense, y: Labels, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct ReducedSearch(SearchAdapter):
    """Mirror sklearn GridSearchCV(Pipeline([("pca", PCA()), ("svc", SVC())]),
    {"pca__n_components": [...], "svc__C": [...]}), as in plot_digits_pipe."""
    comptime Scorer = ReducedAccuracy
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

    def fit(self, mut estimator: ReducedSVC, X: Dense, y: Labels, metadata: NoneType) raises:
        _ = estimator.fit(X, y)

    def configure(self, mut estimator: ReducedSVC, params: Dict[String, Float64]) raises:
        var config = estimator.get_params()
        for key in params.keys():
            if key == "pca__n_components":
                config.transformer.n_components = Int(params[key])
            elif key == "svc__C":
                config.estimator.C = params[key]
            else:
                raise Error("ValueError: Invalid parameter " + key)
        _ = estimator.set_params(config)

    def default_scorer(self) raises -> ReducedAccuracy:
        return ReducedAccuracy()

    def default_cv(self, X: Dense, y: Labels) raises -> List[IndexSplit]:
        return folds()


def scaled() raises -> ScaledSVC:
    return Pipeline(StandardScaler(), SVC(), transformer_name="standardscaler", estimator_name="svc")


def by_hand(C: Float64, X: Dense, y: Labels, splits: List[IndexSplit]) raises -> List[Float64]:
    var out = List[Float64]()
    for k in range(len(splits)):
        var scaler = StandardScaler()
        _ = scaler.fit(_rows(X, splits[k].train))
        var model = SVC(C=C)
        _ = model.fit(scaler.transform(_rows(X, splits[k].train)), take(y, splits[k].train))
        out.append(model.score(scaler.transform(_rows(X, splits[k].test)), take(y, splits[k].test)))
    return out^


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var y = data[1].copy()
    var splits = folds()

    # cross_val_score clones the unfitted pipeline per fold; scores equal manual fits.
    var template = scaled()
    var scores = cross_val_score[ScaledSearch](template, X, y, ScaledSearch(), ScaledAccuracy(), splits, None, None)
    var manual = by_hand(1.0, X, y, splits)
    assert_equal(scores.size, 3)
    for k in range(3):
        assert_equal(scores.item(k), manual[k])
    assert_true(not template.__sklearn_is_fitted__())

    # GridSearchCV over a nested step parameter ("svc__C") refits the best pipeline.
    var cs: List[Float64] = [0.1, 1.0, 10.0]
    var grid = Dict[String, List[Float64]]()
    grid["svc__C"] = cs.copy()
    var search = GridSearchCV[ScaledSearch](scaled(), grid, ScaledSearch(), cv=folds())
    search.fit(X, y, None, None)
    var means = search.cv_results_().mean_test_score.copy()
    var best = 0
    for i in range(3):
        var fold_scores = by_hand(cs[i], X, y, splits)
        var mean = (fold_scores[0] + fold_scores[1] + fold_scores[2]) / 3
        assert_equal(means.item(i), mean)
        if mean > means.item(best):
            best = i
    assert_equal(search.best_params_()["svc__C"], cs[best])
    var refit = scaled()
    var config = refit.get_params()
    config.estimator.C = cs[best]
    _ = refit.set_params(config)
    _ = refit.fit(X, y)
    var a = search.best_estimator_().predict(X)
    var b = refit.predict(X)
    for i in range(len(a)):
        assert_equal(a[i], b[i])

    # Two-step grid over PCA components and SVC C, in sklearn's ParameterGrid order.
    var reduced_grid = Dict[String, List[Float64]]()
    reduced_grid["pca__n_components"] = [1.0, 2.0]
    reduced_grid["svc__C"] = [0.1, 10.0]
    var reduced = GridSearchCV[ReducedSearch](
        Pipeline(PCA(), SVC(), transformer_name="pca", estimator_name="svc"),
        reduced_grid, ReducedSearch(), cv=folds(),
    )
    reduced.fit(X, y, None, None)
    var reduced_means = reduced.cv_results_().mean_test_score.copy()
    # Pinned sklearn values for this fixture (fold accuracies are multiples of
    # 1/10, so means agree to the last bits of a three-term float average).
    var expected: List[Float64] = [1.0 / 3, 1.0 / 3, 1.0 / 3, 0.3]
    assert_equal(reduced_means.size, 4)
    for i in range(4):
        assert_true(abs(reduced_means.item(i) - expected[i]) < 1e-12)
    # Ties keep sklearn's first-ranked candidate.
    assert_equal(reduced.best_params_()["pca__n_components"], 1.0)
    assert_equal(reduced.best_params_()["svc__C"], 0.1)
    var sklearn_scaled: List[Float64] = [0.6, 1.0, 0.6]
    var sklearn_means: List[Float64] = [0.7 / 1.5, 2.2 / 3, 0.6]
    for k in range(3):
        assert_true(abs(scores.item(k) - sklearn_scaled[k]) < 1e-12)
        assert_true(abs(means.item(k) - sklearn_means[k]) < 1e-12)
    assert_equal(cs[best], 1.0)
    print("PASS: Pipeline under cross_val_score and GridSearchCV")
