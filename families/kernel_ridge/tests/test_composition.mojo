"""KernelRidge inside clone, Pipeline, cross_val_score and GridSearchCV.

Each composed result is checked against the same steps fitted by hand, so the
composition adds no behavior of its own (sklearn's Pipeline/CV/search contract).
KernelRidge is a deterministic closed-form solve, so the comparisons are exact.
The grid mirrors plot_kernel_ridge_regression.py: GridSearchCV(KernelRidge(
kernel="rbf", gamma=0.1), {"alpha": [...], "gamma": logspace(-2, 2, 5)}) with
sklearn's default five unshuffled folds and R2 scoring.
"""
from numojo import NDArray, Shape, Item
from std.math import sin
from std.sys import argv
from std.testing import assert_equal, assert_raises, assert_true
from moxie.base import clone
from moxie.kernel_ridge import KernelRidge, ScalarOrVector
from moxie.preprocessing import StandardScaler
from moxie.pipeline import Pipeline, make_pipeline
from moxie.model_selection import (
    EstimatorScore, GridSearchCV, IndexArray, IndexSplit, KFold, SearchAdapter,
    cross_val_score, select_rows,
)

comptime Dense = NDArray[DType.float64]
comptime ScaledKRR = Pipeline[StandardScaler[], KernelRidge[]]
comptime R2 = EstimatorScore[KernelRidge[], Dense, Dense, ScalarOrVector]
comptime PipelineR2 = EstimatorScore[ScaledKRR, Dense, Dense, ScalarOrVector]
comptime Weights = Optional[ScalarOrVector]


def dataset() raises -> Tuple[Dense, Dense]:
    var X = Dense(Shape(40, 1))
    var y = Dense(Shape(40))
    for i in range(40):
        var x = Float64((i * 37) % 40) / 8.0
        X[Item(i, 0)] = x
        var noise = Float64(0)
        if i % 5 == 0:
            noise = Float64((i * 13) % 7) / 3.0 - 1.0
        y[Item(i)] = sin(x) + noise
    return (X^, y^)


def kfold(n: Int) raises -> List[IndexSplit]:
    var cursor = KFold(5).split(n)
    var out = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        out.append(pair.value().copy())
    return out^


def values(prediction: KernelRidge[].Prediction) raises -> Dense:
    return prediction[Dense].copy()


def assert_same(a: Dense, b: Dense) raises:
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_equal(a.item(i), b.item(i))


struct AlphaGammaSearch(SearchAdapter):
    """Tunes alpha and gamma, as the gallery's GridSearchCV(KernelRidge(...))."""
    comptime Scorer = R2
    comptime Value = Float64

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return y.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return select_rows(X, rows)

    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return select_rows(y, rows)

    def metadata_rows(self, metadata: Weights, rows: IndexArray) raises -> Weights:
        if metadata:
            raise Error("ValueError: this test adapter uses unweighted folds")
        return None

    def fit(self, mut estimator: KernelRidge[], X: Dense, y: Dense, metadata: Weights) raises:
        _ = estimator.fit(X, y)

    def configure(self, mut estimator: KernelRidge[], params: Dict[String, Float64]) raises:
        var mapping = KernelRidge[].ParameterMap()
        for key in params.keys():
            mapping[key] = KernelRidge[].ParameterValue(params[key])
        _ = estimator.set_params(mapping)

    def default_scorer(self) raises -> R2:
        return R2()

    def default_cv(self, X: Dense, y: Dense) raises -> List[IndexSplit]:
        return kfold(y.shape[0])


struct PipelineSearch(SearchAdapter):
    comptime Scorer = PipelineR2
    comptime Value = Float64

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return y.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return select_rows(X, rows)

    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return select_rows(y, rows)

    def metadata_rows(self, metadata: Weights, rows: IndexArray) raises -> Weights:
        if metadata:
            raise Error("ValueError: this test adapter uses unweighted folds")
        return None

    def fit(self, mut estimator: ScaledKRR, X: Dense, y: Dense, metadata: Weights) raises:
        _ = estimator.fit(X, y)

    def configure(self, mut estimator: ScaledKRR, params: Dict[String, Float64]) raises:
        var config = estimator.get_params()
        for key in params.keys():
            if key != "alpha" and key != "gamma":
                raise Error("ValueError: Invalid KernelRidge parameter " + key)
            config.estimator[key] = KernelRidge[].ParameterValue(params[key])
        _ = estimator.set_params(config)

    def default_scorer(self) raises -> PipelineR2:
        return PipelineR2()

    def default_cv(self, X: Dense, y: Dense) raises -> List[IndexSplit]:
        return kfold(y.shape[0])


def check_pipeline_search(X: Dense, y: Dense, emit_parity: Bool) raises:
    var template = make_pipeline(StandardScaler(), KernelRidge(kernel="rbf"))
    var grid = Dict[String, List[Float64]]()
    grid["alpha"] = [0.01, 0.1, 1.0]
    grid["gamma"] = [0.1, 1.0]
    var search = GridSearchCV[PipelineSearch](template, grid, PipelineSearch())
    _ = search.fit(X, y)
    assert_true(not template.estimator.__sklearn_is_fitted__())

    var bad = Dict[String, Float64]()
    bad["invalid"] = 0.0
    with assert_raises():
        PipelineSearch().configure(template, bad)

    ref results = search.cv_results_()
    assert_equal(len(results.params), 6)
    var splits = kfold(y.shape[0])
    var best = 0
    for candidate in range(len(results.params)):
        var params = results.params[candidate].copy()
        var alpha = params["alpha"]
        var gamma = params["gamma"]
        if emit_parity:
            print("params", alpha, gamma)
        for fold in range(len(splits)):
            var train = splits[fold].train.copy()
            var test = splits[fold].test.copy()
            var X_train = select_rows(X, train)
            var X_test = select_rows(X, test)
            var scaler = StandardScaler()
            _ = scaler.fit(X_train)
            var model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            _ = model.fit(scaler.transform(X_train), select_rows(y, train))
            var score = results.scores.test_scores.item(candidate, fold, 0)
            assert_equal(score, model.score(scaler.transform(X_test), select_rows(y, test)))
            if emit_parity:
                print("fold", score)
        var mean = results.mean_test_score.item(candidate, 0)
        if mean > results.mean_test_score.item(best, 0):
            best = candidate
        if emit_parity:
            print("mean", mean)
            print("rank", results.rank_test_score.item(candidate, 0))

    assert_equal(search.best_index_(), best)
    var refit = clone(template)
    PipelineSearch().configure(refit, results.params[best])
    _ = refit.fit(X, y)
    var prediction = values(search.predict(X))
    assert_same(prediction, values(refit.predict(X)))
    if emit_parity:
        print("best", search.best_index_())
        for i in range(prediction.size):
            print("prediction", prediction.item(i))


def cv_scores(template: KernelRidge[], X: Dense, y: Dense, splits: List[IndexSplit]) raises -> Dense:
    return cross_val_score[AlphaGammaSearch](template, X, y, AlphaGammaSearch(), R2(), splits, None, None)


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var y = data[1].copy()

    # clone copies configuration, not learned state; refitting reproduces it.
    var model = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5)
    _ = model.fit(X, y)
    var copied = clone(model)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_equal(copied.get_params()["alpha"][Float64], 0.1)
    assert_equal(copied.get_params()["kernel"][String], "rbf")
    assert_equal(copied.get_params()["gamma"][Optional[Float64]].value(), 0.5)
    _ = copied.fit(X, y)
    assert_same(values(model.predict(X)), values(copied.predict(X)))
    assert_same(model.dual_coef_[Dense], copied.dual_coef_[Dense])

    # Pipeline: scaling then KernelRidge equals the manual composition.
    var pipe = make_pipeline(StandardScaler(), KernelRidge(alpha=0.01, kernel="laplacian", gamma=2.0))
    _ = pipe.fit(X, y)
    var scaler = StandardScaler()
    _ = scaler.fit(X)
    var Xs = scaler.transform(X)
    var manual = KernelRidge(alpha=0.01, kernel="laplacian", gamma=2.0)
    _ = manual.fit(Xs, y)
    assert_same(values(pipe.predict(X)), values(manual.predict(Xs)))
    assert_equal(pipe.score(X, y), manual.score(Xs, y))

    # cross_val_score clones the template per fold; scores equal manual fits.
    var splits = kfold(40)
    var template = KernelRidge(alpha=0.1, kernel="rbf", gamma=1.0)
    var scores = cv_scores(template, X, y, splits)
    assert_equal(scores.size, 5)
    for k in range(5):
        var fold = clone(template)
        _ = fold.fit(select_rows(X, splits[k].train), select_rows(y, splits[k].train))
        assert_equal(scores.item(k), fold.score(select_rows(X, splits[k].test), select_rows(y, splits[k].test)))

    # GridSearchCV over alpha x gamma refits the best configuration on all rows.
    var alphas: List[Float64] = [1.0, 0.1, 0.01, 0.001]
    var gammas: List[Float64] = [0.01, 0.1, 1.0, 10.0, 100.0]
    var grid = Dict[String, List[Float64]]()
    grid["alpha"] = alphas.copy()
    grid["gamma"] = gammas.copy()
    var search = GridSearchCV[AlphaGammaSearch](
        KernelRidge(kernel="rbf", gamma=0.1), grid, AlphaGammaSearch()
    )
    _ = search.fit(X, y)
    var results = search.cv_results_().copy()
    assert_equal(len(results.params), 20)
    var best = 0
    for i in range(20):
        var a = results.params[i]["alpha"]
        var g = results.params[i]["gamma"]
        var by_hand = cv_scores(KernelRidge(alpha=a, kernel="rbf", gamma=g), X, y, splits)
        var mean = (by_hand.item(0) + by_hand.item(1) + by_hand.item(2) + by_hand.item(3) + by_hand.item(4)) / 5
        assert_equal(results.mean_test_score.item(i), mean)
        if mean > results.mean_test_score.item(best):
            best = i
    # sklearn enumerates ParameterGrid keys in sorted order with the last key fastest.
    assert_equal(results.params[1]["alpha"], 1.0)
    assert_equal(results.params[1]["gamma"], 0.1)
    assert_equal(search.best_params_()["alpha"], results.params[best]["alpha"])
    assert_equal(search.best_params_()["gamma"], results.params[best]["gamma"])
    var refit = KernelRidge(alpha=results.params[best]["alpha"], kernel="rbf", gamma=results.params[best]["gamma"])
    _ = refit.fit(X, y)
    assert_same(values(search.best_estimator_().predict(X)), values(refit.predict(X)))
    check_pipeline_search(X, y, len(argv()) > 1)
    print("PASS: KernelRidge clone, Pipeline, cross_val_score and GridSearchCV composition")
