from moxie.base import clone
from moxie.model_selection import DenseSupervisedAdapter, GridSearchCV, PrecisionMacro, cross_val_score
from moxie.svm import SVC
from moxie.pipeline import Pipeline
from moxie.preprocessing import StandardScaler
from numojo import Item, NDArray, Shape
from std.testing import assert_equal, assert_false, assert_raises

comptime Dense = NDArray[DType.float64]
comptime Model = Pipeline[StandardScaler[], SVC[]]
comptime Labels = List[Float64]
comptime Scorer = PrecisionMacro[Model, Dense, Labels]
comptime Adapter = DenseSupervisedAdapter[Model, Labels, S=Scorer]


def configure(mut model: Model, params: Dict[String, Float64]) raises:
    var config = model.get_params()
    config.estimator.C = params["C"]
    _ = model.set_params(config)


def main() raises:
    var X = Dense(Shape(30, 2))
    var y = Labels()
    var weights = Dense(Shape(30))
    for row in range(30):
        X[Item(row, 0)] = Float64(row % 3) + Float64(row) / 10.0
        X[Item(row, 1)] = Float64((row * 7) % 5)
        y.append(Float64(row % 3))
        weights[Item(row)] = Float64(row % 7 + 1)
    var model = Pipeline(StandardScaler(), SVC())
    var adapter = Adapter(configure, classifier=True, scoring=Scorer())
    var scores = cross_val_score[Adapter](
        model, X, y, adapter, fit_metadata=None, score_metadata=weights.copy(),
    )
    assert_equal(scores.size, 5)
    for fold in range(scores.size):
        print("weighted_cv", scores[Item(fold)])

    var grid: Dict[String, List[Float64]] = {"C": [0.1, 1.0, 10.0]}
    var search = GridSearchCV[Adapter](model, grid, adapter)
    with assert_raises(contains="NotFittedError"):
        _ = search.best_score_()
    _ = search.fit(X, y)
    assert_equal(search.n_splits_(), 5)
    var results = search.cv_results_()
    for candidate in range(len(results.params)):
        for fold in range(search.n_splits_()):
            print("search_fold", results.scores.test_scores[Item(candidate, fold, 0)])
    print("best_index", search.best_index_())
    print("best_score", search.best_score_())
    print("refit_score", search.score(X, y))
    assert_false(model.__sklearn_is_fitted__())
    assert_false(clone(search).__sklearn_is_fitted__())
