from moxie.base import clone
from moxie.dummy import DummyRegressor
from moxie.model_selection import (
    DenseSupervisedAdapter, GridSearchCV, NegMeanAbsoluteError,
    NegRootMeanSquaredError, cross_val_score,
)
from numojo import NDArray, Shape, Item, zeros
from std.testing import assert_equal, assert_false, assert_raises

comptime Dense = NDArray[DType.float64]
comptime Model = DummyRegressor[DType.float64]
comptime MAE = NegMeanAbsoluteError[Model, Dense]
comptime RMSE = NegRootMeanSquaredError[Model, Dense]
comptime MAEAdapter = DenseSupervisedAdapter[Model, V=String, S=MAE]
comptime RMSEAdapter = DenseSupervisedAdapter[Model, S=RMSE]


def apply_strategy(mut model: Model, values: Dict[String, String]) raises:
    var params = model.get_params()
    params.strategy = values["strategy"].copy()
    _ = model.set_params(params)


def main() raises:
    var X = zeros[DType.float64](Shape(12, 1))
    var y = Dense(Shape(12, 2))
    var weights = Dense(Shape(12))
    for i in range(12):
        y[Item(i, 0)] = Float64(i % 4)
        y[Item(i, 1)] = Float64(i * i % 17)
        weights[Item(i)] = Float64(i + 1)
    var sample_weight = Optional[Dense](weights^)
    var model = Model()

    var mae_adapter = MAEAdapter(apply_strategy, scoring=MAE())
    var mae_scores = cross_val_score[MAEAdapter](
        model, X, y, mae_adapter, score_metadata=sample_weight, fit_metadata=None,
    )
    var rmse_scores = cross_val_score[RMSEAdapter](model, X, y, RMSEAdapter())
    assert_equal(mae_scores.size, 5)
    assert_equal(rmse_scores.size, 5)
    for fold in range(5):
        print("mae_fold", fold, mae_scores[Item(fold)])
        print("rmse_fold", fold, rmse_scores[Item(fold)])

    var grid: Dict[String, List[String]] = {"strategy": ["mean", "median"]}
    var search = GridSearchCV[MAEAdapter](model, grid, mae_adapter)
    with assert_raises(contains="NotFittedError"):
        _ = search.best_score_()
    _ = search.fit(X, y)
    print("best_index", 0, search.best_index_())
    print("best_score", 0, search.best_score_())
    print("refit_score", 0, search.score(X, y))
    var results = search.cv_results_()
    for candidate in range(2):
        for fold in range(5):
            print("search_fold", candidate * 5 + fold, results.scores.test_scores[Item(candidate, fold, 0)])
    assert_false(model.__sklearn_is_fitted__())
    var fresh = clone(search)
    assert_false(fresh.__sklearn_is_fitted__())
    _ = search.fit(X, y)
    assert_equal(search.best_score_(), results.mean_test_score[Item(search.best_index_(), 0)])
