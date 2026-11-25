"""Cross-validation and grid search over a Ridge model with a transformed target."""

from moxie.compose import TransformedTargetRegressor
from moxie.linear_model import Ridge
from moxie.model_selection import (
    DenseSupervisedAdapter,
    ErrorPolicy,
    GridSearchCV,
    cross_val_score,
)
from numojo import Item, NDArray, Shape, array, sqrt, zeros
from std.testing import assert_equal, assert_false, assert_raises

from parity_io import ResultLine


comptime Matrix = NDArray[DType.float64]
comptime Model = TransformedTargetRegressor[DType.float64, Ridge[]]
comptime Adapter = DenseSupervisedAdapter[Model]


def root(y: Matrix) raises -> Matrix:
    return sqrt(y)


def square(y: Matrix) raises -> Matrix:
    return y * y


def apply_alpha(mut model: Model, params: Dict[String, Float64]) raises:
    var params_before = model.get_params()
    if not params_before.regressor:
        raise Error("Expected a Ridge regressor")
    var config = params_before.regressor.value().get_params()
    config.alpha = params["regressor__alpha"]
    _ = model.set_params(regressor_params=config^)


def main() raises:
    var X = array[DType.float64](
        [0, 2, 1, -1, 2, 3, 3, -2, 4, 2, 5, -3, 6, 1, 7, 0, 8, 4],
        [9, 2],
    )
    var y = array[DType.float64]([9, 16, 25, 49, 64, 81, 121, 144, 196], [9])
    var test = array[DType.float64]([1.5, 1, 4.5, -1, 8.5, 2], [3, 2])
    var model = Model(Ridge(), func=root, inverse_func=square)
    var adapter = Adapter(apply_alpha, n_splits=3)
    var raise_errors = ErrorPolicy(True, 0)
    with assert_raises(contains="NotFittedError"):
        _ = model.get_n_features_in()

    var result = ResultLine("ridge_target_search")
    result.add("cv", cross_val_score[Adapter](model, X, y, adapter, error_score=raise_errors))
    assert_false(model.__sklearn_is_fitted__())

    var grid: Dict[String, List[Float64]] = {"regressor__alpha": [25.0, 0.1, 1.0]}
    var search = GridSearchCV[Adapter](model, grid, adapter, error_score=raise_errors)
    _ = search.fit(X, y)
    assert_false(model.__sklearn_is_fitted__())
    assert_equal(search.n_features_in_(), 2)
    ref best = search.best_estimator_()
    assert_equal(
        best.regressor_.value().get_params().alpha, search.best_params_()["regressor__alpha"],
    )

    var results = search.cv_results_()
    var scores = zeros[DType.float64](Shape(3, 3))
    for candidate in range(3):
        for fold in range(3):
            scores[Item(candidate, fold)] = results.scores.test_scores.item(candidate, fold, 0)
    result.add("candidate_scores", scores)
    result.integer("best_index", search.best_index_())
    result.scalar("best_alpha", search.best_params_()["regressor__alpha"])
    result.scalar("best_score", search.best_score_())
    result.integer("n_features", search.n_features_in_())
    result.add("predict", search.predict(test))
    result.emit()
