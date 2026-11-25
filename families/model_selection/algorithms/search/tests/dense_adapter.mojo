from moxie.base import clone
from moxie.dummy import DummyRegressor
from moxie.model_selection import DenseSupervisedAdapter, GridSearchCV, IndexArray, cross_val_score
from numojo import NDArray, Shape, Item, zeros
from std.testing import assert_equal, assert_false, assert_raises

comptime Dense = NDArray[DType.float32]
comptime Regression = DummyRegressor[DType.float32]
comptime Adapter = DenseSupervisedAdapter[Regression, Dense, DType.float32]


def main() raises:
    var X = zeros[DType.float32](Shape(12, 1))
    var y = Dense(Shape(12))
    var weights = Dense(Shape(12))
    for i in range(12):
        X[Item(i, 0)] = Float32(i)
        y[Item(i)] = Float32(i % 5)
        weights[Item(i)] = Float32(i + 1)

    var sample_weight = Optional[Dense](weights^)
    var model = Regression()
    var adapter = Adapter()
    var folds = adapter.default_cv(X, y)
    var scores = cross_val_score[Adapter](
        model, X, y, adapter, fit_metadata=sample_weight, score_metadata=sample_weight,
    )
    assert_equal(scores.size, 5)
    var supplied = adapter.default_cv(X, y)
    _ = supplied.pop()
    var explicit_scores = cross_val_score[Adapter](
        model, X, y, adapter, cv=supplied^, fit_metadata=sample_weight, score_metadata=sample_weight,
    )
    assert_equal(explicit_scores.size, 4)
    for fold in range(explicit_scores.size):
        assert_equal(scores[Item(fold)], explicit_scores[Item(fold)])
    for fold in range(len(folds)):
        print("score", fold, scores[Item(fold)])
        for row in range(folds[fold].test.size):
            print("index", fold, folds[fold].test.item(row))

    var grid = Dict[String, List[Float64]]()
    var search = GridSearchCV[Adapter](model, grid, adapter)
    _ = search.fit(X, y, sample_weight, sample_weight)
    assert_equal(search.n_splits_(), 5)
    var prediction = search.predict(X)
    print("prediction", prediction[Item(0)])
    assert_false(model.__sklearn_is_fitted__())
    var fresh = clone(search)
    assert_false(fresh.__sklearn_is_fitted__())

    var incompatible = Dense(Shape(13))
    with assert_raises(contains="one value per sample"):
        _ = cross_val_score[Adapter](
            model, X, y, adapter, fit_metadata=incompatible.copy(), score_metadata=None,
        )
    with assert_raises(contains="one value per sample"):
        _ = search.fit(X, y, None, incompatible.copy())
    assert_false(search.__sklearn_is_fitted__())
    with assert_raises(contains="inconsistent sample counts"):
        _ = adapter.default_cv(X, incompatible)
    var matrix_weights = Dense(Shape(12, 1))
    with assert_raises(contains="one value per sample"):
        _ = adapter.fit(model, X, y, matrix_weights^)
    var multioutput = zeros[DType.float32](Shape(12, 2))
    multioutput[Item(2, 1)] = 7
    multioutput[Item(8, 1)] = 9
    var rows = IndexArray(Shape(2))
    rows[Item(0)] = 8
    rows[Item(1)] = 2
    var selected = adapter.target_rows(multioutput, rows)
    assert_equal(selected.ndim, 2)
    assert_equal(selected.shape[0], 2)
    assert_equal(selected.shape[1], 2)
    assert_equal(selected[Item(0, 1)], Float32(9))
    assert_equal(selected[Item(1, 1)], Float32(7))

    grid["constant"] = [1.0]
    var unconfigured = GridSearchCV[Adapter](model, grid, adapter)
    with assert_raises(contains="nonempty parameters require an apply_params function"):
        _ = unconfigured.fit(X, y)
