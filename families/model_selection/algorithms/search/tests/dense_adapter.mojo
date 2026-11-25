from moxie.base import clone
from moxie.dummy import DummyRegressor
from moxie.model_selection import (
    DenseSupervisedAdapter, GridSearchCV, IndexArray, IndexSplit, cross_val_score, cross_validate,
    learning_curve, validation_curve,
)
from numojo import NDArray, Shape, Item, zeros
from std.testing import assert_equal, assert_false, assert_raises

comptime Dense = NDArray[DType.float32]
comptime Regression = DummyRegressor[DType.float32]
comptime Adapter = DenseSupervisedAdapter[Regression, Dense, DType.float32]
comptime CurveAdapter = DenseSupervisedAdapter[Regression, Dense, DType.float32, String]


def apply_strategy(mut model: Regression, values: Dict[String, String]) raises:
    var params = model.get_params()
    params.strategy = values["strategy"].copy()
    _ = model.set_params(params)


def check_curves(model: Regression, X: Dense, y: Dense, weights: Optional[Dense]) raises:
    var adapter = CurveAdapter(apply_strategy)
    var strategies: List[String] = ["mean", "median"]
    var validation = validation_curve[CurveAdapter](
        model, X, y, adapter, param_name="strategy", param_range=strategies,
    )
    assert_equal(validation[0].shape[1], 5)
    assert_equal(validation[1].shape[1], 5)
    for parameter in range(len(strategies)):
        for fold in range(5):
            print("validation_curve", parameter, fold,
                  validation[0].item(parameter, fold), validation[1].item(parameter, fold))

    var learning = learning_curve[CurveAdapter](
        model, X, y, adapter, fit_metadata=weights, score_metadata=weights,
        return_times=True,
    )
    assert_equal(learning.train_scores.shape[1], 5)
    assert_equal(learning.test_scores.shape[1], 5)
    assert_equal(learning.fit_times.value().shape[1], 5)
    assert_equal(learning.score_times.value().shape[1], 5)
    for size in range(learning.train_sizes.size):
        print("learning_size", learning.train_sizes.item(size))
        for fold in range(5):
            print("learning_curve", size, fold,
                  learning.train_scores.item(size, fold), learning.test_scores.item(size, fold))

    var folds = adapter.default_cv(X, y)
    _ = folds.pop()
    var explicit_validation = validation_curve[CurveAdapter](
        model, X, y, adapter, param_name="strategy", param_range=strategies,
        scoring=adapter.default_scorer(), cv=folds^,
    )
    var learning_folds = adapter.default_cv(X, y)
    _ = learning_folds.pop()
    var explicit_learning = learning_curve[CurveAdapter](
        model, X, y, adapter, scoring=adapter.default_scorer(), cv=learning_folds^,
        fit_metadata=weights, score_metadata=weights,
    )
    var unweighted_learning = learning_curve[CurveAdapter](model, X, y, adapter)
    assert_equal(explicit_validation[1].shape[1], 4)
    assert_equal(explicit_learning.test_scores.shape[1], 4)
    assert_equal(unweighted_learning.test_scores.shape[1], 5)
    assert_equal(explicit_validation[1].item(0, 0), validation[1].item(0, 0))
    assert_equal(explicit_learning.test_scores.item(0, 0), learning.test_scores.item(0, 0))
    assert_equal(unweighted_learning.test_scores.item(unweighted_learning.train_sizes.size - 1, 0),
                 validation[1].item(0, 0))
    with assert_raises(contains="at least one split"):
        _ = validation_curve[CurveAdapter](
            model, X, y, adapter, param_name="strategy", param_range=strategies,
            cv=List[IndexSplit](),
        )
    with assert_raises(contains="at least one split"):
        _ = learning_curve[CurveAdapter](model, X, y, adapter, cv=List[IndexSplit]())
    with assert_raises(contains="requires an integer seed"):
        _ = learning_curve[CurveAdapter](model, X, y, adapter, shuffle=True)
    assert_false(model.__sklearn_is_fitted__())


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
    check_curves(model, X, y, sample_weight)
    var adapter = Adapter()
    var folds = adapter.default_cv(X, y)
    var scores = cross_val_score[Adapter](
        model, X, y, adapter, fit_metadata=sample_weight, score_metadata=sample_weight,
    )
    var scorers: List[Adapter.Scorer] = [adapter.default_scorer()]
    var metrics: List[String] = ["weighted_r2"]
    var validation = cross_validate[Adapter](
        model, X, y, adapter, scorers.copy(), metrics, fit_metadata=sample_weight,
        score_metadata=sample_weight, return_train_score=True, return_indices=True,
    )
    assert_equal(validation.scores.metric_names[0], "weighted_r2")
    assert_equal(scores.size, 5)
    var supplied = adapter.default_cv(X, y)
    _ = supplied.pop()
    var override_cv = Optional[List[IndexSplit]](supplied^)
    var explicit_scores = cross_val_score[Adapter](
        model, X, y, adapter, cv=override_cv, fit_metadata=sample_weight, score_metadata=sample_weight,
    )
    assert_equal(explicit_scores.size, 4)
    var explicit_validation = cross_validate[Adapter](
        model, X, y, adapter, scorers.copy(), metrics, cv=override_cv^,
    )
    assert_equal(len(explicit_validation.status), 4)
    for fold in range(explicit_scores.size):
        assert_equal(scores[Item(fold)], explicit_scores[Item(fold)])
    for fold in range(len(folds)):
        assert_equal(scores[Item(fold)], validation.scores.test_scores.item(0, fold, 0))
        print("train_score", fold, validation.scores.train_scores.value().item(0, fold, 0))
        print("score", fold, scores[Item(fold)])
        ref test = validation.indices.value()[fold].test
        assert_equal(test.size, folds[fold].test.size)
        for row in range(test.size):
            print("index", fold, test.item(row))

    with assert_raises(contains="at least one split"):
        _ = cross_validate[Adapter](model, X, y, adapter, cv=List[IndexSplit]())
    with assert_raises(contains="scorers and metric names"):
        _ = cross_validate[Adapter](model, X, y, adapter, scorers^, ["one", "two"])

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
        _ = cross_validate[Adapter](
            model, X, y, adapter, fit_metadata=None, score_metadata=incompatible.copy(),
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
