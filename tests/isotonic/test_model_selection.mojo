"""Isotonic CV/search parity fixtures from the pinned sklearn isotonic module."""

from moxie.isotonic import IsotonicCVAdapter, IsotonicRegression, IsotonicSearchValue
from moxie.model_selection import GridSearchCV, cross_val_score
from numojo import NDArray, NDArrayStrides, Shape, array, zeros
from std.math import abs
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def assert_scores(actual: NDArray[DType.float64], expected: List[Float64], atol: Float64) raises:
    var scores = actual.tolist()
    assert_equal(len(scores), len(expected))
    for i in range(len(expected)):
        assert_true(abs(scores[i] - expected[i]) <= atol)


def test_search[dtype: DType, list_inputs: Bool](
    X: IsotonicCVAdapter[dtype, list_inputs].Scorer.Input,
    y: IsotonicCVAdapter[dtype, list_inputs].Scorer.Target,
    var fit_weights: IsotonicCVAdapter[dtype, list_inputs].Scorer.Metadata,
    var score_weights: IsotonicCVAdapter[dtype, list_inputs].Scorer.Metadata,
) raises:
    comptime Adapter = IsotonicCVAdapter[dtype, list_inputs]
    var model = IsotonicRegression[dtype](out_of_bounds="clip")
    var adapter = Adapter()
    var scorer = adapter.default_scorer()
    var folds = adapter.default_cv(X, y)
    assert_equal(len(folds), 5)
    var scores = cross_val_score[Adapter](
        model, X, y, adapter, scorer.copy(), folds, None, None,
    )
    var weighted = cross_val_score[Adapter](
        model, X, y, adapter, scorer^, folds, fit_weights, score_weights,
    )
    comptime if dtype == DType.float64:
        assert_scores(
            scores, [0, 0.20987654320987637, 0.20987654320987692, 0.20987654320987692, 0], 2e-12,
        )
        assert_scores(
            weighted,
            [-0.8, -0.053497942386831365, 0.08578664134219727,
             -0.2345679012345674, -0.7401315789473684],
            2e-12,
        )
    else:
        assert_scores(
            scores, [0, 0.20987647771835327, 0.2098766565322876, 0.2098766565322876, 0], 2e-6,
        )
        assert_scores(
            weighted,
            [-0.8000000715255737, -0.053498029708862305, 0.08578681945800781,
             -0.23456799983978271, -0.7401316165924072],
            2e-6,
        )
    assert_false(model.__sklearn_is_fitted__())

    var grid = Dict[String, List[IsotonicSearchValue]]()
    grid["y_max"] = [
        IsotonicSearchValue(4.0),
        IsotonicSearchValue(8.0),
        IsotonicSearchValue(NoneType()),
    ]
    var search = GridSearchCV[Adapter](model, grid, adapter)
    _ = search.fit(X, y)
    var results = search.cv_results_()
    ref means = results.mean_test_score
    assert_equal(means.shape[0], 3)
    assert_equal(means.shape[1], 1)
    comptime if dtype == DType.float64:
        assert_scores(means, [-3.7358024691358027, 0.12592592592592605, 0.12592592592592605], 2e-12)
    else:
        assert_scores(means, [-3.735802471637726, 0.1259259581565857, 0.1259259581565857], 2e-6)
    assert_equal(search.best_index_(), 1)
    var best = search.best_params_()
    assert_equal(best["y_max"][Float64], 8.0)
    var expected: List[Scalar[dtype]] = [0, 1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5, 8]
    comptime if list_inputs:
        assert_equal(search.best_estimator_().predict(rebind[List[Scalar[dtype]]](X)), expected)
    else:
        assert_equal(search.predict(X).tolist(), expected)
        assert_equal(search.transform(X).tolist(), expected)
    assert_false(model.__sklearn_is_fitted__())


def test_validation() raises:
    var adapter = IsotonicCVAdapter()
    var X = zeros[DType.float64](Shape(5, 1))
    var y = zeros[DType.float64](Shape(5))
    with assert_raises(contains="positive sample count"):
        _ = adapter.default_cv(X, zeros[DType.float64](Shape(6)))
    with assert_raises(contains="1 feature"):
        _ = adapter.default_cv(zeros[DType.float64](Shape(5, 2)), y)
    with assert_raises(contains="y should be a 1d array"):
        _ = adapter.default_cv(X, X)
    var folds = adapter.default_cv(X, y)
    with assert_raises(contains="sample_weight should be a 1d array"):
        _ = adapter.metadata_rows(X.copy(), folds[0].train)
    with assert_raises(contains="sample index out of bounds"):
        _ = adapter.metadata_rows(zeros[DType.float64](Shape(1)), folds[0].train)

    var model = IsotonicRegression(y_min=0.0, y_max=1.0)
    var params = Dict[String, IsotonicSearchValue]()
    params["y_min"] = IsotonicSearchValue(NoneType())
    params["increasing"] = IsotonicSearchValue(False)
    params["out_of_bounds"] = IsotonicSearchValue(String("clip"))
    adapter.configure(model, params)
    assert_false(Bool(model.y_min))
    assert_true(model.y_max == Optional[Float64](1.0))
    assert_false(model.increasing[Bool])
    assert_equal(model.out_of_bounds, "clip")
    params["increasing"] = IsotonicSearchValue(String("auto"))
    adapter.configure(model, params)
    assert_equal(model.increasing[String], "auto")
    params["y_min"] = IsotonicSearchValue(1.0)
    params["y_max"] = IsotonicSearchValue(True)
    with assert_raises(contains="y_max requires Float64 or None"):
        adapter.configure(model, params)
    assert_false(Bool(model.y_min))
    params["y_max"] = IsotonicSearchValue(NoneType())
    params["unknown"] = IsotonicSearchValue(0.0)
    with assert_raises(contains="invalid isotonic parameter unknown"):
        adapter.configure(model, params)
    assert_true(model.y_max == Optional[Float64](1.0))


def main() raises:
    var reversed_x: List[Float64] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    var backing = array(reversed_x, [10])
    var X = backing.view_with_layout(Shape(10, 1), NDArrayStrides(-1, 1), 9)
    var y: List[Float64] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 9]
    var fit_weights: List[Float64] = [1, 2, 5, 3, 1, 4, 2, 4, 8, 6]
    var score_weights: List[Float64] = [5, 1, 9, 3, 6, 13, 8, 2, 4, 19]
    test_search[DType.float64, False](
        X, array(y, [10]), array(fit_weights, [10]), array(score_weights, [10]),
    )

    var xs: List[Float32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    var ys: List[Float32] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 9]
    var fw: List[Float32] = [1, 2, 5, 3, 1, 4, 2, 4, 8, 6]
    var sw: List[Float32] = [5, 1, 9, 3, 6, 13, 8, 2, 4, 19]
    test_search[DType.float32, True](xs, ys, fw^, sw^)
    test_validation()
    print("PASS: isotonic model selection against pinned sklearn")
