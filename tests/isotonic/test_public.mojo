"""Public exports and fluent contracts, also compiled outside the repository."""
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from moxie.base import RegressorMixin, WeightedSupervisedEstimator, clone
from moxie.isotonic import IsotonicRegression, isotonic_regression, check_increasing


def generic[E: RegressorMixin & WeightedSupervisedEstimator](
    mut model: E, params: E.Params, X: E.Input, y: E.Target, weights: E.SampleWeight,
) raises -> E.Prediction:
    return model.set_params(params).fit(X, y, sample_weight=weights).predict(X)


def exercise[dtype: DType]() raises:
    var x: List[Scalar[dtype]] = [0, 1, 2]
    var y: List[Scalar[dtype]] = [3, 1, 2]
    var weights: List[Scalar[dtype]] = [1, 1, 1]
    var model = IsotonicRegression[dtype]()
    var expected: List[Scalar[dtype]] = [2, 2, 2]
    var fitted: List[Scalar[dtype]] = isotonic_regression(y)
    assert_equal(fitted, expected)
    assert_true(check_increasing(x, x))
    var params = model.get_params()
    params.out_of_bounds = "clip"
    var predictions = generic(model, params, x, y, weights)
    assert_equal(predictions, expected)
    assert_equal(model.set_params(out_of_bounds=String("clip")).fit(x, y).predict(x), expected)
    var columns: List[List[Scalar[dtype]]] = [[0], [1], [2]]
    assert_equal(model.fit(columns, y, sample_weight=weights.copy()).predict(columns), expected)
    var copied = model.copy()
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.fit(x, List[Scalar[dtype]]())
    assert_equal(model.predict(x), expected)
    _ = model.fit(x, x)
    assert_equal(copied.predict(x), expected)
    assert_equal(model.score(x, x, sample_weight=weights.copy()), 1.0)
    var retained: List[Scalar[dtype]] = copied.X_thresholds_.copy()
    var endpoints: List[Scalar[dtype]] = [0, 2]
    assert_equal(retained, endpoints)


def main() raises:
    exercise[DType.float32]()
    exercise[DType.float64]()
    print("PASS: isotonic public exports, fluent contracts and dtype preservation")
