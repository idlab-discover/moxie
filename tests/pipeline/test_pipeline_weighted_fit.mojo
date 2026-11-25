from std.testing import assert_equal, assert_false, assert_raises
from numojo import NDArray, Shape
from moxie.base import WeightedSupervisedEstimator, clone
from moxie.dummy import DummyRegressor
from moxie.pipeline import Pipeline
from moxie.preprocessing import StandardScaler
from parity_io import ResultLine
from test_real_pipeline import CopyStep
from test_sequential import Recorder, Shift


def weighted_fit[E: WeightedSupervisedEstimator](
    mut estimator: E, X: E.Input, y: E.Target, weights: E.SampleWeight,
) raises:
    _ = estimator.fit(X, y, sample_weight=weights)


def check[dtype: DType]() raises:
    var X = NDArray[dtype](Shape(4, 1))
    var y = NDArray[dtype](Shape(4))
    var weights = NDArray[dtype](Shape(4))
    for i in range(4):
        X.itemset(i, Scalar[dtype](2 * i))
        y.itemset(i, Scalar[dtype](2 * i))
        weights.itemset(i, Scalar[dtype](5 if i == 3 else 1))

    var pipe = Pipeline(StandardScaler[dtype](), DummyRegressor[dtype](),
                        transformer_name="scale", estimator_name="model")
    var weighted = pipe.fit(X, y, sample_weight=weights).predict(X)
    assert_equal(weighted.ndim, 1)
    assert_equal(weighted.shape[0], 4)
    var prefix = pipe.transformer.transform(X)
    assert_equal(Float64(weights.item(3)), 5.0)
    assert_equal(Float64(y.item(3)), 6.0)
    assert_equal(Float64(X.item(3)), 6.0)

    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    var short = NDArray[dtype](Shape(1))
    short.itemset(0, 1)
    with assert_raises():
        _ = fresh.fit(X, y, sample_weight=short)
    assert_false(fresh.__sklearn_is_fitted__())
    var named = pipe.fit(X, y, fit_params={"model__sample_weight": weights.copy()}).predict(X)
    _ = pipe.fit(X, y, fit_params={})
    var unweighted = pipe.predict(X)

    var nested = Pipeline(CopyStep[dtype](), fresh^, estimator_name="tail")
    weights.itemset(0, 5)
    weights.itemset(3, 1)
    weighted_fit(nested, X, y, weights)
    assert_equal(nested.transformer.target_rows, 4)
    var nested_prediction = nested.predict(X)
    var nested_params = nested.get_params()
    nested_params.estimator.transformer_name = "scaler"
    nested_params.estimator.estimator_name = "regressor"
    _ = nested.set_params(nested_params)
    with assert_raises(contains="expects only 'tail__regressor__sample_weight'"):
        _ = nested.fit(X, y, fit_params={"tail__model__sample_weight": weights.copy()})
    var renamed = nested.fit(
        X, y, fit_params={"tail__regressor__sample_weight": weights.copy()},
    ).predict(X)
    with assert_raises():
        _ = nested.fit(X, y, fit_params={"tail__regressor__sample_weight": short^})

    for i in range(4):
        assert_equal(weighted.item(i), Scalar[dtype](4.5))
        assert_equal(named.item(i), weighted.item(i))
        assert_equal(unweighted.item(i), Scalar[dtype](3))
        assert_equal(nested_prediction.item(i), Scalar[dtype](1.5))
        assert_equal(renamed.item(i), nested_prediction.item(i))
    var line = ResultLine(String(dtype))
    line.add("weighted", named)
    line.add("unweighted", unweighted)
    line.add("nested", renamed)
    line.add("scaled", prefix)
    line.emit()


def check_invalid_routes() raises:
    var X = NDArray[DType.float64](Shape(4, 1))
    var y = NDArray[DType.float64](Shape(4))
    var weights = NDArray[DType.float64](Shape(4))
    var pipe = Pipeline(CopyStep[DType.float64](), DummyRegressor[DType.float64](),
                        transformer_name="copy", estimator_name="model")
    var nested = Pipeline(CopyStep[DType.float64](), pipe^, estimator_name="tail")

    with assert_raises(contains="expects only 'tail__model__sample_weight'"):
        _ = nested.fit(X, y, fit_params={"tail__sample_weight": weights.copy()})
    with assert_raises(contains="expects only 'tail__model__sample_weight'"):
        _ = nested.fit(X, y, fit_params={"sample_weight": weights.copy()})
    with assert_raises(contains="expects only 'tail__model__sample_weight'"):
        _ = nested.fit(X, y, fit_params={"tail__copy__sample_weight": weights.copy()})
    with assert_raises(contains="expects only 'tail__model__sample_weight'"):
        _ = nested.fit(X, y, fit_params={
            "tail__model__sample_weight": weights.copy(),
            "tail__model__unknown": weights.copy(),
        })
    assert_false(nested.transformer.__sklearn_is_fitted__())
    assert_false(nested.estimator.transformer.__sklearn_is_fitted__())
    assert_false(nested.__sklearn_is_fitted__())


def main() raises:
    comptime assert not conforms_to(Pipeline[Shift, Recorder], WeightedSupervisedEstimator)
    check[DType.float32]()
    check[DType.float64]()
    check_invalid_routes()
