"""Fluent mutations preserve receiver identity and existing estimator semantics."""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import RegressorMixin, WeightedSupervisedEstimator, clone
from moxie.kernel_ridge import KernelRidge, ScalarOrVector
from callable_kernel import AffineKernel


def generic_chain[
    E: RegressorMixin,
    x_origin: ImmOrigin,
    y_origin: ImmOrigin,
    params_origin: ImmOrigin,
](
    mut model: E,
    ref[x_origin] X: E.Input,
    ref[y_origin] y: E.Target,
    ref[params_origin] params: E.Params,
) raises -> E.Prediction:
    return model.set_params(params).fit(X, y).predict(X)


def generic_weighted[
    E: RegressorMixin & WeightedSupervisedEstimator,
    x_origin: ImmOrigin,
    y_origin: ImmOrigin,
    weight_origin: ImmOrigin,
](
    mut model: E,
    ref[x_origin] X: E.Input,
    ref[y_origin] y: E.Target,
    ref[weight_origin] weights: E.SampleWeight,
) raises -> E.Prediction:
    return model.fit(X, y, sample_weight=weights).predict(X)


def exercise[dtype: DType]() raises:
    var X = nm.arange[dtype](6).reshape(nm.Shape(3, 2))
    var y = nm.arange[dtype](3)
    var model = KernelRidge[dtype]()
    var baseline = KernelRidge[dtype](alpha=2.0)
    _ = baseline.fit(X, y)
    var expected = baseline.predict(X)
    var actual = model.set_params(alpha=ScalarOrVector(2.0)).fit(X, y).predict(X)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.alpha[Float64], 2.0)
    for i in range(3):
        assert_equal(
            actual[nm.NDArray[dtype]].unsafe_get(i),
            expected[nm.NDArray[dtype]].unsafe_get(i),
        )

    # Mutating a bound result must update the original receiver.
    ref fitted = model.fit(X, y)
    _ = fitted.set_params(alpha=ScalarOrVector(3.0))
    assert_equal(model.alpha[Float64], 3.0)
    var params = baseline.get_params()
    actual = generic_chain(model, X, y, params)
    assert_equal(model.alpha[Float64], 2.0)
    for i in range(3):
        assert_equal(
            actual[nm.NDArray[dtype]].unsafe_get(i),
            expected[nm.NDArray[dtype]].unsafe_get(i),
        )
    var record = baseline.get_params_record()
    actual = model.set_params(record).fit(X, y).predict(X)
    var weights = ScalarOrVector(1.0)
    actual = generic_weighted(model, X, y, weights)
    for i in range(3):
        assert_equal(
            actual[nm.NDArray[dtype]].unsafe_get(i),
            expected[nm.NDArray[dtype]].unsafe_get(i),
        )

    var scalar_weight = Float64(1.0)
    actual = model.fit(X, y, sample_weight=scalar_weight).predict(X)
    for i in range(3):
        assert_equal(
            actual[nm.NDArray[dtype]].unsafe_get(i),
            expected[nm.NDArray[dtype]].unsafe_get(i),
        )
    var list_weights: List[Float64] = [1.0, 1.0, 1.0]
    actual = model.fit(X, y, sample_weight=list_weights).predict(X)
    var wrapped = model.fit(
        X, y, sample_weight=ScalarOrVector(list_weights.copy())
    ).predict(X)
    for i in range(3):
        assert_equal(
            actual[nm.NDArray[DType.float64]].unsafe_get(i),
            wrapped[nm.NDArray[DType.float64]].unsafe_get(i),
        )

    var names: List[String] = ["a", "b"]
    actual = model.fit(X, y, feature_names=names).predict(X)
    assert_equal(model.feature_names_in_.value()[0], "a")
    actual = model.fit(
        X, y, feature_names=names, sample_weight=weights.copy()
    ).predict(X)
    # Exercise generic dtype overloads on a differently typed input.
    var X64 = X.astype[DType.float64]()
    var y64 = y.astype[DType.float64]()
    actual = model.fit(X64, y64).predict(X64)
    actual = model.fit(X64, y64, sample_weight=weights).predict(X64)
    assert_true(model.X_fit_.isa[nm.NDArray[DType.float64]]())

    _ = model.set_fit_request(sample_weight=True).set_score_request(
        sample_weight=False
    )
    assert_true(model.get_metadata_routing().fit.sample_weight.value()[Bool])
    assert_false(model.get_metadata_routing().score.sample_weight.value()[Bool])
    _ = model.set_fit_request(
        sample_weight=String("fit_weights")
    ).set_score_request(
        sample_weight=String("score_weights")
    ).set_fit_request().set_score_request()
    assert_equal(
        model.get_metadata_routing().fit.sample_weight.value()[String],
        "fit_weights",
    )
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_true(model.__sklearn_is_fitted__())

    var callable_model = KernelRidge[dtype, AffineKernel]()
    var callable_prediction = callable_model.set_params(
        kernel=AffineKernel(2.0)
    ).fit(X, y).predict(X)
    assert_true(callable_model.__sklearn_is_fitted__())
    assert_equal(callable_model.native_kernel.value().offset, 2.0)
    var callable_expected = callable_model.predict(X)
    for i in range(3):
        assert_equal(
            callable_prediction[nm.NDArray[dtype]].unsafe_get(i),
            callable_expected[nm.NDArray[dtype]].unsafe_get(i),
        )

    # A failed fit returns no borrow; preceding parameter updates remain applied.
    var bad = nm.zeros[DType.float64](nm.Shape(2))
    with assert_raises():
        _ = model.set_params(alpha=ScalarOrVector(4.0)).fit(X64, bad).predict(X64)
    assert_equal(model.alpha[Float64], 4.0)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.X_fit_[nm.NDArray[DType.float64]].shape[0], 3)


def main() raises:
    exercise[DType.float32]()
    exercise[DType.float64]()
    print("Fluent KernelRidge contracts passed")
