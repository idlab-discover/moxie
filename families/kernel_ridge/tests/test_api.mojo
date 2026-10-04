from moxie.kernel_ridge import KernelParameters, KernelParameterValue
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.base import clone, BaseEstimator
from moxie.kernel_ridge import (
    KernelRidge,
    RowKernel,
    ScalarOrVector,
    WeightRequest,
)
from moxie.kernel_ridge_serialization import KernelRidgeSchema
from std.utils import Variant


@fieldwise_init
struct ParameterKernel(Copyable, RowKernel):
    var offset: Float64

    def evaluate[
        dtype: DType
    ](
        self, X: nm.NDArray[dtype], i: Int, Y: nm.NDArray[dtype], j: Int
    ) raises -> SIMD[dtype, 1]:
        return X.item(i, 0) * Y.item(j, 0) + SIMD[dtype, 1](self.offset)

    def evaluate_with_params[
        dtype: DType
    ](
        self,
        X: nm.NDArray[dtype],
        i: Int,
        Y: nm.NDArray[dtype],
        j: Int,
        params: Optional[KernelParameters],
    ) raises -> SIMD[dtype, 1]:
        var scale: Float64 = 1
        if params:
            for entry in params.value().items():
                if entry.key != "scale":
                    raise Error("TypeError: unexpected kernel parameter")
            if "scale" in params.value():
                if not params.value()["scale"].isa[Float64]():
                    raise Error("TypeError: scale must be Float64")
                scale = params.value()["scale"][Float64]
        return SIMD[dtype, 1](scale) * self.evaluate(X, i, Y, j)


def configuration_roundtrip[E: BaseEstimator](mut model: E, params: E.Params) raises:
    _ = model.set_params(params)


def contracts[dtype: DType]() raises:
    var X = nm.arange[dtype](6).reshape(nm.Shape(3, 2))
    var y = nm.arange[dtype](3)
    var names: List[String] = ["height", "width"]
    var wrong: List[String] = ["width", "height"]
    var model = KernelRidge[dtype]()
    with assert_raises():
        _ = model.predict(X, feature_names=names)
    _ = model.fit(X, y, feature_names=names)
    names[0] = "mutated"
    assert_equal(model.feature_names_in_.value()[0], "height")
    names[0] = "height"
    var output = model.predict(X, feature_names=names)
    _ = model.score(X, y, feature_names=names)
    with assert_raises():
        _ = model.predict(X, feature_names=wrong)
    with assert_raises():
        _ = model.score(X, y, feature_names=wrong)
    with assert_raises():
        _ = model.fit(X, y, feature_names=List[String](["one"]))
    assert_equal(model.feature_names_in_.value()[0], "height")
    _ = model.predict(X)
    var fresh = clone(model)
    assert_true(not fresh.feature_names_in_)
    assert_true(not fresh.__sklearn_is_fitted__())

    var metadata = Dict[String, ScalarOrVector]()
    metadata["sample_weight"] = ScalarOrVector(2.0)
    var routing = model.get_metadata_routing()
    with assert_raises():
        _ = routing.fit.route(metadata)
    _ = model.set_fit_request(sample_weight=True)
    _ = model.set_score_request(sample_weight=String("weights"))
    _ = model.set_fit_request()
    routing = model.get_metadata_routing()
    assert_equal(routing.fit.route(metadata).value()[Float64], 2.0)
    assert_true(not routing.score.route(metadata))
    metadata["weights"] = ScalarOrVector(3.0)
    assert_equal(routing.score.route(metadata).value()[Float64], 3.0)
    var cloned = clone(model)
    assert_equal(
        cloned.get_metadata_routing().score.route(metadata).value()[Float64],
        3.0,
    )
    _ = model.set_fit_request(sample_weight=False)
    assert_true(not model.get_metadata_routing().fit.route(metadata))
    assert_true(Bool(routing.fit.route(metadata)))
    with assert_raises():
        _ = model.set_fit_request(sample_weight=String("bad alias"))
    _ = model.set_fit_request(sample_weight=None)
    with assert_raises():
        _ = model.get_metadata_routing().fit.route(metadata)
    # Direct calls remain independent of parent routing requests.
    _ = model.fit(X, y, feature_names=names, sample_weight=ScalarOrVector(2.0))
    comptime Schema = KernelRidgeSchema[dtype]
    var archive = Schema.encode(model)
    var restored = Schema.decode(archive)
    assert_equal(restored.feature_names_in_.value()[1], "width")
    assert_equal(
        restored.get_metadata_routing().score.route(metadata).value()[Float64],
        3.0,
    )
    with assert_raises():
        _ = restored.predict(X, feature_names=wrong)
    restored.feature_names_in_ = List[String](["invalid"])
    with assert_raises():
        _ = Schema.encode(restored)
    _ = model.fit(X, y)
    assert_true(not model.feature_names_in_)
    _ = model.predict(X, feature_names=names)
    var duplicate: List[String] = ["same", "same"]
    _ = model.fit(X, y, feature_names=duplicate)
    _ = model.predict(X, feature_names=duplicate)
    for mismatched in [
        List[String](["height"]),
        List[String](["same", "renamed"]),
        List[String](["same", "same", "extra"]),
    ]:
        with assert_raises():
            _ = model.predict(X, feature_names=mismatched)

    _ = model.set_params(gamma=0.5)
    assert_equal(model.gamma.value(), 0.5)
    _ = model.set_params(coef0=2.0)
    assert_equal(model.gamma.value(), 0.5)
    _ = model.set_params(gamma=None, kernel_params=None, native_kernel=None)
    assert_true(not model.gamma)
    var configuration = model.get_params()
    configuration_roundtrip(model, configuration)
    var mapping = model.get_params()
    mapping["alpha"] = KernelRidge[dtype].ParameterValue(2.0)
    _ = model.set_params(mapping)
    assert_equal(model.alpha[Float64], 2.0)
    mapping["alpha"] = KernelRidge[dtype].ParameterValue(5.0)
    mapping["unexpected"] = KernelRidge[dtype].ParameterValue(1.0)
    with assert_raises():
        _ = model.set_params(mapping)
    assert_equal(model.alpha[Float64], 2.0)
    var bad = KernelRidge[dtype].ParameterMap()
    bad["degree"] = KernelRidge[dtype].ParameterValue(String("wrong"))
    with assert_raises():
        _ = model.set_params(bad)

    var mixed = KernelParameters()
    mixed["scale"] = KernelParameterValue(1.5)
    mixed["degree"] = KernelParameterValue(Int(2))
    mixed["enabled"] = KernelParameterValue(False)
    mixed["mode"] = KernelParameterValue(String("scaled"))
    _ = model.set_params(kernel_params=mixed.copy())
    var mixed_clone = clone(model)
    var mixed_mapping = model.get_params()
    configuration_roundtrip(model, mixed_mapping)
    mixed["mode"] = KernelParameterValue(String("changed"))
    assert_equal(model.kernel_params.value()["mode"][String], "scaled")
    _ = model.set_params(kernel_params=None)
    assert_equal(mixed_clone.kernel_params.value()["degree"][Int], 2)
    assert_true(not mixed_clone.kernel_params.value()["enabled"][Bool])
    assert_equal(mixed_clone.kernel_params.value()["scale"][Float64], 1.5)

    var params = KernelParameters()
    params["scale"] = 2.0
    var callable = KernelRidge[dtype, ParameterKernel](
        kernel=ParameterKernel(1), kernel_params=params.copy()
    )
    # Copies retain independent typed parameter maps.
    var configured_copy = clone(callable)
    params["scale"] = KernelParameterValue(99.0)
    assert_equal(configured_copy.kernel_params.value()["scale"][Float64], 2.0)
    var wrong_type = KernelParameters()
    wrong_type["scale"] = KernelParameterValue(String("wrong"))
    _ = configured_copy.set_params(kernel_params=wrong_type^)
    with assert_raises():
        _ = configured_copy.fit(X, y)
    _ = callable.fit(X, y)
    var called = callable.predict(X)
    var gram = nm.zeros[dtype](nm.Shape(3, 3))
    for i in range(3):
        for j in range(3):
            gram.unsafe_set(i * 3 + j, 2 * (X.item(i, 0) * X.item(j, 0) + 1))
    var precomputed = KernelRidge[dtype](kernel="precomputed")
    _ = precomputed.fit(gram, y)
    var expected = precomputed.predict(gram)
    for i in range(3):
        assert_equal(
            called[nm.NDArray[dtype]].item(i),
            expected[nm.NDArray[dtype]].item(i),
        )
    var copied = clone(callable)
    _ = copied.set_params(
        kernel=ParameterKernel(2), gamma=None, alpha=ScalarOrVector(0.5)
    )
    assert_equal(copied.alpha[Float64], 0.5)
    assert_equal(copied.native_kernel.value().offset, 2.0)
    _ = copied.set_params(kernel_params=None)
    assert_true(not copied.kernel_params)
    _ = callable.set_params(kernel="linear")
    assert_true(not callable.native_kernel)
    params["unknown"] = 1.0
    _ = copied.set_params(kernel_params=params^)
    with assert_raises():
        _ = copied.fit(X, y)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("KernelRidge native API contracts passed")
