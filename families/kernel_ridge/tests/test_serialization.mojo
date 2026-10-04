"""KernelRidge schema round trips, ownership, and invalid-state rejection."""
from moxie.kernel_ridge import KernelParameters, KernelParameterValue
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from std.math import isnan
from std.memory import bitcast
from moxie.base import clone
from moxie.kernel_ridge import (
    KernelRidge,
    ScalarOrVector,
    FloatingArray,
    FittedArray,
    RowKernel,
)
from moxie.kernel_ridge_serialization import (
    KernelRidgeSchema,
    NativeKernelAdapter,
    _NoKernelAdapter,
)
from moxie._kernel_ridge._estimator import NoKernel
from moxie.serialization import (
    dumps,
    loads,
    save_model,
    load_model,
    BinaryWriter,
    BinaryReader,
)
from callable_kernel import AffineKernel, AffineKernelAdapter


def same_bytes(a: List[UInt8], b: List[UInt8]) raises:
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def same_predictions(a: FloatingArray, b: FloatingArray) raises:
    comptime types = (DType.float32, DType.float64)
    comptime for i in range(2):
        comptime dt = types[i]
        assert_equal(a.isa[nm.NDArray[dt]](), b.isa[nm.NDArray[dt]]())
        if a.isa[nm.NDArray[dt]]():
            var left = a[nm.NDArray[dt]].copy()
            var right = b[nm.NDArray[dt]].copy()
            assert_equal(left.ndim, right.ndim)
            for j in range(left.ndim):
                assert_equal(left.shape[j], right.shape[j])
            for j in range(left.size):
                assert_equal(left.unsafe_get(j), right.unsafe_get(j))


def round_trips[dtype: DType]() raises:
    comptime Schema = KernelRidgeSchema[dtype]
    var X = nm.arange[dtype](6).reshape(nm.Shape(3, 2))
    var y = nm.arange[dtype](3)
    var fresh = KernelRidge[dtype]()
    var unfitted = Schema.decode(Schema.encode(fresh))
    assert_true(not unfitted.__sklearn_is_fitted__())
    with assert_raises():
        _ = unfitted.predict(X)
    _ = unfitted.fit(X, y)
    for kernel in [
        String("linear"),
        "rbf",
        "laplacian",
        "poly",
        "polynomial",
        "sigmoid",
        "cosine",
        "chi2",
        "additive_chi2",
        "precomputed",
    ]:
        for rank in range(3):
            var train = X.copy()
            if kernel == "precomputed":
                train = nm.eye[dtype](3, 3)
            var targets = y.copy()
            if rank == 1:
                targets = y.reshape(nm.Shape(3, 1))
            elif rank == 2:
                targets = X.copy()
            var model = KernelRidge[dtype](
                100.0, kernel=kernel.copy(), gamma=0.25
            )
            if rank == 2:
                model.alpha = ScalarOrVector(List[Float64]([100.0, 101.0]))
            _ = model.fit(
                train,
                targets,
                sample_weight=ScalarOrVector(List[Float64]([0.5, 1.0, 2.0])),
            )
            var bytes = dumps(Schema.encode(model))
            var restored = Schema.decode(loads(bytes))
            same_bytes(bytes, dumps(Schema.encode(restored)))
            same_predictions(model.predict(train), restored.predict(train))
            assert_equal(
                model.score(train, targets), restored.score(train, targets)
            )
            assert_true(not clone(restored).__sklearn_is_fitted__())
            # Constructor changes do not imply any change to stored learned state.
            _ = model.set_params(
                alpha=ScalarOrVector(List[Float64]([-1.0, 2.0, 3.0])),
                degree=-1.0,
            )
            var changed = Schema.decode(Schema.encode(model))
            same_bytes(
                dumps(Schema.encode(model)), dumps(Schema.encode(changed))
            )
            with assert_raises():
                _ = changed.fit(train, targets)
            _ = restored.fit(train, targets)
            same_predictions(
                restored.predict(train),
                Schema.decode(Schema.encode(restored)).predict(train),
            )
    # Empty versus absent dictionaries, gamma None, and nonfinite ignored settings.
    fresh.kernel_params = KernelParameters()
    var empty = Schema.decode(Schema.encode(fresh))
    assert_true(Bool(empty.kernel_params))
    assert_equal(len(empty.kernel_params.value()), 0)
    var params = KernelParameters()
    params["ignored\x00parameter"] = -0.0
    params["count"] = KernelParameterValue(Int(-7))
    params["enabled"] = KernelParameterValue(True)
    params["mode"] = KernelParameterValue(String("mixed\x00λ"))
    fresh.kernel_params = params^
    fresh.degree = Float64("nan")
    var preserved = Schema.decode(Schema.encode(fresh))
    same_bytes(dumps(Schema.encode(fresh)), dumps(Schema.encode(preserved)))
    assert_true(not preserved.gamma)
    assert_true(isnan(preserved.degree))
    assert_equal(len(preserved.kernel_params.value()), 4)
    assert_equal(preserved.kernel_params.value()["count"][Int], -7)
    assert_true(preserved.kernel_params.value()["enabled"][Bool])
    assert_equal(preserved.kernel_params.value()["mode"][String], "mixed\x00λ")
    assert_equal(
        bitcast[DType.uint64, 1](
            preserved.kernel_params.value()["ignored\x00parameter"][Float64]
        ),
        bitcast[DType.uint64, 1](Float64(-0.0)),
    )
    var configured = KernelRidge[dtype](
        ScalarOrVector(List[Float64]([0.25, 0.5])),
        kernel="polynomial",
        gamma=0.75,
        degree=2.5,
        coef0=1.25,
    )
    var config = Schema.decode(Schema.encode(configured)).get_params_record()
    assert_equal(config.kernel, "polynomial")
    assert_equal(config.gamma.value(), 0.75)
    assert_equal(config.degree, 2.5)
    assert_equal(config.coef0, 1.25)
    assert_equal(config.alpha[List[Float64]][0], 0.25)
    assert_equal(config.alpha[List[Float64]][1], 0.5)
    assert_true(not config.kernel_params)
    var fitted = KernelRidge[dtype]()
    _ = fitted.fit(X, y)
    # Save/load invokes the same shared engine; no refit is needed.
    var path = "builds/kernel-ridge-" + String(dtype) + ".moxie"
    save_model[Schema](fitted, path)
    var loaded = load_model[Schema](path)
    same_predictions(fitted.predict(X), loaded.predict(X))
    loaded.X_fit_[nm.NDArray[dtype]].unsafe_set(0, 99)
    assert_equal(fitted.X_fit_[nm.NDArray[dtype]].item(0, 0), 0)
    # A changed kernel need not describe the historical training matrix.
    _ = fitted.set_params(kernel="precomputed")
    var changed_kernel = Schema.decode(Schema.encode(fitted))
    same_bytes(
        dumps(Schema.encode(fitted)), dumps(Schema.encode(changed_kernel))
    )
    with assert_raises():
        _ = changed_kernel.predict(X)


def mixed_dtypes_and_strides() raises:
    comptime types = (DType.float32, DType.float64, DType.int32, DType.int64)
    comptime for i in range(4):
        comptime dt = types[i]
        var X = nm.arange[dt](6).reshape(nm.Shape(3, 2))
        var reversed = X.view_with_layout(
            nm.Shape(3, 2), nm.NDArrayStrides(-2, 1), 4
        )
        var y = nm.arange[DType.float32](3)
        var model = KernelRidge[DType.float32]()
        _ = model.fit(reversed, y)
        var restored = KernelRidgeSchema[DType.float32].decode(
            KernelRidgeSchema[DType.float32].encode(model)
        )
        assert_true(restored.X_fit_.isa[nm.NDArray[dt]]())
        assert_equal(restored.X_fit_[nm.NDArray[dt]].item(0, 0), 4)
        same_predictions(model.predict(X), restored.predict(X))
        # Both coefficient ranks/dtypes are independent of estimator dtype.
        var y64 = nm.arange[DType.float64](3).reshape(nm.Shape(3, 1))
        _ = model.fit(X, y64)
        restored = KernelRidgeSchema[DType.float32].decode(
            KernelRidgeSchema[DType.float32].encode(model)
        )
        assert_true(restored.dual_coef_.isa[nm.NDArray[DType.float64]]())
        same_predictions(model.predict(X), restored.predict(X))
    var large = nm.ones[DType.int64](nm.Shape(2, 1))
    large.unsafe_set(0, 9007199254740993)
    var model = KernelRidge(kernel="rbf")
    _ = model.fit(large, nm.arange[DType.float64](2))
    var restored = KernelRidgeSchema.decode(KernelRidgeSchema.encode(model))
    assert_equal(
        restored.X_fit_[nm.NDArray[DType.int64]].item(0, 0), 9007199254740993
    )
    # Public learned arrays may also be views; preserve logical coefficients.
    var backing = nm.arange[DType.float64](4)
    var coefficients = backing.view_with_layout(
        nm.Shape(2), nm.NDArrayStrides(-2), 3
    )
    model.dual_coef_ = FloatingArray(coefficients^)
    restored = KernelRidgeSchema.decode(KernelRidgeSchema.encode(model))
    assert_equal(restored.dual_coef_[nm.NDArray[DType.float64]].item(0), 3)
    assert_equal(restored.dual_coef_[nm.NDArray[DType.float64]].item(1), 1)
    same_predictions(model.predict(large), restored.predict(large))


@fieldwise_init
struct CustomKernel(Copyable, RowKernel):
    var offset: Float64

    def evaluate[
        dtype: DType
    ](
        self, X: nm.NDArray[dtype], i: Int, Y: nm.NDArray[dtype], j: Int
    ) raises -> Scalar[dtype]:
        return Scalar[dtype](self.offset)


def callable_round_trips[dtype: DType]() raises:
    comptime Schema = KernelRidgeSchema[
        dtype, AffineKernel, AffineKernelAdapter
    ]
    var X = nm.arange[dtype](6).reshape(nm.Shape(3, 2))
    var query = nm.arange[dtype](4).reshape(nm.Shape(2, 2))
    var y = nm.arange[dtype](3)
    # A typed estimator can still have no native kernel configured.
    var absent = KernelRidge[dtype, AffineKernel]()
    var absent_copy = Schema.decode(Schema.encode(absent))
    assert_true(not absent_copy.native_kernel)
    same_bytes(
        dumps(Schema.encode(absent)),
        dumps(KernelRidgeSchema[dtype].encode(KernelRidge[dtype]())),
    )
    var model = KernelRidge[dtype, AffineKernel](
        alpha=0.25, native_kernel=AffineKernel(1.5)
    )
    var unfitted = Schema.decode(loads(dumps(Schema.encode(model))))
    assert_true(not unfitted.__sklearn_is_fitted__())
    assert_equal(unfitted.native_kernel.value().offset, 1.5)
    with assert_raises():
        _ = unfitted.predict(query)
    _ = unfitted.fit(X, y)
    for rank in range(3):
        var targets = y.copy()
        if rank == 1:
            targets = y.reshape(nm.Shape(3, 1))
        elif rank == 2:
            targets = X.copy()
        _ = model.fit(X, targets)
        var path = "builds/callable-" + String(dtype) + ".moxie"
        save_model[Schema](model, path)
        var restored = load_model[Schema](path)
        same_predictions(model.predict(query), restored.predict(query))
        same_predictions(model.dual_coef_, restored.dual_coef_)
        assert_equal(model.score(X, targets), restored.score(X, targets))
        same_bytes(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
        var fresh = clone(restored)
        assert_true(not fresh.__sklearn_is_fitted__())
        assert_equal(fresh.native_kernel.value().offset, 1.5)
        restored.native_kernel = AffineKernel(2.5)
        assert_equal(model.native_kernel.value().offset, 1.5)
        restored.X_fit_[nm.NDArray[dtype]].unsafe_set(0, 99)
        assert_equal(model.X_fit_[nm.NDArray[dtype]].item(0, 0), 0)
        _ = restored.fit(X, targets)
        same_predictions(
            restored.predict(query),
            Schema.decode(Schema.encode(restored)).predict(query),
        )
    # Current callable parameters are separate from the historical fit.
    _ = model.set_params(native_kernel=AffineKernel(3.0))
    var changed = Schema.decode(Schema.encode(model))
    assert_equal(changed.native_kernel.value().offset, 3.0)
    same_predictions(model.predict(query), changed.predict(query))
    same_predictions(model.dual_coef_, changed.dual_coef_)
    # Preserve signed zero exactly in the adapter's Float64 parameter.
    model.native_kernel = AffineKernel(-0.0)
    same_bytes(
        dumps(Schema.encode(model)),
        dumps(Schema.encode(Schema.decode(Schema.encode(model)))),
    )


def invalid_callable_states() raises:
    comptime Schema = KernelRidgeSchema[
        DType.float64, AffineKernel, AffineKernelAdapter
    ]
    var model = KernelRidge[DType.float64, AffineKernel](
        native_kernel=AffineKernel(1.5)
    )
    var valid = Schema.encode(model)
    with assert_raises():
        _ = KernelRidgeSchema.decode(valid)
    with assert_raises():
        _ = KernelRidgeSchema[DType.float64, AffineKernel].decode(valid)
    # Build well-formed container state so semantic rejection is exercised.
    var absent = KernelRidge[DType.float64, AffineKernel]()
    var base = Schema.encode(absent)
    for problem in range(8):
        var bad = base.copy()
        var extension = BinaryWriter()
        extension.string(
            "unknown.kernel" if problem
            == 0 else AffineKernelAdapter.identifier()
        )
        extension.uint(UInt64(2 if problem == 1 else 1))
        var state = BinaryWriter()
        if problem != 2:
            state.scalar[DType.float64](
                Float64("nan") if problem
                == 3 else Float64("inf") if problem
                == 4 else 1.5
            )
        if problem == 5:
            state.boolean(False)
        var bytes = state^.finish()
        extension.bytes(bytes)
        if problem == 6:
            extension.boolean(False)
        var suffix = extension^.finish()
        if problem == 7:
            _ = suffix.pop()
        bad.manifest.extend(Span(suffix))
        with assert_raises():
            _ = Schema.decode(bad)
    for offset in [Float64("nan"), Float64("inf"), Float64("-inf")]:
        model.native_kernel = AffineKernel(offset)
        with assert_raises():
            _ = Schema.encode(model)


def invalid_states() raises:
    comptime Schema = KernelRidgeSchema
    var model = KernelRidge()
    var X = nm.arange[DType.float64](6).reshape(nm.Shape(3, 2))
    _ = model.fit(X, nm.arange[DType.float64](3))
    var valid = Schema.encode(model)
    for problem in range(9):
        var bad = valid.copy()
        if problem == 0:
            bad.type_id = "wrong.estimator"
        elif problem == 1:
            bad.schema_version = 4
        elif problem == 2:
            bad.manifest.append(0)
        elif problem == 3:
            bad.payloads.append(bad.payloads[0].copy())
        elif problem == 4:
            bad.payloads[0].name = "unknown"
        elif problem == 5:
            bad.payloads[0].shape = List[Int]([2, 3])
        elif problem == 6:
            bad.payloads[1].shape = List[Int]([1, 3])
        elif problem == 7:
            bad.payloads[1].dtype = 4
        else:
            bad.payloads[0].shape = List[Int]([6])
        with assert_raises():
            _ = Schema.decode(bad)
    with assert_raises():
        _ = KernelRidgeSchema[DType.float32].decode(valid)
    model.dual_coef_[nm.NDArray[DType.float64]].unsafe_set(0, Float64("nan"))
    with assert_raises():
        _ = Schema.encode(model)
    _ = model.fit(X, nm.arange[DType.float64](3))
    model.X_fit_[nm.NDArray[DType.float64]].unsafe_set(0, Float64("inf"))
    with assert_raises():
        _ = Schema.encode(model)
    var fresh = KernelRidge()
    fresh.n_features_in_ = 2
    with assert_raises():
        _ = Schema.encode(fresh)
    fresh.n_features_in_ = 0
    fresh.dual_coef_ = FloatingArray(nm.ones[DType.float64](nm.Shape(1)))
    with assert_raises():
        _ = Schema.encode(fresh)
    var custom = KernelRidge[DType.float64, CustomKernel](
        native_kernel=CustomKernel(1.0)
    )
    with assert_raises():
        _ = KernelRidgeSchema[DType.float64, CustomKernel].encode(custom)


def saved_fixtures[
    dtype: DType,
    Metric: RowKernel & Copyable & Deinitable = NoKernel,
    Adapter: NativeKernelAdapter = _NoKernelAdapter[Metric],
](name: String, native: Bool) raises:
    comptime Schema = KernelRidgeSchema[dtype, Metric, Adapter]
    var prefix = "tests/fixtures/schema-v1/" + name + "-" + String(dtype)
    var query = nm.arange[dtype](4).reshape(nm.Shape(2, 2))
    for fitted in [False, True]:
        var path = prefix + ("-fitted.moxie" if fitted else "-unfitted.moxie")
        var bytes: List[UInt8]
        with open(path, "r") as file:
            bytes = file.read_bytes()
        var archive = loads(bytes)
        assert_equal(archive.schema_version, 1)
        var model = Schema.decode(archive)
        var migrated = Schema.encode(model)
        assert_equal(migrated.schema_version, 3)
        same_bytes(
            dumps(migrated), dumps(Schema.encode(Schema.decode(migrated)))
        )
        assert_equal(model.__sklearn_is_fitted__(), fitted)
        assert_equal(Bool(model.native_kernel), native)
        assert_equal(model.alpha[Float64], 0.25 if native else 1.0)
        assert_equal(model.kernel, "linear")
        assert_true(not model.gamma)
        assert_equal(model.degree, 3.0)
        assert_equal(model.coef0, 1.0)
        assert_true(not model.kernel_params)
        assert_equal(model.n_features_in_, 2 if fitted else 0)
        comptime if Metric == AffineKernel:
            assert_equal(
                rebind[AffineKernel](model.native_kernel.value()).offset, 1.5
            )
        if not fitted:
            with assert_raises():
                _ = model.predict(query)
            continue
        var X = model.X_fit_[nm.NDArray[dtype]].copy()
        assert_equal(X.shape[0], 3)
        assert_equal(X.shape[1], 2)
        for i in range(6):
            assert_equal(X.unsafe_get(i), Scalar[dtype](i))
        var predictions = model.predict(query)
        var values = predictions[nm.NDArray[dtype]].copy()
        assert_equal(values.ndim, 1)
        assert_equal(values.shape[0], 2)
        var expected: List[UInt8]
        with open(prefix + ".predictions", "r") as file:
            expected = file.read_bytes()
        var reader = BinaryReader(Span(expected))
        comptime bits = DType.uint32 if dtype == DType.float32 else DType.uint64
        for i in range(2):
            assert_equal(
                bitcast[bits, 1](values.item(i)),
                bitcast[bits, 1](reader.scalar[dtype]()),
            )
        reader.finish()



def legacy_parameters_and_invalid_tags() raises:
    comptime Schema = KernelRidgeSchema
    var params: KernelParameters = {"scale": 2.5}
    var model = KernelRidge(kernel_params=params^)
    var archive = Schema.encode(model)
    var reader = BinaryReader(Span(archive.manifest))
    _ = reader.uint()  # dtype
    assert_true(not reader.boolean())  # scalar alpha
    _ = reader.scalar[DType.float64]()
    _ = reader.string()  # kernel
    assert_true(not reader.boolean())  # gamma
    _ = reader.scalar[DType.float64]()  # degree
    _ = reader.scalar[DType.float64]()  # coef0
    assert_true(reader.boolean())
    assert_equal(reader.uint(), UInt64(1))
    assert_equal(reader.string(), "scale")
    var tag_offset = reader.position
    assert_equal(reader.uint(), UInt64(0))
    # Schema 2 used identical surrounding fields but no value discriminator.
    var legacy = archive.copy()
    legacy.schema_version = 2
    legacy.manifest = List[UInt8]()
    for i in range(len(archive.manifest)):
        if i < tag_offset or i >= tag_offset + 8:
            legacy.manifest.append(archive.manifest[i])
    var restored = Schema.decode(legacy)
    assert_equal(restored.kernel_params.value()["scale"][Float64], 2.5)
    assert_equal(Schema.encode(restored).schema_version, 3)
    var invalid = archive.copy()
    invalid.manifest[tag_offset] = 99
    with assert_raises():
        _ = Schema.decode(invalid)
    # Reinterpreting a float payload as Bool must reject its non-boolean bits.
    invalid = archive.copy()
    invalid.manifest[tag_offset] = 2
    with assert_raises():
        _ = Schema.decode(invalid)

def main() raises:
    legacy_parameters_and_invalid_tags()
    round_trips[DType.float32]()
    round_trips[DType.float64]()
    mixed_dtypes_and_strides()
    invalid_states()
    callable_round_trips[DType.float32]()
    callable_round_trips[DType.float64]()
    invalid_callable_states()
    saved_fixtures[DType.float32]("linear", False)
    saved_fixtures[DType.float64]("linear", False)
    saved_fixtures[DType.float32, AffineKernel, AffineKernelAdapter](
        "affine", True
    )
    saved_fixtures[DType.float64, AffineKernel, AffineKernelAdapter](
        "affine", True
    )
    print("KernelRidge serialization contracts passed")
