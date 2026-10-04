"""Family-owned KernelRidge state schema for Moxie's native model container."""
from moxie._kernel_ridge._api import KernelParameters, KernelParameterValue
import numojo as nm
from moxie.serialization import (
    ModelSchema,
    ModelArchive,
    BinaryWriter,
    BinaryReader,
    dtype_code,
)
from .kernel_ridge import (
    KernelRidge,
    ScalarOrVector,
    FloatingArray,
    FittedArray,
    RowKernel,
)
from ._kernel_ridge._estimator import NoKernel
from ._kernel_ridge._api import WeightRequest, validate_request, unchanged
from std.utils import Variant
from ._kernel_ridge._validation import validate


def _write_request(mut writer: BinaryWriter, request: WeightRequest) raises:
    validate_request(request)
    if unchanged(request):
        raise Error("ModelSchemaError: stored request cannot be UNCHANGED")
    writer.boolean(Bool(request))
    if request:
        writer.boolean(request.value().isa[Bool]())
        if request.value().isa[Bool]():
            writer.boolean(request.value()[Bool])
        else:
            writer.string(request.value()[String])


def _read_request(mut reader: BinaryReader) raises -> WeightRequest:
    if not reader.boolean():
        return None
    var result: WeightRequest
    if reader.boolean():
        result = Variant[Bool, String](reader.boolean())
    else:
        result = Variant[Bool, String](reader.string())
    validate_request(result)
    if unchanged(result):
        raise Error("ModelSchemaError: stored request cannot be UNCHANGED")
    return result^


trait NativeKernelAdapter:
    """Trusted, compile-time codec for one native kernel's validated state."""

    comptime Kernel: RowKernel & Copyable & Deinitable

    @staticmethod
    def identifier() -> String:
        ...

    @staticmethod
    def state_version() -> Int:
        ...

    @staticmethod
    def encode(kernel: Self.Kernel) raises -> List[UInt8]:
        ...

    @staticmethod
    def decode(state: Span[UInt8, _]) raises -> Self.Kernel:
        ...


struct _NoKernelAdapter[Metric: RowKernel & Copyable & Deinitable](
    NativeKernelAdapter
):
    comptime Kernel = Self.Metric

    @staticmethod
    def identifier() -> String:
        return ""

    @staticmethod
    def state_version() -> Int:
        return 0

    @staticmethod
    def encode(kernel: Self.Kernel) raises -> List[UInt8]:
        raise Error(
            "ModelSchemaError: native kernel requires an explicit adapter"
        )

    @staticmethod
    def decode(state: Span[UInt8, _]) raises -> Self.Kernel:
        raise Error(
            "ModelSchemaError: native kernel requires an explicit adapter"
        )


def _check_adapter[Adapter: NativeKernelAdapter]() raises:
    var identifier = Adapter.identifier()
    if (
        identifier.byte_length() == 0
        or identifier.byte_length() > 4096
        or Adapter.state_version() < 1
    ):
        raise Error(
            "ModelSchemaError: native kernel requires a valid explicit adapter"
        )


def _write_array[
    dtype: DType
](mut archive: ModelArchive, name: String, value: nm.NDArray[dtype]) raises:
    if value.ndim != 1 and value.ndim != 2:
        raise Error("ModelSchemaError: KernelRidge arrays require rank 1 or 2")
    var shape = List[Int]()
    for i in range(value.ndim):
        shape.append(value.shape[i])
    var values = List[Scalar[dtype]]()
    for i in range(value.shape[0]):
        if value.ndim == 1:
            # Adopted NuMojo item(i) ignores strides for rank-one arrays.
            values.append(value.unsafe_get(i * Int(value.strides[0])))
        else:
            for j in range(value.shape[1]):
                values.append(value.item(i, j))
    archive.add_array[dtype](name, shape, values)


def _read_array[
    dtype: DType
](archive: ModelArchive, name: String) raises -> nm.NDArray[dtype]:
    var payload = archive.array(name)
    if len(payload.shape) != 1 and len(payload.shape) != 2:
        raise Error("ModelSchemaError: KernelRidge arrays require rank 1 or 2")
    var values = payload.values[dtype]()
    var shape = nm.Shape(payload.shape[0])
    if len(payload.shape) == 2:
        shape = nm.Shape(payload.shape[0], payload.shape[1])
    var result = nm.zeros[dtype](shape)
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


struct KernelRidgeSchema[
    dtype: DType = DType.float64,
    Metric: RowKernel & Copyable & Deinitable = NoKernel,
    Adapter: NativeKernelAdapter = _NoKernelAdapter[Metric],
](ModelSchema):
    """Schema 3 (reads schemas 1 and 2) with an opt-in, compile-time native kernel state adapter.
    """

    comptime Model = KernelRidge[Self.dtype, Self.Metric]

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        comptime assert (
            Self.Adapter.Kernel == Self.Metric
        ), "adapter kernel type mismatch"
        var archive = ModelArchive("moxie.kernel_ridge.KernelRidge", 3)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.boolean(model.alpha.isa[List[Float64]]())
        if model.alpha.isa[List[Float64]]():
            writer.uint(UInt64(len(model.alpha[List[Float64]])))
            for value in model.alpha[List[Float64]]:
                writer.scalar[DType.float64](value)
        else:
            writer.scalar[DType.float64](model.alpha[Float64])
        writer.string(model.kernel)
        writer.boolean(Bool(model.gamma))
        if model.gamma:
            writer.scalar[DType.float64](model.gamma.value())
        writer.scalar[DType.float64](model.degree)
        writer.scalar[DType.float64](model.coef0)
        writer.boolean(Bool(model.kernel_params))
        if model.kernel_params:
            writer.uint(UInt64(len(model.kernel_params.value())))
            for entry in model.kernel_params.value().items():
                writer.string(entry.key)
                if entry.value.isa[Float64]():
                    writer.uint(0)
                    writer.scalar[DType.float64](entry.value[Float64])
                elif entry.value.isa[Int]():
                    writer.uint(1)
                    writer.integer(entry.value[Int])
                elif entry.value.isa[Bool]():
                    writer.uint(2)
                    writer.boolean(entry.value[Bool])
                else:
                    writer.uint(3)
                    writer.string(entry.value[String])
        writer.boolean(model._fitted)
        writer.integer(model.n_features_in_)
        writer.boolean(Bool(model.feature_names_in_))
        if model.feature_names_in_:
            writer.uint(UInt64(len(model.feature_names_in_.value())))
            for name in model.feature_names_in_.value():
                writer.string(name)
        _write_request(writer, model._metadata.fit.sample_weight)
        _write_request(writer, model._metadata.score.sample_weight)
        if model.native_kernel:
            _check_adapter[Self.Adapter]()
            var state = Self.Adapter.encode(
                rebind[Self.Adapter.Kernel](model.native_kernel.value())
            )
            writer.string(Self.Adapter.identifier())
            writer.uint(UInt64(Self.Adapter.state_version()))
            writer.bytes(state)
        archive.manifest = writer^.finish()
        comptime types = (
            DType.float32,
            DType.float64,
            DType.int32,
            DType.int64,
        )
        comptime for i in range(4):
            comptime dt = types[i]
            if model.X_fit_.isa[nm.NDArray[dt]]():
                _write_array(archive, "X_fit_", model.X_fit_[nm.NDArray[dt]])
        comptime for i in range(2):
            comptime dt = types[i]
            if model.dual_coef_.isa[nm.NDArray[dt]]():
                _write_array(
                    archive, "dual_coef_", model.dual_coef_[nm.NDArray[dt]]
                )
        _ = Self.decode(archive)
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        comptime assert (
            Self.Adapter.Kernel == Self.Metric
        ), "adapter kernel type mismatch"
        archive.validate()
        if archive.schema_version not in (1, 2, 3):
            raise Error(
                "ModelSchemaError: unsupported KernelRidge schema version"
            )
        archive.expect("moxie.kernel_ridge.KernelRidge", archive.schema_version)
        if len(archive.payloads) != 2:
            raise Error(
                "ModelSchemaError: expected only X_fit_ and dual_coef_ arrays"
            )
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != UInt64(dtype_code[Self.dtype]()):
            raise Error(
                "ModelSchemaError: KernelRidge estimator dtype mismatch"
            )
        var model = Self.Model()
        if reader.boolean():
            var count = reader.count(reader.remaining() // 8)
            var alpha = List[Float64]()
            for _ in range(count):
                alpha.append(reader.scalar[DType.float64]())
            model.alpha = ScalarOrVector(alpha^)
        else:
            model.alpha = ScalarOrVector(reader.scalar[DType.float64]())
        model.kernel = reader.string()
        if reader.boolean():
            model.gamma = reader.scalar[DType.float64]()
        model.degree = reader.scalar[DType.float64]()
        model.coef0 = reader.scalar[DType.float64]()
        if reader.boolean():
            var count = reader.count(reader.remaining() // 16)
            var params = KernelParameters()
            for _ in range(count):
                var key = reader.string()
                if key in params:
                    raise Error("ModelSchemaError: duplicate kernel_params key")
                var tag = UInt64(0)
                if archive.schema_version >= 3:
                    tag = reader.uint()
                if tag == 0:
                    params[key] = KernelParameterValue(reader.scalar[DType.float64]())
                elif tag == 1:
                    params[key] = KernelParameterValue(reader.integer())
                elif tag == 2:
                    params[key] = KernelParameterValue(reader.boolean())
                elif tag == 3:
                    params[key] = KernelParameterValue(reader.string())
                else:
                    raise Error("ModelSchemaError: unknown kernel parameter type")
            model.kernel_params = params^
        model._fitted = reader.boolean()
        model.n_features_in_ = reader.integer()
        if archive.schema_version >= 2:
            if reader.boolean():
                var count = reader.count(reader.remaining() // 8)
                var names = List[String]()
                for _ in range(count):
                    names.append(reader.string())
                if not model._fitted or len(names) != model.n_features_in_:
                    raise Error(
                        "ModelSchemaError: invalid learned feature names"
                    )
                model.feature_names_in_ = names^
            model._metadata.fit.sample_weight = _read_request(reader)
            model._metadata.score.sample_weight = _read_request(reader)
        if reader.remaining():
            _check_adapter[Self.Adapter]()
            if reader.string(4096) != Self.Adapter.identifier():
                raise Error(
                    "ModelSchemaError: native kernel identifier mismatch"
                )
            if reader.uint() != UInt64(Self.Adapter.state_version()):
                raise Error(
                    "ModelSchemaError: native kernel state version mismatch"
                )
            var state = reader.bytes(reader.remaining())
            reader.finish()
            var kernel = Self.Adapter.decode(state)
            model.native_kernel = rebind[Self.Metric](kernel).copy()
        reader.finish()
        var inputs = archive.array("X_fit_")
        var coefficients = archive.array("dual_coef_")
        if (
            inputs.dtype < 1
            or inputs.dtype > 4
            or coefficients.dtype < 1
            or coefficients.dtype > 2
        ):
            raise Error("ModelSchemaError: unsupported learned dtype")
        if model._fitted:
            if (
                len(inputs.shape) != 2
                or inputs.shape[0] < 1
                or inputs.shape[1] < 1
                or inputs.shape[1] != model.n_features_in_
            ):
                raise Error(
                    "ModelSchemaError: invalid fitted input shape or feature"
                    " count"
                )
            if len(coefficients.shape) != 1 and len(coefficients.shape) != 2:
                raise Error("ModelSchemaError: invalid coefficient rank")
            if coefficients.shape[0] != inputs.shape[0] or (
                len(coefficients.shape) == 2 and coefficients.shape[1] < 1
            ):
                raise Error(
                    "ModelSchemaError: coefficient shape does not match"
                    " training samples"
                )
        else:
            if model.n_features_in_ != 0:
                raise Error(
                    "ModelSchemaError: unfitted model has learned feature count"
                )
            for payload in archive.payloads:
                if (
                    len(payload.shape) != 1
                    or payload.shape[0] != 0
                    or payload.dtype != dtype_code[Self.dtype]()
                ):
                    raise Error(
                        "ModelSchemaError: unfitted model has learned arrays"
                    )
        comptime types = (
            DType.float32,
            DType.float64,
            DType.int32,
            DType.int64,
        )
        comptime for i in range(4):
            comptime dt = types[i]
            if inputs.dtype == dtype_code[dt]():
                var value = _read_array[dt](archive, "X_fit_")
                if model._fitted:
                    validate(value, "X_fit_", True)
                model.X_fit_ = FittedArray(value^)
        comptime for i in range(2):
            comptime dt = types[i]
            if coefficients.dtype == dtype_code[dt]():
                var value = _read_array[dt](archive, "dual_coef_")
                if model._fitted:
                    validate(value, "dual_coef_")
                model.dual_coef_ = FloatingArray(value^)
        # Settings may have changed after fit, including values invalid for refit.
        # Never infer learned shapes/dtypes from the current kernel or alpha.
        return model^
