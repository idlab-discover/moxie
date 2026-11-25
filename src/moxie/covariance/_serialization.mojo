"""Experimental schemas for dense covariance estimators."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import isfinite

from moxie.covariance._empirical_covariance import CovarianceModel, EmpiricalCovariance
from moxie.covariance._shrunk_covariance_adaptive import LedoitWolf, OAS
from moxie.covariance._shrunk_covariance_fixed import ShrunkCovariance
from moxie.covariance._state import validate_location, validate_square
from moxie.serialization import BinaryReader, BinaryWriter, ModelArchive, ModelSchema, dtype_code


def _write_array[dtype: DType](
    mut archive: ModelArchive, name: String, array: NDArray[dtype],
) raises:
    var dimensions = List[Int]()
    var values = List[Scalar[dtype]](capacity=array.size)
    dimensions.append(array.shape[0])
    var stride = Int(array.strides[0])
    if array.ndim == 1:
        for i in range(array.shape[0]):
            # NuMojo's flat item() skips strides for rank-one arrays.
            values.append(array.unsafe_get(i * stride))
    else:
        dimensions.append(array.shape[1])
        var column_stride = Int(array.strides[1])
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                values.append(array.unsafe_get(i * stride + j * column_stride))
    archive.add_array[dtype](name, dimensions, values)


def _read_array[dtype: DType](
    archive: ModelArchive, name: String, features: Int, rank: Int,
) raises -> NDArray[dtype]:
    var payload = archive.array(name)
    if len(payload.shape) != rank or payload.shape[0] != features:
        raise Error(String("ModelSchemaError: invalid shape for ", name))
    var shape = Shape(features)
    if rank == 2:
        if payload.shape[1] != features:
            raise Error(String("ModelSchemaError: invalid shape for ", name))
        shape = Shape(features, features)
    var values = payload.values[dtype]()
    var array = nm.zeros[dtype](shape)
    for i in range(len(values)):
        array.unsafe_set(i, values[i])
    return array^


def _validate_widened[dtype: DType](
    public: NDArray[dtype], widened: NDArray[DType.float64], features: Int,
) raises:
    validate_square(widened, features)
    for i in range(features):
        for j in range(features):
            if public.item(i, j) != widened.item(i, j).cast[dtype]():
                raise Error("ModelSchemaError: widened matrix differs from its public snapshot")


def _validate_state[dtype: DType](model: CovarianceModel[dtype]) raises:
    if not model.is_fitted():
        if (model.state.fitted.is_fitted() or model.state.location_
            or model.state.precision_ or model._covariance64 or model._precision64):
            raise Error("ModelSchemaError: unfitted covariance has learned state")
        return
    if not model.state.fitted.is_fitted() or not model.state.location_:
        raise Error("ModelSchemaError: missing fitted covariance metadata")
    var features = model.state.fitted.n_features_in_.value()
    validate_location(model.state.location_.value(), features)
    validate_square(model.state.covariance_.value(), features)
    if model.state.precision_:
        validate_square(model.state.precision_.value(), features)
    if model._covariance64:
        _validate_widened(model.state.covariance_.value(), model._covariance64.value(), features)
        if Bool(model.state.precision_) != Bool(model._precision64):
            raise Error("ModelSchemaError: inconsistent widened precision")
        if model._precision64:
            _validate_widened(model.state.precision_.value(), model._precision64.value(), features)
    elif model._precision64:
        raise Error("ModelSchemaError: widened precision requires widened covariance")


def _write_state[dtype: DType](
    mut archive: ModelArchive, mut writer: BinaryWriter, model: CovarianceModel[dtype],
) raises:
    _validate_state(model)
    writer.boolean(model.is_fitted())
    if not model.is_fitted():
        return
    writer.integer(model.state.fitted.n_features_in_.value())
    writer.boolean(Bool(model.state.precision_))
    writer.boolean(Bool(model._covariance64))
    _write_array(archive, "location_", model.state.location_.value())
    _write_array(archive, "covariance_", model.state.covariance_.value())
    if model.state.precision_:
        _write_array(archive, "precision_", model.state.precision_.value())
    if model._covariance64:
        _write_array(archive, "covariance64", model._covariance64.value())
        if model._precision64:
            _write_array(archive, "precision64", model._precision64.value())


def _read_state[dtype: DType](
    archive: ModelArchive, mut reader: BinaryReader[_], allow_widened: Bool = True,
) raises -> CovarianceModel[dtype]:
    var model = CovarianceModel[dtype]()

    if reader.boolean():
        var features = reader.integer()
        if features < 1:
            raise Error("ModelSchemaError: invalid feature count")
        var stored = reader.boolean()
        var widened = reader.boolean()
        if widened and not allow_widened:
            raise Error("ModelSchemaError: estimator has no widened covariance")
        var expected = 2 + Int(stored) + Int(widened) + Int(widened and stored)
        if len(archive.payloads) != expected:
            raise Error("ModelSchemaError: unexpected covariance arrays")
        model.state.location_ = _read_array[dtype](archive, "location_", features, 1)
        model.state.covariance_ = _read_array[dtype](archive, "covariance_", features, 2)
        if stored:
            model.state.precision_ = _read_array[dtype](archive, "precision_", features, 2)
        if widened:
            model._covariance64 = _read_array[DType.float64](archive, "covariance64", features, 2)
            if stored:
                model._precision64 = _read_array[DType.float64](archive, "precision64", features, 2)
        model.state.fitted.record_fit(1, features)
    elif len(archive.payloads) != 0:
        raise Error("ModelSchemaError: unfitted covariance has arrays")
    _validate_state(model)
    return model^


def _write_config[dtype: DType](store_precision: Bool, assume_centered: Bool) -> BinaryWriter:
    var writer = BinaryWriter()
    writer.uint(UInt64(dtype_code[dtype]()))
    writer.boolean(store_precision)
    writer.boolean(assume_centered)
    return writer^


def _check_dtype[dtype: DType](mut reader: BinaryReader[_]) raises:
    if reader.uint() != UInt64(dtype_code[dtype]()):
        raise Error("ModelSchemaError: covariance dtype mismatch")


def _write_shrinkage[dtype: DType](
    mut writer: BinaryWriter, model: CovarianceModel[dtype], shrinkage: Optional[Float64],
) raises:
    if model._covariance64 or Bool(shrinkage) != model.is_fitted():
        raise Error("ModelSchemaError: inconsistent adaptive shrinkage state")
    if shrinkage:
        if not isfinite(shrinkage.value()):
            raise Error("ModelSchemaError: nonfinite learned shrinkage")
        writer.scalar[DType.float64](shrinkage.value())


def _read_shrinkage(mut reader: BinaryReader[_], fitted: Bool) raises -> Optional[Float64]:
    if not fitted:
        return None
    var shrinkage = reader.scalar[DType.float64]()
    if not isfinite(shrinkage):
        raise Error("ModelSchemaError: nonfinite learned shrinkage")
    return shrinkage


struct EmpiricalCovarianceSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = EmpiricalCovariance[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.EmpiricalCovariance"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = _write_config[Self.dtype](model.store_precision, model.assume_centered)
        _write_state(archive, writer, model._model)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            store_precision=reader.boolean(),
            assume_centered=reader.boolean(),
        )
        model._model = _read_state[Self.dtype](archive, reader)
        reader.finish()
        return model^


struct ShrunkCovarianceSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = ShrunkCovariance[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.ShrunkCovariance"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = _write_config[Self.dtype](model.store_precision, model.assume_centered)
        writer.scalar[DType.float64](model.shrinkage)
        _write_state(archive, writer, model._model)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            store_precision=reader.boolean(),
            assume_centered=reader.boolean(),
            shrinkage=reader.scalar[DType.float64](),
        )
        model._model = _read_state[Self.dtype](archive, reader)
        reader.finish()
        return model^


struct LedoitWolfSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = LedoitWolf[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.LedoitWolf"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = _write_config[Self.dtype](model.store_precision, model.assume_centered)
        writer.integer(model.block_size)
        _write_state(archive, writer, model._model)
        _write_shrinkage(writer, model._model, model._shrinkage)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            store_precision=reader.boolean(),
            assume_centered=reader.boolean(),
            block_size=reader.integer(),
        )
        model._model = _read_state[Self.dtype](archive, reader, allow_widened=False)
        model._shrinkage = _read_shrinkage(reader, model.__sklearn_is_fitted__())
        reader.finish()
        return model^


struct OASSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = OAS[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.OAS"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = _write_config[Self.dtype](model.store_precision, model.assume_centered)
        _write_state(archive, writer, model._model)
        _write_shrinkage(writer, model._model, model._shrinkage)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            store_precision=reader.boolean(),
            assume_centered=reader.boolean(),
        )
        model._model = _read_state[Self.dtype](archive, reader, allow_widened=False)
        model._shrinkage = _read_shrinkage(reader, model.__sklearn_is_fitted__())
        reader.finish()
        return model^
