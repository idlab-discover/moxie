"""Native state schema for dense IsotonicRegression."""

from std.math import isfinite
from moxie.metadata_routing import SampleWeightRequest, SampleWeightUpdate
from moxie.isotonic import Increasing, IsotonicRegression
from moxie.serialization import (
    BinaryReader,
    BinaryWriter,
    ModelArchive,
    ModelSchema,
    dtype_code,
)


def _write_bound(mut writer: BinaryWriter, bound: Optional[Float64]):
    writer.boolean(Bool(bound))
    if bound:
        writer.scalar[DType.float64](bound.or_else(0.0))


def _read_bound(mut reader: BinaryReader[_]) raises -> Optional[Float64]:
    if reader.boolean():
        return reader.scalar[DType.float64]()
    return None


def _write_request(mut writer: BinaryWriter, request: SampleWeightRequest) raises:
    var value = request.value()
    if value.isa[NoneType]():
        writer.uint(0)
    elif value.isa[Bool]():
        writer.uint(UInt64(2 if value[Bool] else 1))
    else:
        writer.uint(3)
        writer.string(value[String])


def _read_request(mut reader: BinaryReader[_]) raises -> SampleWeightUpdate:
    var tag = reader.uint()
    if tag == 0:
        return SampleWeightUpdate(None)
    if tag == 1 or tag == 2:
        return SampleWeightUpdate(tag == 2)
    if tag != 3:
        raise Error("ModelSchemaError: unknown isotonic sample-weight request")

    var name = reader.string(reader.remaining())
    var update = SampleWeightUpdate(name.copy())
    var request = SampleWeightRequest()
    try:
        request.set(update)
    except:
        raise Error("ModelSchemaError: invalid isotonic sample-weight alias")
    # The setter accepts the unchanged sentinel and normalizes the standard
    # name to True. Neither represents a stored alias.
    var value = request.value()
    if not value.isa[String]():
        raise Error("ModelSchemaError: invalid isotonic sample-weight alias")
    return update^


def _read_thresholds[dtype: DType](
    archive: ModelArchive, name: String
) raises -> List[Scalar[dtype]]:
    for payload in archive.payloads:
        if payload.name == name:
            if len(payload.shape) != 1:
                raise Error("ModelSchemaError: thresholds must be one-dimensional")
            return payload.values[dtype]()
    raise Error("ModelSchemaError: missing array " + name)


def _validate_state[dtype: DType](model: IsotonicRegression[dtype]) raises:
    var count = len(model.X_thresholds_)
    if not model.__sklearn_is_fitted__():
        if (
            count != 0
            or len(model.y_thresholds_) != 0
            or model.X_min_ != 0
            or model.X_max_ != 0
            or not model.increasing_
        ):
            raise Error("ModelSchemaError: unfitted isotonic model has learned state")
        return
    if count == 0 or count != len(model.y_thresholds_):
        raise Error("ModelSchemaError: invalid isotonic threshold lengths")
    if (
        model.X_min_ != model.X_thresholds_[0]
        or model.X_max_ != model.X_thresholds_[count - 1]
    ):
        raise Error("ModelSchemaError: domain must match threshold endpoints")
    for i in range(count):
        var x = model.X_thresholds_[i]
        var y = model.y_thresholds_[i]
        if not isfinite(x) or not isfinite(y):
            raise Error("ModelSchemaError: thresholds must be finite")
        if i == 0:
            continue
        if x <= model.X_thresholds_[i - 1]:
            raise Error("ModelSchemaError: X thresholds must be strictly ascending")
        var previous_y = model.y_thresholds_[i - 1]
        if (model.increasing_ and y < previous_y) or (
            not model.increasing_ and y > previous_y
        ):
            raise Error("ModelSchemaError: y thresholds contradict fitted direction")


struct IsotonicRegressionSchema[dtype: DType = DType.float64](ModelSchema):
    """Version 2 state for fitted and unfitted Float32 or Float64 models."""

    comptime Model = IsotonicRegression[Self.dtype]
    comptime TypeId = "moxie.isotonic.IsotonicRegression"
    comptime StateVersion = 2

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        _validate_state(model)
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        _write_bound(writer, model.y_min)
        _write_bound(writer, model.y_max)
        writer.boolean(model.increasing.isa[Bool]())
        if model.increasing.isa[Bool]():
            writer.boolean(model.increasing[Bool])
        else:
            writer.string(model.increasing[String])
        writer.string(model.out_of_bounds)
        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        if fitted:
            writer.scalar[Self.dtype](model.X_min_)
            writer.scalar[Self.dtype](model.X_max_)
            writer.boolean(model.increasing_)
            var shape: List[Int] = [len(model.X_thresholds_)]
            archive.add_array[Self.dtype]("X_thresholds_", shape, model.X_thresholds_)
            archive.add_array[Self.dtype]("y_thresholds_", shape, model.y_thresholds_)
        var routing = model.get_metadata_routing()
        _write_request(writer, routing.fit)
        _write_request(writer, routing.score)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var legacy = archive.schema_version == 1
        archive.expect(Self.TypeId, 1 if legacy else Self.StateVersion)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != UInt64(dtype_code[Self.dtype]()):
            raise Error("ModelSchemaError: isotonic dtype mismatch")
        var y_min = _read_bound(reader)
        var y_max = _read_bound(reader)
        var increasing: Increasing
        if reader.boolean():
            increasing = Increasing(reader.boolean())
        else:
            increasing = Increasing(reader.string(reader.remaining()))
        var model = Self.Model(
            y_min=y_min,
            y_max=y_max,
            increasing=increasing^,
            out_of_bounds=reader.string(reader.remaining()),
        )
        var fitted = reader.boolean()
        if len(archive.payloads) != (2 if fitted else 0):
            raise Error("ModelSchemaError: unexpected number of isotonic arrays")
        if fitted:
            model.X_min_ = reader.scalar[Self.dtype]()
            model.X_max_ = reader.scalar[Self.dtype]()
            model.increasing_ = reader.boolean()
        if not legacy:
            var fit_request = _read_request(reader)
            var score_request = _read_request(reader)
            _ = model.set_fit_request(sample_weight=fit_request).set_score_request(
                sample_weight=score_request
            )
        reader.finish()
        if fitted:
            model.X_thresholds_ = _read_thresholds[Self.dtype](archive, "X_thresholds_")
            model.y_thresholds_ = _read_thresholds[Self.dtype](archive, "y_thresholds_")
            model._fitted = True
            # sklearn.__setstate__ rebuilds the interpolator using current configuration.
            model._bounds_error = model.out_of_bounds == "raise"
        _validate_state(model)
        return model^
