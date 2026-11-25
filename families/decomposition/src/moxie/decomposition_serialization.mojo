"""Native state schemas for the runtime dense decomposition estimators."""

from std.math import isfinite, isnan
from numojo import NDArray, Shape, zeros
from .decomposition._pca import PCA, ComponentCount, IteratedPower
from .decomposition._incremental_pca import IncrementalPCA
from .serialization import BinaryReader, BinaryWriter, ModelArchive, ModelSchema, dtype_code


def _write_optional_int(mut writer: BinaryWriter, value: Optional[Int]):
    writer.boolean(Bool(value))
    if value:
        writer.integer(value.value())


def _read_optional_int(mut reader: BinaryReader) raises -> Optional[Int]:
    if reader.boolean():
        return reader.integer()
    return None


def _write_components(mut writer: BinaryWriter, count: ComponentCount) raises:
    if count.isa[Int]():
        writer.uint(1)
        writer.integer(count[Int])
    elif count.isa[Float64]():
        writer.uint(2)
        writer.scalar(count[Float64])
    elif count.isa[String]():
        writer.uint(3)
        writer.string(count[String])
    else:
        writer.uint(0)


def _read_components(mut reader: BinaryReader) raises -> ComponentCount:
    var tag = reader.uint()
    if tag == 0:
        return NoneType()
    if tag == 1:
        return reader.integer()
    if tag == 2:
        return reader.scalar[DType.float64]()
    if tag == 3:
        return reader.string()
    raise Error("ModelSchemaError: invalid n_components tag")


@fieldwise_init
struct _ProjectionState(Movable):
    var fitted: Bool
    var samples: Int
    var features: Int
    var rank: Int
    var noise: Float64
    var names: Optional[List[String]]

    def validate(self, *, minimum_rank: Int) raises:
        if self.fitted:
            if (
                self.samples < 1
                or self.features < 1
                or self.rank < minimum_rank
                or self.rank > min(self.samples, self.features)
                or not (isfinite(self.noise) or (self.samples == 1 and isnan(self.noise)))
            ):
                raise Error("ModelSchemaError: invalid fitted projection metadata")
            if self.names and len(self.names.value()) != self.features:
                raise Error("ModelSchemaError: invalid feature name count")
        elif self.samples or self.features or self.rank or self.noise != 0 or self.names:
            raise Error("ModelSchemaError: unfitted projection has learned metadata")

    def write(self, mut writer: BinaryWriter):
        writer.boolean(self.fitted)
        writer.integer(self.samples)
        writer.integer(self.features)
        writer.integer(self.rank)
        writer.scalar(self.noise)
        writer.boolean(Bool(self.names))
        if self.names:
            writer.uint(UInt64(len(self.names.value())))
            for name in self.names.value():
                writer.string(name)

    @staticmethod
    def read(mut reader: BinaryReader) raises -> Self:
        var fitted = reader.boolean()
        var samples = reader.integer()
        var features = reader.integer()
        var rank = reader.integer()
        var noise = reader.scalar[DType.float64]()
        var names: Optional[List[String]] = None
        if reader.boolean():
            var count = reader.count(reader.remaining() // 8)
            var values = List[String](capacity=count)
            for _ in range(count):
                values.append(reader.string())
            names = values^
        return Self(fitted, samples, features, rank, noise, names^)


def _check_value[dtype: DType](
    name: String, value: Scalar[dtype], allow_nan: Bool, nonnegative: Bool,
) raises:
    if not (isfinite(value) or (allow_nan and isnan(value))) or (nonnegative and value < 0):
        raise Error("ModelSchemaError: invalid value in " + name)


def _write_array[dtype: DType](
    mut archive: ModelArchive, name: String, array: Optional[NDArray[dtype]],
    fitted: Bool, shape: List[Int], *, allow_nan: Bool = False,
) raises:
    if not fitted:
        if array:
            raise Error("ModelSchemaError: unfitted projection has " + name)
        return
    if not array:
        raise Error("ModelSchemaError: missing " + name)
    ref value = array.value()
    if value.ndim != len(shape):
        raise Error("ModelSchemaError: invalid shape for " + name)
    for axis in range(len(shape)):
        if value.shape[axis] != shape[axis]:
            raise Error("ModelSchemaError: invalid shape for " + name)
    var nonnegative = name == "singular_values_" or name == "explained_variance_"
    var values = List[Scalar[dtype]](capacity=value.size)
    for i in range(shape[0]):
        var offset = i * Int(value.strides[0])
        if len(shape) == 1:
            var item = value.unsafe_get(offset)
            _check_value(name, item, allow_nan, nonnegative)
            values.append(item)
        else:
            for j in range(shape[1]):
                var item = value.unsafe_get(offset + j * Int(value.strides[1]))
                _check_value(name, item, allow_nan, nonnegative)
                values.append(item)
    archive.add_array(name, shape, values)


def _read_array[dtype: DType](
    archive: ModelArchive, name: String, fitted: Bool, shape: List[Int],
    *, allow_nan: Bool = False,
) raises -> Optional[NDArray[dtype]]:
    if not fitted:
        return None
    var payload = archive.array(name)
    if payload.shape != shape:
        raise Error("ModelSchemaError: invalid shape for " + name)
    var values = payload.values[dtype]()
    var dimensions = Shape(shape[0])
    if len(shape) == 2:
        dimensions = Shape(shape[0], shape[1])
    var array = zeros[dtype](dimensions)
    var nonnegative = name == "singular_values_" or name == "explained_variance_"
    for i in range(len(values)):
        var value = values[i]
        _check_value(name, value, allow_nan, nonnegative)
        array.unsafe_set(i, value)
    return array^


def _check_dtype[dtype: DType](mut reader: BinaryReader) raises:
    if reader.uint() != UInt64(dtype_code[dtype]()):
        raise Error("ModelSchemaError: projection dtype mismatch")


def _check_arrays(archive: ModelArchive, fitted: Bool, expected: Int) raises:
    if len(archive.payloads) != (expected if fitted else 0):
        raise Error("ModelSchemaError: unexpected projection array count")


def _check_solver(state: _ProjectionState, solver: String) raises:
    if state.fitted:
        if solver not in [String("full"), "covariance_eigh", "arpack", "randomized"]:
            raise Error("ModelSchemaError: invalid fitted PCA solver")
        if (solver == "arpack" or solver == "randomized") and state.rank == 0:
            raise Error("ModelSchemaError: truncated PCA requires a positive learned rank")
        if solver == "arpack" and state.rank == min(state.samples, state.features):
            raise Error("ModelSchemaError: invalid arpack learned rank")
    elif solver != "":
        raise Error("ModelSchemaError: unfitted PCA has a learned solver")


struct PCASchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of the runtime PCA configuration and learned projection."""

    comptime Model = PCA[Self.dtype]
    comptime _type_id = "moxie.decomposition.PCA"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self._type_id, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        _write_components(writer, model.n_components)
        writer.boolean(model.copy_input)
        writer.boolean(model.whiten)
        writer.string(model.svd_solver)
        writer.scalar(model.tol)
        writer.boolean(model.iterated_power.isa[Int]())
        if model.iterated_power.isa[Int]():
            writer.integer(model.iterated_power[Int])
        else:
            writer.string(model.iterated_power[String])
        writer.integer(model.n_oversamples)
        writer.string(model.power_iteration_normalizer)
        _write_optional_int(writer, model.random_state)
        var state = _ProjectionState(
            model.__sklearn_is_fitted__(), model.n_samples_, model.n_features_in_,
            model.n_components_, model.noise_variance_, model.feature_names_in_.copy(),
        )
        state.validate(minimum_rank=0)
        _check_solver(state, model._fit_svd_solver)
        state.write(writer)
        writer.string(model._fit_svd_solver)
        archive.manifest = writer^.finish()
        _write_array(archive, "components_", model.components_, state.fitted, [state.rank, state.features])
        _write_array(archive, "mean_", model.mean_, state.fitted, [state.features])
        _write_array(archive, "singular_values_", model.singular_values_, state.fitted, [state.rank])
        _write_array(archive, "explained_variance_", model.explained_variance_, state.fitted, [state.rank], allow_nan=state.samples == 1)
        _write_array(archive, "explained_variance_ratio_", model.explained_variance_ratio_, state.fitted, [state.rank], allow_nan=True)
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self._type_id, 1)
        archive.validate()
        var reader = BinaryReader(archive.manifest)
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(_read_components(reader))
        model.copy_input = reader.boolean()
        model.whiten = reader.boolean()
        model.svd_solver = reader.string()
        model.tol = reader.scalar[DType.float64]()
        if reader.boolean():
            model.iterated_power = IteratedPower(reader.integer())
        else:
            model.iterated_power = IteratedPower(reader.string())
        model.n_oversamples = reader.integer()
        model.power_iteration_normalizer = reader.string()
        model.random_state = _read_optional_int(reader)
        var state = _ProjectionState.read(reader)
        state.validate(minimum_rank=0)
        model._fit_svd_solver = reader.string()
        _check_solver(state, model._fit_svd_solver)
        reader.finish()
        _check_arrays(archive, state.fitted, 5)
        model.components_ = _read_array[Self.dtype](archive, "components_", state.fitted, [state.rank, state.features])
        model.mean_ = _read_array[Self.dtype](archive, "mean_", state.fitted, [state.features])
        model.singular_values_ = _read_array[Self.dtype](archive, "singular_values_", state.fitted, [state.rank])
        model.explained_variance_ = _read_array[Self.dtype](archive, "explained_variance_", state.fitted, [state.rank], allow_nan=state.samples == 1)
        model.explained_variance_ratio_ = _read_array[Self.dtype](archive, "explained_variance_ratio_", state.fitted, [state.rank], allow_nan=True)
        model.n_samples_ = state.samples
        model.n_features_in_ = state.features
        model.n_components_ = state.rank
        model.noise_variance_ = state.noise
        model.feature_names_in_ = state.names.copy()
        return model^


struct IncrementalPCASchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of the runtime IncrementalPCA state, including running moments."""

    comptime Model = IncrementalPCA[Self.dtype]
    comptime _type_id = "moxie.decomposition.IncrementalPCA"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self._type_id, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        _write_optional_int(writer, model.n_components)
        writer.boolean(model.whiten)
        writer.boolean(model.copy)
        _write_optional_int(writer, model.batch_size)
        var state = _ProjectionState(
            model.__sklearn_is_fitted__(), model.n_samples_seen_, model.n_features_in_,
            model.n_components_, model.noise_variance_, model.feature_names_in_.copy(),
        )
        state.validate(minimum_rank=1)
        if model.batch_size_ < 0 or (not state.fitted and model.batch_size_ != 0):
            raise Error("ModelSchemaError: invalid learned batch size")
        state.write(writer)
        writer.integer(model.batch_size_)
        archive.manifest = writer^.finish()
        _write_array(archive, "components_", model.components_, state.fitted, [state.rank, state.features])
        _write_array(archive, "mean_", model.mean_, state.fitted, [state.features])
        _write_array(archive, "var_", model.var_, state.fitted, [state.features])
        _write_array(archive, "singular_values_", model.singular_values_, state.fitted, [state.rank])
        _write_array(archive, "explained_variance_", model.explained_variance_, state.fitted, [state.rank], allow_nan=state.samples == 1)
        _write_array(archive, "explained_variance_ratio_", model.explained_variance_ratio_, state.fitted, [state.rank], allow_nan=True)
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self._type_id, 1)
        archive.validate()
        var reader = BinaryReader(archive.manifest)
        _check_dtype[Self.dtype](reader)
        var count = _read_optional_int(reader)
        var whiten = reader.boolean()
        var copy_input = reader.boolean()
        var model = Self.Model(count, whiten=whiten, copy=copy_input, batch_size=_read_optional_int(reader))
        var state = _ProjectionState.read(reader)
        state.validate(minimum_rank=1)
        model.batch_size_ = reader.integer()
        if model.batch_size_ < 0 or (not state.fitted and model.batch_size_ != 0):
            raise Error("ModelSchemaError: invalid learned batch size")
        reader.finish()
        _check_arrays(archive, state.fitted, 6)
        model.components_ = _read_array[Self.dtype](archive, "components_", state.fitted, [state.rank, state.features])
        model.mean_ = _read_array[DType.float64](archive, "mean_", state.fitted, [state.features])
        model.var_ = _read_array[DType.float64](archive, "var_", state.fitted, [state.features])
        model.singular_values_ = _read_array[Self.dtype](archive, "singular_values_", state.fitted, [state.rank])
        model.explained_variance_ = _read_array[Self.dtype](archive, "explained_variance_", state.fitted, [state.rank], allow_nan=state.samples == 1)
        model.explained_variance_ratio_ = _read_array[Self.dtype](archive, "explained_variance_ratio_", state.fitted, [state.rank], allow_nan=True)
        model.n_samples_seen_ = state.samples
        model.n_features_in_ = state.features
        model.n_components_ = state.rank
        model.noise_variance_ = state.noise
        model.feature_names_in_ = state.names.copy()
        return model^
