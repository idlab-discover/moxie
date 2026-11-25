"""Native state schemas for the runtime dense decomposition estimators."""

from std.math import isfinite, isinf, isnan
from numojo import NDArray, Shape, array, zeros
from .decomposition._pca import PCA, ComponentCount, IteratedPower
from .decomposition._incremental_pca import IncrementalPCA
from .decomposition._truncated_svd import TruncatedSVD
from .decomposition._factor_analysis import FactorAnalysis
from .decomposition._fastica import FastICA, WhitenOption
from .decomposition._kernel_pca import KernelPCA, KernelPower
from .decomposition._lda import LatentDirichletAllocation, LatentDirichletAllocationParams
from .decomposition._sparse_coding import SparseCoder
from .decomposition._nmf import (
    AlphaH, BetaLoss, MiniBatchNMF, MiniBatchNMFParams, NMF, NMFComponents,
    NMFConfig, NMFParams, _gamma,
)
from .serialization import ArrayPayload, BinaryReader, BinaryWriter, ModelArchive, ModelSchema, dtype_code
from ._random import RandomState


def _write_optional_int(mut writer: BinaryWriter, value: Optional[Int]):
    writer.boolean(Bool(value))
    if value:
        writer.integer(value.value())


def _read_optional_int(mut reader: BinaryReader) raises -> Optional[Int]:
    if reader.boolean():
        return reader.integer()
    return None


def _write_optional_float(mut writer: BinaryWriter, value: Optional[Float64]):
    writer.boolean(Bool(value))
    if value:
        writer.scalar(value.value())


def _read_optional_float(mut reader: BinaryReader) raises -> Optional[Float64]:
    if reader.boolean():
        return reader.scalar[DType.float64]()
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
    name: String, value: Scalar[dtype], allow_nan: Bool, allow_positive_inf: Bool,
    nonnegative: Bool, positive: Bool,
) raises:
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise Error("ModelSchemaError: invalid value in " + name)
    if not (
        isfinite(value) or (allow_nan and isnan(value))
        or (allow_positive_inf and isinf(value) and value > 0)
    ):
        raise Error("ModelSchemaError: invalid value in " + name)


def _write_array[dtype: DType](
    mut archive: ModelArchive, name: String, array: Optional[NDArray[dtype]],
    fitted: Bool, shape: List[Int], *, allow_nan: Bool = False,
    allow_positive_inf: Bool = False, nonnegative: Bool = False,
) raises:
    if not fitted:
        if array:
            raise Error("ModelSchemaError: unfitted projection has " + name)
        return
    if not array:
        raise Error("ModelSchemaError: missing " + name)
    _write_dense_array(archive, name, array.value(), shape, allow_nan=allow_nan,
        allow_positive_inf=allow_positive_inf, nonnegative=nonnegative)


def _write_dense_array[dtype: DType](
    mut archive: ModelArchive, name: String, value: NDArray[dtype], shape: List[Int],
    *, allow_nan: Bool = False, allow_positive_inf: Bool = False, nonnegative: Bool = False,
) raises:
    if value.ndim != len(shape):
        raise Error("ModelSchemaError: invalid shape for " + name)
    for axis in range(len(shape)):
        if value.shape[axis] != shape[axis]:
            raise Error("ModelSchemaError: invalid shape for " + name)
    var require_nonnegative = nonnegative or name in (
        "singular_values_", "explained_variance_", "eigenvalues_"
    )
    var positive = name == "noise_variance_"
    var values = List[Scalar[dtype]](capacity=value.size)
    for i in range(shape[0]):
        var offset = i * Int(value.strides[0])
        if len(shape) == 1:
            var item = value.unsafe_get(offset)
            _check_value(name, item, allow_nan, allow_positive_inf, require_nonnegative, positive)
            values.append(item)
        else:
            for j in range(shape[1]):
                var item = value.unsafe_get(offset + j * Int(value.strides[1]))
                _check_value(name, item, allow_nan, allow_positive_inf, require_nonnegative, positive)
                values.append(item)
    archive.add_array(name, shape, values)


def _read_array[dtype: DType](
    archive: ModelArchive, name: String, fitted: Bool, shape: List[Int],
    *, allow_nan: Bool = False, allow_positive_inf: Bool = False,
    nonnegative: Bool = False,
) raises -> Optional[NDArray[dtype]]:
    if not fitted:
        return None
    var payload = archive.array(name)
    if payload.shape != shape:
        raise Error("ModelSchemaError: invalid shape for " + name)
    return _decode_dense_array[dtype](payload, allow_nan=allow_nan,
        allow_positive_inf=allow_positive_inf, nonnegative=nonnegative)


def _decode_dense_array[dtype: DType](
    payload: ArrayPayload, *, allow_nan: Bool = False, allow_positive_inf: Bool = False,
    nonnegative: Bool = False,
) raises -> NDArray[dtype]:
    ref shape = payload.shape
    ref name = payload.name
    var values = payload.values[dtype]()
    var dimensions = Shape(shape[0])
    if len(shape) == 2:
        dimensions = Shape(shape[0], shape[1])
    var array = zeros[dtype](dimensions)
    var require_nonnegative = nonnegative or name in (
        "singular_values_", "explained_variance_", "eigenvalues_"
    )
    var positive = name == "noise_variance_"
    for i in range(len(values)):
        var value = values[i]
        _check_value(name, value, allow_nan, allow_positive_inf, require_nonnegative, positive)
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


@fieldwise_init
struct _ComponentState(Movable):
    var fitted: Bool
    var features: Int
    var rank: Int

    def validate(self, *, minimum_features: Int, minimum_rank: Int) raises:
        if self.fitted:
            if (
                self.features < minimum_features
                or self.rank < minimum_rank
                or self.rank > self.features
            ):
                raise Error("ModelSchemaError: invalid fitted component dimensions")
        elif self.features or self.rank:
            raise Error("ModelSchemaError: unfitted projection has learned dimensions")

    def write(self, mut writer: BinaryWriter):
        writer.boolean(self.fitted)
        writer.integer(self.features)
        writer.integer(self.rank)

    @staticmethod
    def read(mut reader: BinaryReader) raises -> Self:
        return Self(reader.boolean(), reader.integer(), reader.integer())


def _component_state[dtype: DType](
    components: Optional[NDArray[dtype]], features: Int,
) raises -> _ComponentState:
    if not components:
        return _ComponentState(False, features, 0)
    ref matrix = components.value()
    if matrix.ndim != 2:
        raise Error("ModelSchemaError: invalid shape for components_")
    return _ComponentState(True, features, matrix.shape[0])


struct TruncatedSVDSchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of the dense TruncatedSVD configuration and learned projection."""

    comptime Model = TruncatedSVD[Self.dtype]
    comptime _type_id = "moxie.decomposition.TruncatedSVD"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self._type_id, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.integer(model.n_components)
        writer.string(model.algorithm)
        writer.integer(model.n_iter)
        writer.integer(model.n_oversamples)
        writer.string(model.power_iteration_normalizer)
        _write_optional_int(writer, model.random_state)
        writer.scalar(model.tol)
        var state = _component_state(model.components_, model.n_features_in_)
        state.validate(minimum_features=2, minimum_rank=1)
        state.write(writer)
        archive.manifest = writer^.finish()
        _write_array(archive, "components_", model.components_, state.fitted, [state.rank, state.features])
        _write_array(archive, "singular_values_", model.singular_values_, state.fitted, [state.rank])
        _write_array(archive, "explained_variance_", model.explained_variance_, state.fitted, [state.rank])
        _write_array(
            archive, "explained_variance_ratio_", model.explained_variance_ratio_,
            state.fitted, [state.rank], allow_nan=True, allow_positive_inf=True,
        )
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self._type_id, 1)
        archive.validate()
        var reader = BinaryReader(archive.manifest)
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            reader.integer(), algorithm=reader.string(), n_iter=reader.integer(),
            n_oversamples=reader.integer(), power_iteration_normalizer=reader.string(),
            random_state=_read_optional_int(reader), tol=reader.scalar[DType.float64](),
        )
        var state = _ComponentState.read(reader)
        state.validate(minimum_features=2, minimum_rank=1)
        reader.finish()
        _check_arrays(archive, state.fitted, 4)
        model.components_ = _read_array[Self.dtype](archive, "components_", state.fitted, [state.rank, state.features])
        model.singular_values_ = _read_array[Self.dtype](archive, "singular_values_", state.fitted, [state.rank])
        model.explained_variance_ = _read_array[Self.dtype](archive, "explained_variance_", state.fitted, [state.rank])
        model.explained_variance_ratio_ = _read_array[Self.dtype](
            archive, "explained_variance_ratio_", state.fitted, [state.rank],
            allow_nan=True, allow_positive_inf=True,
        )
        model.n_features_in_ = state.features
        return model^


def _write_float64_list(mut writer: BinaryWriter, values: List[Float64]):
    writer.uint(UInt64(len(values)))
    for value in values:
        writer.scalar(value)


def _read_float64_list(mut reader: BinaryReader) raises -> List[Float64]:
    var count = reader.count(reader.remaining() // 8)
    var values = List[Float64](capacity=count)
    for _ in range(count):
        values.append(reader.scalar[DType.float64]())
    return values^


def _check_loglike(fitted: Bool, iterations: Int, history: List[Float64]) raises:
    if (iterations < 1 if fitted else iterations != 0) or len(history) != iterations:
        raise Error("ModelSchemaError: invalid FactorAnalysis likelihood history length")
    for value in history:
        # A zero singular value contributes log(0) to the unscaled objective.
        _check_value("loglike_", value, False, True, False, False)


struct FactorAnalysisSchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of the dense FactorAnalysis state with Float64 learned arrays."""

    comptime Model = FactorAnalysis[Self.dtype]
    comptime _type_id = "moxie.decomposition.FactorAnalysis"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self._type_id, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        _write_optional_int(writer, model.n_components)
        writer.scalar(model.tol)
        writer.boolean(model.copy)
        writer.integer(model.max_iter)
        writer.boolean(Bool(model.noise_variance_init))
        if model.noise_variance_init:
            _write_float64_list(writer, model.noise_variance_init.value())
        writer.string(model.svd_method)
        writer.integer(model.iterated_power)
        writer.boolean(Bool(model.rotation))
        if model.rotation:
            writer.string(model.rotation.value())
        _write_optional_int(writer, model.random_state)
        var state = _component_state(model.components_, model.n_features_in_)
        state.validate(minimum_features=1, minimum_rank=0)
        state.write(writer)
        _check_loglike(state.fitted, model.n_iter_, model.loglike_)
        writer.integer(model.n_iter_)
        _write_float64_list(writer, model.loglike_)
        archive.manifest = writer^.finish()
        _write_array(archive, "components_", model.components_, state.fitted, [state.rank, state.features])
        _write_array(archive, "noise_variance_", model.noise_variance_, state.fitted, [state.features])
        _write_array(archive, "mean_", model.mean_, state.fitted, [state.features])
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self._type_id, 1)
        archive.validate()
        var reader = BinaryReader(archive.manifest)
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(_read_optional_int(reader))
        model.tol = reader.scalar[DType.float64]()
        model.copy = reader.boolean()
        model.max_iter = reader.integer()
        if reader.boolean():
            model.noise_variance_init = _read_float64_list(reader)
        model.svd_method = reader.string()
        model.iterated_power = reader.integer()
        if reader.boolean():
            model.rotation = reader.string()
        model.random_state = _read_optional_int(reader)
        var state = _ComponentState.read(reader)
        state.validate(minimum_features=1, minimum_rank=0)
        model.n_iter_ = reader.integer()
        model.loglike_ = _read_float64_list(reader)
        _check_loglike(state.fitted, model.n_iter_, model.loglike_)
        reader.finish()
        _check_arrays(archive, state.fitted, 3)
        model.components_ = _read_array[DType.float64](archive, "components_", state.fitted, [state.rank, state.features])
        model.noise_variance_ = _read_array[DType.float64](archive, "noise_variance_", state.fitted, [state.features])
        model.mean_ = _read_array[DType.float64](archive, "mean_", state.fitted, [state.features])
        model.n_features_in_ = state.features
        return model^


def _write_float_params(mut writer: BinaryWriter, params: Optional[Dict[String, Float64]]):
    writer.boolean(Bool(params))
    if params:
        ref values = params.value()
        writer.uint(UInt64(len(values)))
        for entry in values.items():
            writer.string(entry.key)
            writer.scalar(entry.value)


def _read_float_params(mut reader: BinaryReader) raises -> Optional[Dict[String, Float64]]:
    if not reader.boolean():
        return None
    var count = reader.count(reader.remaining() // 16)
    var params = Dict[String, Float64]()
    for _ in range(count):
        var key = reader.string()
        if key in params:
            raise Error("ModelSchemaError: duplicate numeric parameter " + key)
        params[key] = reader.scalar[DType.float64]()
    return params^


def _write_ica_initial_weights(mut archive: ModelArchive, initial: NDArray[DType.float64]) raises:
    var shape = List[Int](capacity=initial.ndim)
    for axis in range(initial.ndim):
        shape.append(initial.shape[axis])
    var values = List[Float64](capacity=initial.size)
    # Configuration can contain any array accepted by the constructor.
    # Materialize views in logical row order, preserving scalar bits.
    if initial.is_c_contiguous():
        for i in range(initial.size):
            values.append(initial.unsafe_get(i))
    elif initial.size:
        var dense = initial.contiguous()
        for i in range(dense.size):
            values.append(dense.unsafe_get(i))
    archive.add_array("w_init", shape, values)


def _read_ica_initial_weights(archive: ModelArchive) raises -> NDArray[DType.float64]:
    var payload = archive.array("w_init")
    var values = payload.values[DType.float64]()
    if not payload.shape:
        var scalar = array(values, [1])
        return scalar[0]
    return array(values, payload.shape)


def _check_ica_state(state: _ComponentState, iterations: Int, centered: Bool) raises:
    state.validate(minimum_features=1, minimum_rank=1)
    if state.fitted:
        if iterations < 1 or (not centered and state.rank != state.features):
            raise Error("ModelSchemaError: invalid fitted FastICA metadata")
    elif iterations or centered:
        raise Error("ModelSchemaError: unfitted FastICA has learned metadata")


struct FastICASchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of dense FastICA state, including the private unmixing matrix."""

    comptime Model = FastICA[Self.dtype]
    comptime _type_id = "moxie.decomposition.FastICA"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self._type_id, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        _write_optional_int(writer, model.n_components)
        writer.string(model.algorithm)
        writer.boolean(model.whiten.isa[Bool]())
        if model.whiten.isa[Bool]():
            writer.boolean(model.whiten[Bool])
        else:
            writer.string(model.whiten[String])
        writer.string(model.fun)
        _write_float_params(writer, model.fun_args)
        writer.integer(model.max_iter)
        writer.scalar(model.tol)
        writer.boolean(Bool(model.w_init))
        if model.w_init:
            _write_ica_initial_weights(archive, model.w_init.value())
        writer.string(model.whiten_solver)
        _write_optional_int(writer, model.random_state)

        var state = _component_state(model.components_, model.n_features_in_)
        var centered = Bool(model.mean_)
        _check_ica_state(state, model.n_iter_, centered)
        state.write(writer)
        writer.integer(model.n_iter_)
        writer.boolean(centered)
        archive.manifest = writer^.finish()
        _write_array(archive, "components_", model.components_, state.fitted, [state.rank, state.features])
        _write_array(archive, "mixing_", model.mixing_, state.fitted, [state.features, state.rank])
        _write_array(archive, "_unmixing", model._unmixing, state.fitted, [state.rank, state.rank])
        _write_array(archive, "mean_", model.mean_, centered, [state.features])
        _write_array(archive, "whitening_", model.whitening_, centered, [state.rank, state.features])
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self._type_id, 1)
        archive.validate()
        var reader = BinaryReader(archive.manifest)
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(_read_optional_int(reader))
        model.algorithm = reader.string()
        if reader.boolean():
            model.whiten = WhitenOption(reader.boolean())
        else:
            model.whiten = WhitenOption(reader.string())
        model.fun = reader.string()
        model.fun_args = _read_float_params(reader)
        model.max_iter = reader.integer()
        model.tol = reader.scalar[DType.float64]()
        var has_init = reader.boolean()
        if has_init:
            model.w_init = _read_ica_initial_weights(archive)
        model.whiten_solver = reader.string()
        model.random_state = _read_optional_int(reader)
        var state = _ComponentState.read(reader)
        model.n_iter_ = reader.integer()
        var centered = reader.boolean()
        reader.finish()
        _check_ica_state(state, model.n_iter_, centered)
        if len(archive.payloads) != Int(has_init) + 3 * Int(state.fitted) + 2 * Int(centered):
            raise Error("ModelSchemaError: unexpected FastICA array count")
        model.n_features_in_ = state.features
        model.components_ = _read_array[Self.dtype](archive, "components_", state.fitted, [state.rank, state.features])
        model.mixing_ = _read_array[Self.dtype](archive, "mixing_", state.fitted, [state.features, state.rank])
        model._unmixing = _read_array[Self.dtype](archive, "_unmixing", state.fitted, [state.rank, state.rank])
        model.mean_ = _read_array[Self.dtype](archive, "mean_", centered, [state.features])
        model.whitening_ = _read_array[Self.dtype](archive, "whitening_", centered, [state.rank, state.features])
        return model^


@fieldwise_init
struct _KernelPCAState(Movable):
    var fitted: Bool
    var samples: Int
    var features: Int
    var rank: Int
    var gamma: Float64
    var kernel_mean: Float64
    var has_inverse: Bool

    def validate(self) raises:
        if self.fitted:
            if (
                self.samples < 1 or self.features < 1
                or self.rank < 0 or self.rank > self.samples
                or not isfinite(self.gamma) or self.gamma < 0 or not isfinite(self.kernel_mean)
            ):
                raise Error("ModelSchemaError: invalid fitted KernelPCA metadata")
        elif self.samples or self.features or self.rank or self.gamma != 0 or self.kernel_mean != 0 or self.has_inverse:
            raise Error("ModelSchemaError: unfitted KernelPCA has learned metadata")

    def write(self, mut writer: BinaryWriter):
        writer.boolean(self.fitted)
        writer.integer(self.samples)
        writer.integer(self.features)
        writer.integer(self.rank)
        writer.scalar(self.gamma)
        writer.scalar(self.kernel_mean)
        writer.boolean(self.has_inverse)

    @staticmethod
    def read(mut reader: BinaryReader) raises -> Self:
        return Self(reader.boolean(), reader.integer(), reader.integer(), reader.integer(),
            reader.scalar[DType.float64](), reader.scalar[DType.float64](), reader.boolean())


struct KernelPCASchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of dense KernelPCA state, including centering and inverse maps."""

    comptime Model = KernelPCA[Self.dtype]
    comptime _type_id = "moxie.decomposition.KernelPCA"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self._type_id, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        _write_optional_int(writer, model.n_components)
        writer.string(model.kernel)
        writer.boolean(Bool(model.gamma))
        if model.gamma:
            writer.scalar(model.gamma.value())
        writer.scalar(model.degree)
        writer.scalar(model.coef0)
        _write_float_params(writer, model.kernel_params)
        writer.scalar(model.alpha)
        writer.boolean(model.fit_inverse_transform)
        writer.string(model.eigen_solver)
        writer.scalar(model.tol)
        _write_optional_int(writer, model.max_iter)
        writer.boolean(model.iterated_power.isa[Int]())
        if model.iterated_power.isa[Int]():
            writer.integer(model.iterated_power[Int])
        else:
            writer.string(model.iterated_power[String])
        writer.boolean(model.remove_zero_eig)
        _write_optional_int(writer, model.random_state)
        writer.boolean(model.copy_X)
        _write_optional_int(writer, model.n_jobs)

        var samples = 0
        if model.X_fit_:
            ref training = model.X_fit_.value()
            if training.ndim != 2:
                raise Error("ModelSchemaError: invalid shape for X_fit_")
            samples = training.shape[0]
        var rank = 0
        if model.eigenvalues_:
            rank = model.eigenvalues_.value().size
        var state = _KernelPCAState(Bool(model.eigenvalues_), samples, model.n_features_in_,
            rank, model.gamma_, model._kernel_all, Bool(model.dual_coef_))
        state.validate()
        state.write(writer)
        archive.manifest = writer^.finish()
        _write_array(archive, "eigenvalues_", model.eigenvalues_, state.fitted, [state.rank])
        _write_array(archive, "eigenvectors_", model.eigenvectors_, state.fitted, [state.samples, state.rank])
        _write_array(archive, "X_fit_", model.X_fit_, state.fitted, [state.samples, state.features])
        _write_array(archive, "_kernel_mean", model._kernel_mean, state.fitted, [state.samples])
        _write_array(archive, "X_transformed_fit_", model.X_transformed_fit_, state.has_inverse, [state.samples, state.rank])
        _write_array(archive, "dual_coef_", model.dual_coef_, state.has_inverse, [state.samples, state.features])
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self._type_id, 1)
        archive.validate()
        var reader = BinaryReader(archive.manifest)
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(_read_optional_int(reader))
        model.kernel = reader.string()
        if reader.boolean():
            model.gamma = reader.scalar[DType.float64]()
        model.degree = reader.scalar[DType.float64]()
        model.coef0 = reader.scalar[DType.float64]()
        model.kernel_params = _read_float_params(reader)
        model.alpha = reader.scalar[DType.float64]()
        model.fit_inverse_transform = reader.boolean()
        model.eigen_solver = reader.string()
        model.tol = reader.scalar[DType.float64]()
        model.max_iter = _read_optional_int(reader)
        if reader.boolean():
            model.iterated_power = KernelPower(reader.integer())
        else:
            model.iterated_power = KernelPower(reader.string())
        model.remove_zero_eig = reader.boolean()
        model.random_state = _read_optional_int(reader)
        model.copy_X = reader.boolean()
        model.n_jobs = _read_optional_int(reader)
        var state = _KernelPCAState.read(reader)
        reader.finish()
        state.validate()
        _check_arrays(archive, state.fitted, 4 + 2 * Int(state.has_inverse))
        model.n_features_in_ = state.features
        model.gamma_ = state.gamma
        model._kernel_all = state.kernel_mean
        model.eigenvalues_ = _read_array[Self.dtype](archive, "eigenvalues_", state.fitted, [state.rank])
        model.eigenvectors_ = _read_array[Self.dtype](archive, "eigenvectors_", state.fitted, [state.samples, state.rank])
        model.X_fit_ = _read_array[Self.dtype](archive, "X_fit_", state.fitted, [state.samples, state.features])
        model._kernel_mean = _read_array[Self.dtype](archive, "_kernel_mean", state.fitted, [state.samples])
        model.X_transformed_fit_ = _read_array[Self.dtype](archive, "X_transformed_fit_", state.has_inverse, [state.samples, state.rank])
        model.dual_coef_ = _read_array[Self.dtype](archive, "dual_coef_", state.has_inverse, [state.samples, state.features])
        return model^


def _write_nmf_config(mut writer: BinaryWriter, config: NMFConfig) raises:
    if config.n_components.isa[Int]():
        writer.uint(1)
        writer.integer(config.n_components[Int])
    elif config.n_components.isa[String]():
        writer.uint(2)
        writer.string(config.n_components[String])
    else:
        writer.uint(0)
    writer.boolean(Bool(config.init))
    if config.init:
        writer.string(config.init.value())
    writer.boolean(config.beta_loss.isa[String]())
    if config.beta_loss.isa[String]():
        writer.string(config.beta_loss[String])
    else:
        writer.scalar(config.beta_loss[Float64])
    writer.scalar(config.tol)
    writer.integer(config.max_iter)
    _write_optional_int(writer, config.random_state)
    writer.scalar(config.alpha_W)
    writer.boolean(config.alpha_H.isa[String]())
    if config.alpha_H.isa[String]():
        writer.string(config.alpha_H[String])
    else:
        writer.scalar(config.alpha_H[Float64])
    writer.scalar(config.l1_ratio)
    writer.integer(config.verbose)


def _read_nmf_config(mut reader: BinaryReader) raises -> NMFConfig:
    var components: NMFComponents
    var tag = reader.uint()
    if tag == 0:
        components = NoneType()
    elif tag == 1:
        components = reader.integer()
    elif tag == 2:
        components = reader.string()
    else:
        raise Error("ModelSchemaError: invalid NMF n_components tag")
    var init: Optional[String] = None
    if reader.boolean():
        init = reader.string()
    var beta: BetaLoss
    if reader.boolean():
        beta = reader.string()
    else:
        beta = reader.scalar[DType.float64]()
    var tol = reader.scalar[DType.float64]()
    var max_iter = reader.integer()
    var seed = _read_optional_int(reader)
    var alpha_w = reader.scalar[DType.float64]()
    var alpha_h: AlphaH
    if reader.boolean():
        alpha_h = reader.string()
    else:
        alpha_h = reader.scalar[DType.float64]()
    var l1_ratio = reader.scalar[DType.float64]()
    var verbose = reader.integer()
    return NMFConfig(components^, init, beta^, tol, max_iter, seed, alpha_w,
        alpha_h^, l1_ratio, verbose)


@fieldwise_init
struct _NMFState(Movable):
    var dimensions: _ComponentState
    var iterations: Int
    var error: Float64

    def validate(self, *, online: Bool) raises:
        if self.dimensions.fitted:
            # NMF permits more components than input features.
            if (
                self.dimensions.rank < 1 or self.dimensions.features < 1
                or self.iterations < Int(not online) or self.error < 0
                or not isfinite(self.error)
            ):
                raise Error("ModelSchemaError: invalid fitted NMF metadata")
        elif self.dimensions.rank or self.dimensions.features or self.iterations or self.error != 0:
            raise Error("ModelSchemaError: unfitted NMF has learned metadata")

    def write(self, mut writer: BinaryWriter):
        self.dimensions.write(writer)
        writer.integer(self.iterations)
        writer.scalar(self.error)

    @staticmethod
    def read(mut reader: BinaryReader) raises -> Self:
        return Self(_ComponentState.read(reader), reader.integer(), reader.scalar[DType.float64]())

    def shape(self) -> List[Int]:
        return [self.dimensions.rank, self.dimensions.features]


struct NMFSchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of dense NMF configuration and learned state."""

    comptime Model = NMF[Self.dtype]
    comptime _type_id = "moxie.decomposition.NMF"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self._type_id, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        _write_nmf_config(writer, model.get_params().config)
        writer.string(model.solver)
        writer.boolean(model.shuffle)
        var state = _NMFState(_ComponentState(model.__sklearn_is_fitted__(),
            model.n_features_in_, model.n_components_), model.n_iter_, model.reconstruction_err_)
        state.validate(online=False)
        state.write(writer)
        archive.manifest = writer^.finish()
        _write_array(archive, "components_", model.components_, state.dimensions.fitted,
            state.shape(), nonnegative=True)
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self._type_id, 1)
        archive.validate()
        var reader = BinaryReader(archive.manifest)
        _check_dtype[Self.dtype](reader)
        var params = NMFParams(_read_nmf_config(reader), reader.string(), reader.boolean())
        var state = _NMFState.read(reader)
        reader.finish()
        state.validate(online=False)
        _check_arrays(archive, state.dimensions.fitted, 1)
        var model = Self.Model()
        _ = model.set_params(params)
        model.n_features_in_ = state.dimensions.features
        model.n_components_ = state.dimensions.rank
        model.n_iter_ = state.iterations
        model.reconstruction_err_ = state.error
        model.components_ = _read_array[Self.dtype](archive, "components_",
            state.dimensions.fitted, state.shape(), nonnegative=True)
        return model^


def _check_minibatch_nmf_state[dtype: DType](model: MiniBatchNMF[dtype], fitted: Bool) raises:
    if fitted:
        if (
            model.n_steps_ < 1 or model.n_steps_ < model.n_iter_ or model._k != model.n_components_
            or not isfinite(model._beta) or model._gamma != _gamma(model._beta)
            or not (model._rho >= 0 and model._rho <= 1) or model._transform_max_iter < 1
        ):
            raise Error("ModelSchemaError: invalid fitted MiniBatchNMF metadata")
    elif model.n_steps_:
        raise Error("ModelSchemaError: unfitted MiniBatchNMF has learned steps")


struct MiniBatchNMFSchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of dense MiniBatchNMF, including the online accumulators."""

    comptime Model = MiniBatchNMF[Self.dtype]
    comptime _type_id = "moxie.decomposition.MiniBatchNMF"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self._type_id, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        _write_nmf_config(writer, model.get_params().config)
        writer.integer(model.batch_size)
        _write_optional_int(writer, model.max_no_improvement)
        writer.scalar(model.forget_factor)
        writer.boolean(model.fresh_restarts)
        writer.integer(model.fresh_restarts_max_iter)
        _write_optional_int(writer, model.transform_max_iter)

        var fitted = model.__sklearn_is_fitted__()
        var state = _NMFState(_ComponentState(fitted, model.n_features_in_, model.n_components_),
            model.n_iter_, model.reconstruction_err_)
        state.validate(online=True)
        _check_minibatch_nmf_state(model, fitted)
        state.write(writer)
        if fitted:
            writer.integer(model.n_steps_)
            writer.scalar(model._beta)
            writer.scalar(model._gamma)
            writer.scalar(model._rho)
            writer.integer(model._transform_max_iter)
        archive.manifest = writer^.finish()
        var shape = state.shape()
        _write_array(archive, "components_", model.components_, fitted, shape, nonnegative=True)
        _write_array(archive, "_components_numerator", model._components_numerator, fitted, shape,
            nonnegative=True)
        _write_array(archive, "_components_denominator", model._components_denominator, fitted, shape,
            nonnegative=True)
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self._type_id, 1)
        archive.validate()
        var reader = BinaryReader(archive.manifest)
        _check_dtype[Self.dtype](reader)
        var params = MiniBatchNMFParams(_read_nmf_config(reader), reader.integer(),
            _read_optional_int(reader), reader.scalar[DType.float64](), reader.boolean(),
            reader.integer(), _read_optional_int(reader))
        var state = _NMFState.read(reader)
        var fitted = state.dimensions.fitted
        state.validate(online=True)
        _check_arrays(archive, fitted, 3)
        var model = Self.Model(max_iter=params.config.max_iter)
        _ = model.set_params(params)
        if fitted:
            model._k = state.dimensions.rank
            model.n_steps_ = reader.integer()
            model._beta = reader.scalar[DType.float64]()
            model._gamma = reader.scalar[DType.float64]()
            model._rho = reader.scalar[DType.float64]()
            model._transform_max_iter = reader.integer()
        reader.finish()

        model.n_features_in_ = state.dimensions.features
        model.n_components_ = state.dimensions.rank
        model.n_iter_ = state.iterations
        model.reconstruction_err_ = state.error
        _check_minibatch_nmf_state(model, fitted)
        var shape = state.shape()
        model.components_ = _read_array[Self.dtype](archive, "components_", fitted, shape,
            nonnegative=True)
        model._components_numerator = _read_array[Self.dtype](archive, "_components_numerator",
            fitted, shape, nonnegative=True)
        model._components_denominator = _read_array[Self.dtype](archive, "_components_denominator",
            fitted, shape, nonnegative=True)
        return model^


def _check_sparse_coder_dimensions(features: Int, atoms: Int) raises:
    if features < 0 or atoms < 0 or (features == 0) != (atoms == 0):
        raise Error("ModelSchemaError: invalid SparseCoder fitted dimensions")


struct SparseCoderSchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of the dense fixed dictionary, settings and fitted dimensions."""

    comptime Model = SparseCoder[Self.dtype]
    comptime _type_id = "moxie.decomposition.SparseCoder"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        _check_sparse_coder_dimensions(model.n_features_in_, model.n_components_)
        if model.dictionary.ndim != 2 or model.dictionary.shape[0] < 1 or model.dictionary.shape[1] < 1:
            raise Error("ModelSchemaError: dictionary must be a nonempty matrix")

        var archive = ModelArchive(Self._type_id, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.string(model.transform_algorithm)
        _write_optional_int(writer, model.transform_n_nonzero_coefs)
        _write_optional_float(writer, model.transform_alpha)
        writer.boolean(model.split_sign)
        _write_optional_int(writer, model.n_jobs)
        writer.boolean(model.positive_code)
        writer.integer(model.transform_max_iter)
        writer.integer(model.n_features_in_)
        writer.integer(model.n_components_)
        archive.manifest = writer^.finish()
        _write_dense_array(archive, "dictionary", model.dictionary,
            [model.dictionary.shape[0], model.dictionary.shape[1]])
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self._type_id, 1)
        archive.validate()
        _check_arrays(archive, True, 1)
        var reader = BinaryReader(archive.manifest)
        _check_dtype[Self.dtype](reader)
        var algorithm = reader.string()
        var nonzero = _read_optional_int(reader)
        var alpha = _read_optional_float(reader)
        var split = reader.boolean()
        var jobs = _read_optional_int(reader)
        var positive = reader.boolean()
        var max_iter = reader.integer()
        var features = reader.integer()
        var atoms = reader.integer()
        reader.finish()
        _check_sparse_coder_dimensions(features, atoms)

        ref payload = archive.payloads[0]
        if payload.name != "dictionary":
            raise Error("ModelSchemaError: missing dictionary")
        ref shape = payload.shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise Error("ModelSchemaError: dictionary must be a nonempty matrix")
        var dictionary = _decode_dense_array[Self.dtype](payload)
        var model = Self.Model(dictionary, transform_algorithm=algorithm,
            transform_n_nonzero_coefs=nonzero, transform_alpha=alpha,
            split_sign=split, n_jobs=jobs, positive_code=positive, transform_max_iter=max_iter)
        model.n_features_in_ = features
        model.n_components_ = atoms
        return model^


def _write_lda_params(mut writer: BinaryWriter, params: LatentDirichletAllocationParams):
    writer.integer(params.n_components)
    _write_optional_float(writer, params.doc_topic_prior)
    _write_optional_float(writer, params.topic_word_prior)
    writer.string(params.learning_method)
    writer.scalar(params.learning_decay)
    writer.scalar(params.learning_offset)
    writer.integer(params.max_iter)
    writer.integer(params.batch_size)
    writer.integer(params.evaluate_every)
    writer.scalar(params.total_samples)
    writer.scalar(params.perp_tol)
    writer.scalar(params.mean_change_tol)
    writer.integer(params.max_doc_update_iter)
    _write_optional_int(writer, params.n_jobs)
    writer.integer(params.verbose)
    _write_optional_int(writer, params.random_state)


def _read_lda_params(mut reader: BinaryReader) raises -> LatentDirichletAllocationParams:
    return LatentDirichletAllocationParams(reader.integer(),
        _read_optional_float(reader), _read_optional_float(reader), reader.string(),
        reader.scalar[DType.float64](), reader.scalar[DType.float64](),
        reader.integer(), reader.integer(), reader.integer(), reader.scalar[DType.float64](),
        reader.scalar[DType.float64](), reader.scalar[DType.float64](),
        reader.integer(), _read_optional_int(reader), reader.integer(), _read_optional_int(reader))


def _check_lda_state[dtype: DType](
    model: LatentDirichletAllocation[dtype], state: _ComponentState,
) raises:
    if state.fitted:
        if (
            state.features < 1 or state.rank < 1 or model.n_iter_ < 0
            or model.n_batch_iter_ <= model.n_iter_
            or not (model.doc_topic_prior_ >= 0 and model.doc_topic_prior_ <= 1)
            or not (model.topic_word_prior_ >= 0 and model.topic_word_prior_ <= 1)
            or model.bound_ < 0 or not model.random_state_
        ):
            raise Error("ModelSchemaError: invalid fitted LDA metadata")
    elif (
        state.features or state.rank or model.n_batch_iter_ or model.n_iter_
        or model.doc_topic_prior_ != 0 or model.topic_word_prior_ != 0
        or model.bound_ != 0 or model.random_state_
    ):
        raise Error("ModelSchemaError: unfitted LDA has learned metadata")


struct LatentDirichletAllocationSchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of dense LDA, including state for continuing partial fit."""

    comptime Model = LatentDirichletAllocation[Self.dtype]
    comptime _type_id = "moxie.decomposition.LatentDirichletAllocation"
    comptime _rng_words = 624

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var state = _component_state(model.components_, model.n_features_in_)
        _check_lda_state(model, state)
        var archive = ModelArchive(Self._type_id, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        _write_lda_params(writer, model.get_params())
        state.write(writer)
        writer.scalar(model.doc_topic_prior_)
        writer.scalar(model.topic_word_prior_)
        writer.integer(model.n_batch_iter_)
        writer.integer(model.n_iter_)
        writer.scalar(model.bound_)
        if model.random_state_:
            ref rng = model.random_state_.value()
            writer.integer(rng.state_position())
            writer.boolean(rng.has_gauss())
            writer.scalar(rng.cached_gauss())
            var words = List[UInt32](capacity=Self._rng_words)
            for i in range(Self._rng_words):
                words.append(rng.state_word(i))
            var rng_shape: List[Int] = [Self._rng_words]
            archive.add_array("random_state_", rng_shape, words)
        archive.manifest = writer^.finish()
        var shape: List[Int] = [state.rank, state.features]
        var zero_prior = model.doc_topic_prior_ == 0 or model.topic_word_prior_ == 0
        _write_array(archive, "components_", model.components_, state.fitted, shape,
            allow_nan=zero_prior, nonnegative=True)
        _write_array(archive, "exp_dirichlet_component_", model.exp_dirichlet_component_,
            state.fitted, shape, allow_nan=zero_prior, nonnegative=True)
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self._type_id, 1)
        archive.validate()
        var reader = BinaryReader(archive.manifest)
        _check_dtype[Self.dtype](reader)
        var params = _read_lda_params(reader)
        var state = _ComponentState.read(reader)
        _check_arrays(archive, state.fitted, 3)
        var model = Self.Model()
        _ = model.set_params(params)
        model.n_features_in_ = state.features
        model.doc_topic_prior_ = reader.scalar[DType.float64]()
        model.topic_word_prior_ = reader.scalar[DType.float64]()
        model.n_batch_iter_ = reader.integer()
        model.n_iter_ = reader.integer()
        model.bound_ = reader.scalar[DType.float64]()
        if state.fitted:
            var position = reader.integer()
            var has_gauss = reader.boolean()
            var cached_gauss = reader.scalar[DType.float64]()
            ref payload = archive.array("random_state_")
            if payload.shape != [Self._rng_words]:
                raise Error("ModelSchemaError: invalid shape for random_state_")
            model.random_state_ = RandomState(state_words=payload.values[DType.uint32](),
                position=position, has_gauss=has_gauss, cached_gauss=cached_gauss)
        reader.finish()
        _check_lda_state(model, state)
        var shape: List[Int] = [state.rank, state.features]
        var zero_prior = model.doc_topic_prior_ == 0 or model.topic_word_prior_ == 0
        model.components_ = _read_array[Self.dtype](archive, "components_", state.fitted,
            shape, allow_nan=zero_prior, nonnegative=True)
        model.exp_dirichlet_component_ = _read_array[Self.dtype](archive,
            "exp_dirichlet_component_", state.fitted, shape, allow_nan=zero_prior, nonnegative=True)
        return model^
