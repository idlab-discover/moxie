"""Native dense-state archives for the random projection estimators."""

from moxie.random_projection import (
    ComponentCount,
    GaussianRandomProjection,
    ProjectionDensity,
    ProjectionMatrix,
    SparseRandomProjection,
)
from moxie.serialization import (
    BinaryReader,
    BinaryWriter,
    ModelArchive,
    ModelSchema,
    dtype_code,
)


def _write_common_config(
    mut writer: BinaryWriter,
    count: ComponentCount,
    eps: Float64,
    cache_inverse: Bool,
    seed: Optional[Int],
):
    writer.boolean(count.isa[String]())
    if count.isa[String]():
        writer.string(count[String])
    else:
        writer.integer(count[Int])
    writer.scalar[DType.float64](eps)
    writer.boolean(cache_inverse)
    writer.boolean(Bool(seed))
    if seed:
        writer.integer(seed.or_else(0))


def _read_count(mut reader: BinaryReader[_]) raises -> ComponentCount:
    if reader.boolean():
        return reader.string(reader.remaining())
    return reader.integer()


def _read_seed(mut reader: BinaryReader[_]) raises -> Optional[Int]:
    if reader.boolean():
        return reader.integer()
    return None


def _write_matrix[dtype: DType](
    mut archive: ModelArchive,
    name: String,
    matrix: ProjectionMatrix[dtype],
    rows: Int,
    cols: Int,
) raises:
    if matrix.is_sparse:
        raise Error("ModelSchemaError: CSR projection state is unsupported")
    if (
        matrix.rows != rows
        or matrix.cols != cols
        or len(matrix.indices) != 0
        or len(matrix.indptr) != 0
    ):
        raise Error("ModelSchemaError: invalid dense matrix " + name)
    matrix.validate()
    var shape: List[Int] = [rows, cols]
    archive.add_array[dtype](name, shape, matrix.data)


def _read_matrix[dtype: DType](
    archive: ModelArchive, name: String, rows: Int, cols: Int
) raises -> ProjectionMatrix[dtype]:
    for payload in archive.payloads:
        if payload.name == name:
            if (
                len(payload.shape) != 2
                or payload.shape[0] != rows
                or payload.shape[1] != cols
            ):
                raise Error("ModelSchemaError: invalid matrix shape for " + name)
            return ProjectionMatrix[dtype](rows, cols, payload.values[dtype]())
    raise Error("ModelSchemaError: missing matrix " + name)


def _validate_dimensions(fitted: Bool, count: Int, features: Int) raises:
    if fitted:
        if count <= 0 or features <= 0:
            raise Error("ModelSchemaError: invalid fitted projection dimensions")
    elif count != 0 or features != 0:
        raise Error("ModelSchemaError: unfitted projection has learned dimensions")


def _validate_density(fitted: Bool, density: Float64) raises:
    if fitted and density != 1.0:
        raise Error(
            "ModelSchemaError: dense sparse-projection state requires fitted density 1"
        )
    if not fitted and density != 0.0:
        raise Error("ModelSchemaError: unfitted projection has learned density")


def _write_state[dtype: DType](
    mut writer: BinaryWriter,
    mut archive: ModelArchive,
    fitted: Bool,
    count: Int,
    features: Int,
    components: ProjectionMatrix[dtype],
    inverse: Optional[ProjectionMatrix[dtype]],
) raises:
    _validate_dimensions(fitted, count, features)
    writer.boolean(fitted)
    writer.integer(count)
    writer.integer(features)
    writer.boolean(Bool(inverse))
    if not fitted:
        if (
            inverse
            or components.rows != 0
            or components.cols != 0
            or components.is_sparse
            or len(components.data) != 0
            or len(components.indices) != 0
            or len(components.indptr) != 0
        ):
            raise Error("ModelSchemaError: unfitted projection has learned matrices")
        return
    _write_matrix(archive, "components_", components, count, features)
    if inverse:
        _write_matrix(
            archive, "inverse_components_", inverse.value(), features, count
        )


def _read_state[dtype: DType](
    mut reader: BinaryReader[_],
    archive: ModelArchive,
    mut count: Int,
    mut features: Int,
    mut components: ProjectionMatrix[dtype],
    mut inverse: Optional[ProjectionMatrix[dtype]],
) raises -> Bool:
    var fitted = reader.boolean()
    count = reader.integer()
    features = reader.integer()
    var has_inverse = reader.boolean()
    reader.finish()
    _validate_dimensions(fitted, count, features)
    if not fitted and has_inverse:
        raise Error("ModelSchemaError: unfitted projection has an inverse")
    # Exact names, shapes and byte lengths are checked by _read_matrix.
    if len(archive.payloads) != Int(fitted) + Int(has_inverse):
        raise Error("ModelSchemaError: unexpected projection arrays")
    if fitted:
        components = _read_matrix[dtype](archive, "components_", count, features)
    if has_inverse:
        inverse = _read_matrix[dtype](
            archive, "inverse_components_", features, count
        )
    return fitted


def _check_dtype[dtype: DType](mut reader: BinaryReader[_]) raises:
    if reader.integer() != dtype_code[dtype]():
        raise Error("ModelSchemaError: projection archive dtype mismatch")


struct GaussianRandomProjectionSchema[dtype: DType = DType.float64](ModelSchema):
    """Persist a GaussianRandomProjection with the specified scalar dtype."""

    comptime Model = GaussianRandomProjection[Self.dtype]
    comptime type_id = "moxie.GaussianRandomProjection"
    comptime version = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.type_id, Self.version)
        var writer = BinaryWriter()
        writer.integer(dtype_code[Self.dtype]())
        _write_common_config(
            writer,
            model.n_components,
            model.eps,
            model.compute_inverse_components,
            model.random_state,
        )
        _write_state(
            writer,
            archive,
            model.__sklearn_is_fitted__(),
            model.n_components_,
            model.n_features_in_,
            model.components_,
            model.inverse_components_,
        )
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.type_id, Self.version)
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var count = _read_count(reader)
        var eps = reader.scalar[DType.float64]()
        var cache_inverse = reader.boolean()
        var seed = _read_seed(reader)
        var model = Self.Model(
            count^,
            eps=eps,
            compute_inverse_components=cache_inverse,
            random_state=seed,
        )
        model._fitted = _read_state(
            reader,
            archive,
            model.n_components_,
            model.n_features_in_,
            model.components_,
            model.inverse_components_,
        )
        return model^


struct SparseRandomProjectionSchema[dtype: DType = DType.float64](ModelSchema):
    """Persist a SparseRandomProjection with unfitted or dense fitted state."""

    comptime Model = SparseRandomProjection[Self.dtype]
    comptime type_id = "moxie.SparseRandomProjection"
    comptime version = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.type_id, Self.version)
        var writer = BinaryWriter()
        writer.integer(dtype_code[Self.dtype]())
        _write_common_config(
            writer,
            model.n_components,
            model.eps,
            model.compute_inverse_components,
            model.random_state,
        )
        writer.boolean(model.density.isa[String]())
        if model.density.isa[String]():
            writer.string(model.density[String])
        else:
            writer.scalar[DType.float64](model.density[Float64])
        writer.boolean(model.dense_output)
        writer.scalar[DType.float64](model.density_)
        _write_state(
            writer,
            archive,
            model.__sklearn_is_fitted__(),
            model.n_components_,
            model.n_features_in_,
            model.components_,
            model.inverse_components_,
        )
        _validate_density(model.__sklearn_is_fitted__(), model.density_)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.type_id, Self.version)
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var count = _read_count(reader)
        var eps = reader.scalar[DType.float64]()
        var cache_inverse = reader.boolean()
        var seed = _read_seed(reader)
        var density: ProjectionDensity
        if reader.boolean():
            density = reader.string(reader.remaining())
        else:
            density = reader.scalar[DType.float64]()
        var dense_output = reader.boolean()
        var fitted_density = reader.scalar[DType.float64]()
        var model = Self.Model(
            count^,
            density=density^,
            eps=eps,
            dense_output=dense_output,
            compute_inverse_components=cache_inverse, random_state=seed,
        )
        model._fitted = _read_state(
            reader,
            archive,
            model.n_components_,
            model.n_features_in_,
            model.components_,
            model.inverse_components_,
        )
        _validate_density(model._fitted, fitted_density)
        model.density_ = fitted_density
        return model^
