"""Native archive schemas for dense discriminant analysis."""

from numojo import NDArray, array
from moxie.discriminant_analysis import ClassLabel, LinearDiscriminantAnalysis
from moxie.serialization import BinaryReader, BinaryWriter, ModelArchive, ModelSchema, dtype_code


def _write_array[dtype: DType](
    mut archive: ModelArchive, name: String, value: NDArray[dtype]
) raises:
    archive.add_array[dtype](name, value.shape.tolist(), value.tolist())


def _read_array[dtype: DType](archive: ModelArchive, name: String) raises -> NDArray[dtype]:
    var payload = archive.array(name)
    if len(payload.shape) < 1 or len(payload.shape) > 2:
        raise Error("ModelSchemaError: invalid array rank for " + name)
    return array(payload.values[dtype](), payload.shape)


def _write_optional_array[dtype: DType](
    mut writer: BinaryWriter, mut archive: ModelArchive, name: String,
    value: Optional[NDArray[dtype]],
) raises:
    writer.boolean(Bool(value))
    if value:
        _write_array(archive, name, value.value())


def _read_optional_array[dtype: DType](
    mut reader: BinaryReader[_], archive: ModelArchive, name: String,
) raises -> Optional[NDArray[dtype]]:
    if reader.boolean():
        return _read_array[dtype](archive, name)
    return None


def _expect_shape[dtype: DType](
    value: NDArray[dtype], expected: List[Int], name: String
) raises:
    if value.shape.tolist() != expected:
        raise Error("ModelSchemaError: invalid shape for " + name)


def _validate_lda[dtype: DType, L: ClassLabel](
    model: LinearDiscriminantAnalysis[dtype, L]
) raises:
    var p = model.n_features_in_
    var k = len(model.classes_)
    if not model.__sklearn_is_fitted__():
        if (
            p != 0 or k != 0 or model._max_components != 0 or model._fitted_solver != ""
            or model.covariance_ or model.scalings_ or model.xbar_ or model.explained_variance_ratio_
        ):
            raise Error("ModelSchemaError: unfitted LDA has learned state")
        _expect_shape(model.coef_, [0, 0], "coef_")
        _expect_shape(model.intercept_, [0], "intercept_")
        _expect_shape(model.means_, [0, 0], "means_")
        _expect_shape(model.priors_, [0], "priors_")
        return

    if p < 1 or k < 1:
        raise Error("ModelSchemaError: invalid fitted LDA dimensions")
    for i in range(1, k):
        if not (model.classes_[i - 1] < model.classes_[i]):
            raise Error("ModelSchemaError: classes_ must be strictly ascending")
    var components = model._max_components
    if components < 0 or components > min(p, k - 1) or (k > 1 and components == 0):
        raise Error("ModelSchemaError: invalid fitted LDA component limit")

    var rows = 1 if k == 2 else k
    _expect_shape(model.coef_, [rows, p], "coef_")
    _expect_shape(model.intercept_, [rows], "intercept_")
    _expect_shape(model.means_, [k, p], "means_")
    _expect_shape(model.priors_, [k], "priors_")
    if model.covariance_:
        _expect_shape(model.covariance_.value(), [p, p], "covariance_")

    var solver = model._fitted_solver
    if solver == "lsqr":
        if not model.covariance_ or model.scalings_ or model.xbar_ or model.explained_variance_ratio_:
            raise Error("ModelSchemaError: inconsistent fitted lsqr arrays")
        return
    if solver != "svd" and solver != "eigen":
        raise Error("ModelSchemaError: invalid fitted LDA solver")
    if not model.scalings_ or not model.explained_variance_ratio_:
        raise Error("ModelSchemaError: missing fitted LDA projection arrays")
    ref scalings = model.scalings_.value()
    ref ratios = model.explained_variance_ratio_.value()
    if scalings.ndim != 2 or ratios.ndim != 1:
        raise Error("ModelSchemaError: invalid rank for LDA projection arrays")
    if solver == "eigen":
        if not model.covariance_ or model.xbar_:
            raise Error("ModelSchemaError: inconsistent fitted eigen arrays")
        _expect_shape(scalings, [p, p], "scalings_")
        _expect_shape(ratios, [components], "explained_variance_ratio_")
        return
    if not model.xbar_:
        raise Error("ModelSchemaError: missing fitted svd center")
    _expect_shape(model.xbar_.value(), [p], "xbar_")
    if scalings.shape[0] != p or scalings.shape[1] > min(p, k):
        raise Error("ModelSchemaError: invalid shape for svd scalings_")
    if ratios.size > components or ratios.size < min(components, scalings.shape[1]):
        raise Error("ModelSchemaError: invalid shape for svd explained_variance_ratio_")


struct LinearDiscriminantAnalysisSchema[
    dtype: DType = DType.float64, L: ClassLabel = Int
](ModelSchema):
    """Version 1 state for dense LDA with Int or String labels and built-in covariance."""

    comptime Model = LinearDiscriminantAnalysis[Self.dtype, Self.L]
    comptime TypeId = "moxie.discriminant_analysis.LinearDiscriminantAnalysis"
    comptime StateVersion = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        comptime assert Self.L == Int or Self.L == String, "LDA schema requires Int or String labels"
        _validate_lda(model)
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.boolean(Self.L == String)
        writer.string(model.solver)
        # Preserve configuration even after set_params changes it between fits.
        writer.boolean(Bool(model.shrinkage.text))
        if model.shrinkage.text:
            writer.string(model.shrinkage.text.value())
        writer.boolean(Bool(model.shrinkage.value))
        if model.shrinkage.value:
            writer.scalar[DType.float64](model.shrinkage.value.value())
        writer.boolean(Bool(model.priors))
        if model.priors:
            ref priors = model.priors.value()
            writer.uint(UInt64(len(priors)))
            for prior in priors:
                writer.scalar[DType.float64](prior)
        writer.boolean(Bool(model.n_components))
        if model.n_components:
            writer.integer(model.n_components.value())
        writer.boolean(model.store_covariance)
        writer.scalar[DType.float64](model.tol)

        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        if fitted:
            writer.integer(model.n_features_in_)
            writer.integer(model._max_components)
            writer.string(model._fitted_solver)
            writer.uint(UInt64(len(model.classes_)))
            for label in model.classes_:
                comptime if Self.L == Int:
                    writer.integer(rebind[Int](label))
                else:
                    writer.string(rebind[String](label))
            _write_array(archive, "coef_", model.coef_)
            _write_array(archive, "intercept_", model.intercept_)
            _write_array(archive, "means_", model.means_)
            _write_array(archive, "priors_", model.priors_)
            _write_optional_array(writer, archive, "covariance_", model.covariance_)
            _write_optional_array(writer, archive, "scalings_", model.scalings_)
            _write_optional_array(writer, archive, "xbar_", model.xbar_)
            _write_optional_array(writer, archive, "explained_variance_ratio_", model.explained_variance_ratio_)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        comptime assert Self.L == Int or Self.L == String, "LDA schema requires Int or String labels"
        archive.expect(Self.TypeId, Self.StateVersion)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != UInt64(dtype_code[Self.dtype]()):
            raise Error("ModelSchemaError: LDA dtype mismatch")
        if reader.boolean() != (Self.L == String):
            raise Error("ModelSchemaError: LDA label type mismatch")
        var model = Self.Model(solver=reader.string(reader.remaining()))
        if reader.boolean():
            model.shrinkage.text = reader.string(reader.remaining())
        if reader.boolean():
            model.shrinkage.value = reader.scalar[DType.float64]()
        if reader.boolean():
            var count = reader.count(reader.remaining() // 8)
            var priors = List[Float64](capacity=count)
            for _ in range(count):
                priors.append(reader.scalar[DType.float64]())
            model.priors = priors^
        if reader.boolean():
            model.n_components = reader.integer()
        model.store_covariance = reader.boolean()
        model.tol = reader.scalar[DType.float64]()

        var expected_arrays = 0
        if reader.boolean():
            model.n_features_in_ = reader.integer()
            model._max_components = reader.integer()
            model._fitted_solver = reader.string(reader.remaining())
            var count = reader.count(reader.remaining() // 8)
            for _ in range(count):
                comptime if Self.L == Int:
                    model.classes_.append(rebind[Self.L](reader.integer()).copy())
                else:
                    model.classes_.append(rebind[Self.L](reader.string(reader.remaining())).copy())
            model.coef_ = _read_array[Self.dtype](archive, "coef_")
            model.intercept_ = _read_array[Self.dtype](archive, "intercept_")
            model.means_ = _read_array[Self.dtype](archive, "means_")
            model.priors_ = _read_array[Self.dtype](archive, "priors_")
            model.covariance_ = _read_optional_array[Self.dtype](reader, archive, "covariance_")
            model.scalings_ = _read_optional_array[Self.dtype](reader, archive, "scalings_")
            model.xbar_ = _read_optional_array[Self.dtype](reader, archive, "xbar_")
            model.explained_variance_ratio_ = _read_optional_array[Self.dtype](
                reader, archive, "explained_variance_ratio_"
            )
            expected_arrays = 4 + Int(Bool(model.covariance_)) + Int(Bool(model.scalings_))
            expected_arrays += Int(Bool(model.xbar_)) + Int(Bool(model.explained_variance_ratio_))
            model._state.record_fit(1, model.n_features_in_)
        reader.finish()
        if len(archive.payloads) != expected_arrays:
            raise Error("ModelSchemaError: unexpected LDA arrays")
        _validate_lda(model)
        return model^
