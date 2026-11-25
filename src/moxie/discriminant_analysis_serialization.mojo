"""Native archive schemas for dense discriminant analysis."""

from numojo import NDArray, array
from moxie.discriminant_analysis import (
    ClassLabel,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    Shrinkage,
)
from moxie.serialization import BinaryReader, BinaryWriter, ModelArchive, ModelSchema, dtype_code
from moxie.warnings import FitWarning
from moxie._discriminant_analysis._lda import PRIOR_RENORMALIZATION_MESSAGE


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


def _write_configuration(
    mut writer: BinaryWriter, shrinkage: Shrinkage, priors: Optional[List[Float64]]
):
    writer.boolean(Bool(shrinkage.text))
    if shrinkage.text:
        writer.string(shrinkage.text.value())
    writer.boolean(Bool(shrinkage.value))
    if shrinkage.value:
        writer.scalar[DType.float64](shrinkage.value.value())
    writer.boolean(Bool(priors))
    if priors:
        ref values = priors.value()
        writer.uint(UInt64(len(values)))
        for prior in values:
            writer.scalar[DType.float64](prior)


def _read_configuration(
    mut reader: BinaryReader[_], mut shrinkage: Shrinkage, mut priors: Optional[List[Float64]]
) raises:
    if reader.boolean():
        shrinkage.text = reader.string(reader.remaining())
    if reader.boolean():
        shrinkage.value = reader.scalar[DType.float64]()
    if reader.boolean():
        var count = reader.count(reader.remaining() // 8)
        var values = List[Float64](capacity=count)
        for _ in range(count):
            values.append(reader.scalar[DType.float64]())
        priors = values^


trait DiscriminantLabelCodec:
    """Opt-in wire encoding for a native discriminant analysis label type."""

    comptime Label: ClassLabel
    comptime TypeId: String

    @staticmethod
    def encode(mut writer: BinaryWriter, label: Self.Label) raises: ...

    @staticmethod
    def decode(mut reader: BinaryReader[_]) raises -> Self.Label: ...


struct _LegacyLabelCodec[L: ClassLabel](DiscriminantLabelCodec):
    comptime Label = Self.L
    comptime TypeId = ""

    @staticmethod
    def encode(mut writer: BinaryWriter, label: Self.L):
        comptime if Self.L == Int:
            writer.integer(rebind[Int](label))
        elif Self.L == String:
            writer.string(rebind[String](label))
        else:
            comptime assert False, "Provide a DiscriminantLabelCodec for this label type"

    @staticmethod
    def decode(mut reader: BinaryReader[_]) raises -> Self.L:
        comptime if Self.L == Int:
            return rebind[Self.L](reader.integer()).copy()
        elif Self.L == String:
            return rebind[Self.L](reader.string(reader.remaining())).copy()
        else:
            comptime assert False, "Provide a DiscriminantLabelCodec for this label type"


struct _LabelArchive[L: ClassLabel, Codec: DiscriminantLabelCodec]:
    comptime UsesCustomCodec = Self.Codec != _LegacyLabelCodec[Self.L]

    @staticmethod
    def _check_codec():
        comptime assert Self.L == Self.Codec.Label, "DiscriminantLabelCodec must match the model label type"
        comptime if Self.UsesCustomCodec:
            comptime assert Self.Codec.TypeId != "", "DiscriminantLabelCodec requires a nonempty TypeId"

    @staticmethod
    def write_type(mut writer: BinaryWriter):
        Self._check_codec()
        comptime if Self.UsesCustomCodec:
            writer.string(Self.Codec.TypeId)
        else:
            writer.boolean(Self.L == String)

    @staticmethod
    def read_type(mut reader: BinaryReader[_], has_custom_codec: Bool) raises:
        Self._check_codec()
        if has_custom_codec != Self.UsesCustomCodec:
            raise Error("ModelSchemaError: discriminant label codec mismatch")
        comptime if Self.UsesCustomCodec:
            if reader.string(reader.remaining()) != Self.Codec.TypeId:
                raise Error("ModelSchemaError: discriminant label codec mismatch")
        else:
            if reader.boolean() != (Self.L == String):
                raise Error("ModelSchemaError: discriminant label type mismatch")

    @staticmethod
    def write_classes(mut writer: BinaryWriter, classes: List[Self.L]) raises:
        writer.uint(UInt64(len(classes)))
        for label in classes:
            comptime if Self.UsesCustomCodec:
                var record = BinaryWriter()
                Self.Codec.encode(record, rebind[Self.Codec.Label](label))
                var payload = record^.finish()
                writer.bytes(Span(payload))
            else:
                _LegacyLabelCodec[Self.L].encode(writer, label)

    @staticmethod
    def read_classes(mut reader: BinaryReader[_], limit: Int) raises -> List[Self.L]:
        var count = reader.count(min(limit, reader.remaining() // 8))
        var classes = List[Self.L](capacity=count)
        for _ in range(count):
            comptime if Self.UsesCustomCodec:
                var record = BinaryReader(reader.bytes(reader.remaining()))
                var label = Self.Codec.decode(record)
                record.finish()
                classes.append(rebind[Self.L](label).copy())
            else:
                classes.append(_LegacyLabelCodec[Self.L].decode(reader))
        return classes^


def _validate_class_order[L: ClassLabel](classes: List[L]) raises:
    for i in range(1, len(classes)):
        if not (classes[i - 1] < classes[i]):
            raise Error("ModelSchemaError: classes_ must be strictly ascending")


def _validate_feature_names(
    names: Optional[List[String]], features: Int, fitted: Bool,
) raises:
    if not names:
        return
    ref values = names.value()
    if not fitted or len(values) != features:
        raise Error("ModelSchemaError: invalid feature_names_in_ count")
    var seen = Dict[String, Bool]()
    for name in values:
        if name in seen:
            raise Error("ModelSchemaError: feature_names_in_ must be unique")
        seen[name] = True


def _write_feature_names(mut writer: BinaryWriter, names: List[String]):
    for name in names:
        writer.string(name)


def _read_feature_names(
    mut reader: BinaryReader[_], features: Int,
) raises -> List[String]:
    # A string begins with an eight-byte length. Check the bound before allocating.
    if features < 1 or features > reader.remaining() // 8:
        raise Error("ModelSchemaError: invalid feature_names_in_ count")
    var names = List[String](capacity=features)
    for _ in range(features):
        names.append(reader.string(reader.remaining()))
    return names^


def _validate_lda[dtype: DType, L: ClassLabel](
    model: LinearDiscriminantAnalysis[dtype, L]
) raises:
    if len(model.fit_warnings) > 1:
        raise Error("ModelSchemaError: invalid LDA fit_warnings")
    if len(model.fit_warnings) == 1:
        ref warning = model.fit_warnings[0]
        if warning.category != "UserWarning" or warning.message != PRIOR_RENORMALIZATION_MESSAGE:
            raise Error("ModelSchemaError: invalid LDA fit_warnings")
    var p = model.n_features_in_
    var k = len(model.classes_)
    var fitted = model.__sklearn_is_fitted__()
    _validate_feature_names(model.feature_names_in_, p, fitted)
    if not fitted:
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
    _validate_class_order(model.classes_)
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
    dtype: DType = DType.float64,
    L: ClassLabel = Int,
    Codec: DiscriminantLabelCodec = _LegacyLabelCodec[L],
](ModelSchema):
    """Dense LDA state with native labels and built-in covariance."""

    comptime Model = LinearDiscriminantAnalysis[Self.dtype, Self.L]
    comptime TypeId = "moxie.discriminant_analysis.LinearDiscriminantAnalysis"
    comptime _NamedStateVersion = 3
    comptime _Labels = _LabelArchive[Self.L, Self.Codec]
    comptime _LabelStateVersion = 4
    comptime StateVersion = Self._LabelStateVersion if Self._Labels.UsesCustomCodec else Self._NamedStateVersion

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        _validate_lda(model)
        var has_warning = len(model.fit_warnings) != 0
        var has_names = Bool(model.feature_names_in_)
        var version = Self._NamedStateVersion if has_names else (2 if has_warning else 1)
        if Self._Labels.UsesCustomCodec:
            version = Self.StateVersion
        var archive = ModelArchive(Self.TypeId, version)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        Self._Labels.write_type(writer)
        writer.string(model.solver)
        # Preserve configuration even after set_params changes it between fits.
        _write_configuration(writer, model.shrinkage, model.priors)
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
            Self._Labels.write_classes(writer, model.classes_)
            _write_array(archive, "coef_", model.coef_)
            _write_array(archive, "intercept_", model.intercept_)
            _write_array(archive, "means_", model.means_)
            _write_array(archive, "priors_", model.priors_)
            _write_optional_array(writer, archive, "covariance_", model.covariance_)
            _write_optional_array(writer, archive, "scalings_", model.scalings_)
            _write_optional_array(writer, archive, "xbar_", model.xbar_)
            _write_optional_array(writer, archive, "explained_variance_ratio_", model.explained_variance_ratio_)
        if Self._Labels.UsesCustomCodec:
            writer.boolean(has_names)
        if has_warning or has_names or Self._Labels.UsesCustomCodec:
            writer.boolean(has_warning)
        if has_names:
            _write_feature_names(writer, model.feature_names_in_.value())
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = archive.schema_version
        var has_custom_codec = version == Self._LabelStateVersion
        var has_names = version == Self._NamedStateVersion
        archive.expect(Self.TypeId, version if version == 2 or has_names or has_custom_codec else 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != UInt64(dtype_code[Self.dtype]()):
            raise Error("ModelSchemaError: LDA dtype mismatch")
        Self._Labels.read_type(reader, has_custom_codec)
        var model = Self.Model(solver=reader.string(reader.remaining()))
        _read_configuration(reader, model.shrinkage, model.priors)
        if reader.boolean():
            model.n_components = reader.integer()
        model.store_covariance = reader.boolean()
        model.tol = reader.scalar[DType.float64]()

        var expected_arrays = 0
        if reader.boolean():
            model.n_features_in_ = reader.integer()
            model._max_components = reader.integer()
            model._fitted_solver = reader.string(reader.remaining())
            model.coef_ = _read_array[Self.dtype](archive, "coef_")
            model.intercept_ = _read_array[Self.dtype](archive, "intercept_")
            model.means_ = _read_array[Self.dtype](archive, "means_")
            model.classes_ = Self._Labels.read_classes(reader, model.means_.shape[0])
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
        if has_custom_codec:
            has_names = reader.boolean()
        var has_warning = False
        if version == 2 or has_names or has_custom_codec:
            has_warning = reader.boolean()
            if version == 2 and not has_warning:
                raise Error("ModelSchemaError: invalid LDA fit_warnings")
        if has_warning:
            model.fit_warnings.append(FitWarning("UserWarning", PRIOR_RENORMALIZATION_MESSAGE))
        if has_names:
            model.feature_names_in_ = _read_feature_names(reader, model.n_features_in_)
        reader.finish()
        if len(archive.payloads) != expected_arrays:
            raise Error("ModelSchemaError: unexpected LDA arrays")
        _validate_lda(model)
        return model^


def _write_class_arrays[dtype: DType](
    mut archive: ModelArchive, name: String, arrays: List[NDArray[dtype]]
) raises:
    # Pack a leading class axis so class count does not consume the payload budget.
    var shape = List[Int]()
    shape.append(len(arrays))
    shape.extend(arrays[0].shape.tolist())
    var values = List[Scalar[dtype]]()
    for value in arrays:
        values.extend(value.tolist())
    archive.add_array[dtype](name, shape, values)


def _read_class_arrays[dtype: DType](
    archive: ModelArchive, name: String, count: Int, item_shape: List[Int]
) raises -> List[NDArray[dtype]]:
    var payload = archive.array(name)
    var shape = List[Int]()
    shape.append(count)
    shape.extend(Span(item_shape))
    if payload.shape != shape:
        raise Error("ModelSchemaError: invalid shape for " + name)
    var values = payload.values[dtype]()
    var width = len(values) // count
    var arrays = List[NDArray[dtype]](capacity=count)
    for index in range(count):
        var item = List[Scalar[dtype]](capacity=width)
        var start = index * width
        item.extend(Span(values)[start:start + width])
        arrays.append(array(item, item_shape))
    return arrays^


def _validate_qda[dtype: DType, L: ClassLabel](
    model: QuadraticDiscriminantAnalysis[dtype, L]
) raises:
    var features = model.n_features_in_
    var classes = len(model.classes_)
    var fitted = model.__sklearn_is_fitted__()
    _validate_feature_names(model.feature_names_in_, features, fitted)
    if not fitted:
        if features != 0 or classes != 0 or model.rotations_ or model.scalings_ or model.covariance_:
            raise Error("ModelSchemaError: unfitted QDA has learned state")
        _expect_shape(model.means_, [0, 0], "means_")
        _expect_shape(model.priors_, [0], "priors_")
        return

    if features < 1 or classes < 2:
        raise Error("ModelSchemaError: invalid fitted QDA dimensions")
    _validate_class_order(model.classes_)
    _expect_shape(model.means_, [classes, features], "means_")
    _expect_shape(model.priors_, [classes], "priors_")
    if len(model.rotations_) != classes or len(model.scalings_) != classes:
        raise Error("ModelSchemaError: QDA factor count must match classes_")
    # Both solvers reject fits that do not have full rank in every class.
    for rotation in model.rotations_:
        _expect_shape(rotation, [features, features], "rotations_")
    for scaling in model.scalings_:
        _expect_shape(scaling, [features], "scalings_")
        for value in scaling.tolist():
            # A positive Float64 working scale can round to zero in Float32.
            if not (value >= 0):
                raise Error("ModelSchemaError: invalid value for scalings_")
    if model.covariance_:
        ref covariances = model.covariance_.value()
        if len(covariances) != classes:
            raise Error("ModelSchemaError: QDA covariance count must match classes_")
        for covariance in covariances:
            _expect_shape(covariance, [features, features], "covariance_")


struct QuadraticDiscriminantAnalysisSchema[
    dtype: DType = DType.float64,
    L: ClassLabel = Int,
    Codec: DiscriminantLabelCodec = _LegacyLabelCodec[L],
](ModelSchema):
    """Dense QDA state with native labels and built-in covariance."""

    comptime Model = QuadraticDiscriminantAnalysis[Self.dtype, Self.L]
    comptime TypeId = "moxie.discriminant_analysis.QuadraticDiscriminantAnalysis"
    comptime _NamedStateVersion = 2
    comptime _Labels = _LabelArchive[Self.L, Self.Codec]
    comptime _LabelStateVersion = 3
    comptime StateVersion = Self._LabelStateVersion if Self._Labels.UsesCustomCodec else Self._NamedStateVersion

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        _validate_qda(model)
        var has_names = Bool(model.feature_names_in_)
        var version = Self._NamedStateVersion if has_names else 1
        if Self._Labels.UsesCustomCodec:
            version = Self.StateVersion
        var archive = ModelArchive(Self.TypeId, version)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        Self._Labels.write_type(writer)
        writer.string(model.solver)
        _write_configuration(writer, model.shrinkage, model.priors)
        writer.scalar[DType.float64](model.reg_param)
        writer.boolean(model.store_covariance)
        writer.scalar[DType.float64](model.tol)
        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        if fitted:
            writer.integer(model.n_features_in_)
            Self._Labels.write_classes(writer, model.classes_)
            writer.boolean(Bool(model.covariance_))
            _write_array(archive, "means_", model.means_)
            _write_array(archive, "priors_", model.priors_)
            _write_class_arrays(archive, "rotations_", model.rotations_)
            _write_class_arrays(archive, "scalings_", model.scalings_)
            if model.covariance_:
                _write_class_arrays(archive, "covariance_", model.covariance_.value())
        if Self._Labels.UsesCustomCodec:
            writer.boolean(has_names)
        if has_names:
            _write_feature_names(writer, model.feature_names_in_.value())
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = archive.schema_version
        var has_custom_codec = version == Self._LabelStateVersion
        var has_names = version == Self._NamedStateVersion
        archive.expect(Self.TypeId, version if has_names or has_custom_codec else 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != UInt64(dtype_code[Self.dtype]()):
            raise Error("ModelSchemaError: QDA dtype mismatch")
        Self._Labels.read_type(reader, has_custom_codec)
        var model = Self.Model(solver=reader.string(reader.remaining()))
        _read_configuration(reader, model.shrinkage, model.priors)
        model.reg_param = reader.scalar[DType.float64]()
        model.store_covariance = reader.boolean()
        model.tol = reader.scalar[DType.float64]()

        var expected_arrays = 0
        if reader.boolean():
            var features = reader.integer()
            model.means_ = _read_array[Self.dtype](archive, "means_")
            var max_classes = model.means_.shape[0]
            model.classes_ = Self._Labels.read_classes(reader, max_classes)
            var classes = len(model.classes_)
            if features < 1 or classes < 2:
                raise Error("ModelSchemaError: invalid fitted QDA dimensions")
            _expect_shape(model.means_, [classes, features], "means_")
            model.n_features_in_ = features
            var has_covariance = reader.boolean()
            model.priors_ = _read_array[Self.dtype](archive, "priors_")
            model.rotations_ = _read_class_arrays[Self.dtype](archive, "rotations_", classes, [features, features])
            model.scalings_ = _read_class_arrays[Self.dtype](archive, "scalings_", classes, [features])
            if has_covariance:
                model.covariance_ = _read_class_arrays[Self.dtype](archive, "covariance_", classes, [features, features])
            expected_arrays = 4 + Int(has_covariance)
            model._state.record_fit(1, features)
        if has_custom_codec:
            has_names = reader.boolean()
        if has_names:
            model.feature_names_in_ = _read_feature_names(reader, model.n_features_in_)
        reader.finish()
        if len(archive.payloads) != expected_arrays:
            raise Error("ModelSchemaError: unexpected QDA arrays")
        _validate_qda(model)
        return model^
