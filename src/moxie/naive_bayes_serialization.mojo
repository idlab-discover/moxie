"""Native archive schemas for dense Naive Bayes estimators."""

from std.math import isfinite, isnan
from std.sys import size_of
from numojo import NDArray, Item, Shape, array, zeros
from moxie._dense import sum_axis2d
from moxie._labels import StringTarget
from moxie._naive_bayes_common import _NaiveBayesMetadata
from moxie._naive_bayes_dense import ClassDomain, DenseMatrix
from moxie.naive_bayes import (
    BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, IntListDomain, MultinomialNB,
    NumericDomain, StringDomain,
)
from moxie.metadata_routing import SampleWeightRequest
from moxie.serialization import ArrayPayload, BinaryReader, BinaryWriter, ModelArchive, ModelSchema, dtype_code
from moxie.warnings import FitWarning


comptime _WEIGHT_REQUEST_VERSION = 2
comptime _FEATURE_NAMES_VERSION = 3


def _write_request(mut writer: BinaryWriter, request: SampleWeightRequest):
    var value = request.value()
    if value.isa[Bool]():
        writer.uint(UInt64(2 if value[Bool] else 1))
    elif value.isa[String]():
        writer.uint(3)
        writer.string(value[String])
    else:
        writer.uint(0)


def _read_request(mut reader: BinaryReader[_]) raises -> SampleWeightRequest:
    var request = SampleWeightRequest()
    var tag = reader.uint()
    if tag == 3:
        var alias_name = reader.string(reader.remaining())
        request.set(alias_name)
        if not request.value().isa[String]():
            raise Error("ModelSchemaError: invalid sample-weight alias")
    elif tag == 1 or tag == 2:
        request.set(tag == 2)
    elif tag != 0:
        raise Error("ModelSchemaError: invalid sample-weight request tag")
    return request^


def _has_metadata(metadata: _NaiveBayesMetadata) -> Bool:
    return (
        not metadata.fit_score.fit.value().isa[NoneType]()
        or not metadata.fit_score.score.value().isa[NoneType]()
        or not metadata.partial_fit.value().isa[NoneType]()
    )


def _write_metadata(
    mut writer: BinaryWriter, mut archive: ModelArchive, metadata: _NaiveBayesMetadata,
    feature_names: Optional[List[String]], features: Int,
) raises:
    var named = Bool(feature_names)
    if named or _has_metadata(metadata):
        archive.schema_version = _FEATURE_NAMES_VERSION if named else _WEIGHT_REQUEST_VERSION
        _write_request(writer, metadata.fit_score.fit)
        _write_request(writer, metadata.fit_score.score)
        _write_request(writer, metadata.partial_fit)
    if feature_names:
        ref names = feature_names.value()
        _validate_feature_names(names, features)
        for name in names:
            writer.string(name)


def _read_metadata(
    mut reader: BinaryReader[_], archive: ModelArchive,
) raises -> _NaiveBayesMetadata:
    var metadata = _NaiveBayesMetadata()
    if archive.schema_version in (_WEIGHT_REQUEST_VERSION, _FEATURE_NAMES_VERSION):
        metadata.fit_score.fit = _read_request(reader)
        metadata.fit_score.score = _read_request(reader)
        metadata.partial_fit = _read_request(reader)
        if archive.schema_version == _WEIGHT_REQUEST_VERSION and not _has_metadata(metadata):
            raise Error("ModelSchemaError: missing configured sample-weight request")
    return metadata^


def _validate_feature_names(names: List[String], features: Int) raises:
    if features < 1 or len(names) != features:
        raise Error("ModelSchemaError: invalid feature names count")
    var seen = Dict[String, Bool]()
    for name in names:
        if name in seen:
            raise Error("ModelSchemaError: duplicate feature name")
        seen[name] = True


def _read_feature_names(mut reader: BinaryReader[_], features: Int) raises -> List[String]:
    # Each entry has at least an eight-byte string length.
    if features < 1 or features > reader.remaining() // 8:
        raise Error("ModelSchemaError: invalid feature names count")
    var names = List[String](capacity=features)
    for _ in range(features):
        names.append(reader.string(reader.remaining()))
    _validate_feature_names(names, features)
    return names^


def _numeric_label_dtype[D: ClassDomain]() -> DType:
    comptime if D == NumericDomain[DType.int64]:
        return DType.int64
    elif D == NumericDomain[DType.float32]:
        return DType.float32
    else:
        comptime assert D == NumericDomain[DType.float64], "Unsupported Naive Bayes archive label domain"
        return DType.float64


def _domain_tag[D: ClassDomain]() -> UInt64:
    comptime if D == IntListDomain:
        return 1
    elif D == StringDomain:
        return 2
    else:
        comptime label = _numeric_label_dtype[D]()
        return 10 + UInt64(dtype_code[label]())


def _write_classes[D: ClassDomain](
    mut writer: BinaryWriter, classes: D.Target, count: Int
) raises:
    writer.uint(UInt64(count))
    comptime if D == IntListDomain:
        for label in rebind[List[Int]](classes):
            writer.integer(label)
    elif D == StringDomain:
        ref target = rebind[StringTarget](classes)
        for i in range(count):
            writer.string(target.get(i))
    else:
        comptime label = _numeric_label_dtype[D]()
        ref target = rebind[NDArray[label]](classes)
        for i in range(count):
            writer.scalar(target[Item(i)])


def _read_classes[D: ClassDomain](mut reader: BinaryReader[_]) raises -> D.Target:
    comptime if D == IntListDomain:
        var count = reader.count(reader.remaining() // 8)
        var labels = List[Int](capacity=count)
        for _ in range(count):
            labels.append(reader.integer())
        return rebind_var[D.Target](labels^)
    elif D == StringDomain:
        var count = reader.count(reader.remaining() // 8)
        var labels = List[String](capacity=count)
        for _ in range(count):
            labels.append(reader.string(reader.remaining()))
        return rebind_var[D.Target](StringTarget(labels))
    else:
        comptime label = _numeric_label_dtype[D]()
        var count = reader.count(reader.remaining() // size_of[Scalar[label]]())
        var labels = List[Scalar[label]](capacity=count)
        for _ in range(count):
            labels.append(reader.scalar[label]())
        return rebind_var[D.Target](array(labels, [count]))


def _check_classes[D: ClassDomain](
    classes: D.Target, count: Int, domain: D, estimator: String,
) raises:
    if count < 1 or domain.n_classes() != count:
        raise Error("ModelSchemaError: invalid " + estimator + " class count")
    var ordered = domain.class_values()
    for i in range(count):
        if not D.equal_at(classes, ordered, i):
            raise Error("ModelSchemaError: " + estimator + " classes must match the sorted label domain")


def _write_array[dtype: DType](
    mut archive: ModelArchive, name: String, value: NDArray[dtype], shape: List[Int],
) raises:
    if value.shape.tolist() != shape:
        raise Error("ModelSchemaError: invalid shape for " + name)
    archive.add_array[dtype](name, shape, value.tolist())


def _read_array[dtype: DType](
    archive: ModelArchive, name: String, shape: List[Int],
) raises -> NDArray[dtype]:
    var payload = archive.array(name)
    if payload.shape != shape:
        raise Error("ModelSchemaError: invalid shape for " + name)
    return array(payload.values[dtype](), shape)


def _check_counts_and_priors[dtype: DType](
    counts: NDArray[dtype], priors: NDArray[dtype], epsilon: Scalar[dtype],
) raises:
    if epsilon < 0:
        raise Error("ModelSchemaError: negative GaussianNB epsilon")
    # Preserve overflow and rounding from the original state dtype, including
    # NaN empirical priors when incremental counts have overflowed.
    var has_positive_count = False
    for i in range(counts.size):
        var count = counts[Item(i)]
        var prior = priors[Item(i)]
        if isnan(count) or count < 0:
            raise Error("ModelSchemaError: invalid GaussianNB class count")
        has_positive_count = has_positive_count or count > 0
        if prior < 0 or prior > 1.000011 or (isnan(prior) and isfinite(count)):
            raise Error("ModelSchemaError: invalid GaussianNB learned prior")
    if not has_positive_count:
        raise Error("ModelSchemaError: fitted GaussianNB has no positive class count")


struct GaussianNBSchema[
    D: ClassDomain = IntListDomain, dtype: DType = DType.float64
](ModelSchema):
    """Native state for GaussianNB with a built-in label domain."""

    comptime Model = GaussianNB[Self.D, Self.dtype]
    comptime TypeId = "moxie.naive_bayes.GaussianNB"
    comptime StateVersion = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.uint(_domain_tag[Self.D]())
        writer.scalar(model.var_smoothing)
        writer.boolean(Bool(model.priors))
        if model.priors:
            ref priors = model.priors.value()
            writer.uint(UInt64(len(priors)))
            for prior in priors:
                writer.scalar(prior)

        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        var count = Self.D.check_target(model.classes_)
        var features = model.n_features_in_
        if fitted:
            if not model._domain:
                raise Error("ModelSchemaError: missing fitted GaussianNB label domain")
            _check_classes[Self.D](model.classes_, count, model._domain.value(), "GaussianNB")
            writer.integer(features)
            writer.scalar(model.epsilon_)
            _write_classes[Self.D](writer, model.classes_, count)
            _write_array(archive, "class_count_", model.class_count_, [count])
            _write_array(archive, "class_prior_", model.class_prior_, [count])
            # These are raw moments. Even finite training inputs can produce
            # nonfinite moments when arithmetic in the selected dtype overflows.
            _write_array(archive, "theta_", model.theta_, [count, features])
            _write_array(archive, "var_", model.var_, [count, features])
            _check_counts_and_priors(model.class_count_, model.class_prior_, model.epsilon_)
        elif (
            features != 0 or count != 0 or model._domain or model.epsilon_ != 0
            or model.class_count_.shape.tolist() != [0] or model.class_prior_.shape.tolist() != [0]
            or model.theta_.shape.tolist() != [0, 0] or model.var_.shape.tolist() != [0, 0]
        ):
            raise Error("ModelSchemaError: unfitted GaussianNB has learned state")
        _write_metadata(writer, archive, model._metadata, model.feature_names_in_, features)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = Self.StateVersion
        if archive.schema_version in (_WEIGHT_REQUEST_VERSION, _FEATURE_NAMES_VERSION):
            version = archive.schema_version
        archive.expect(Self.TypeId, version)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != UInt64(dtype_code[Self.dtype]()):
            raise Error("ModelSchemaError: GaussianNB dtype mismatch")
        if reader.uint() != _domain_tag[Self.D]():
            raise Error("ModelSchemaError: GaussianNB label domain mismatch")
        var model = Self.Model(var_smoothing=reader.scalar[DType.float64]())
        if reader.boolean():
            var count = reader.count(reader.remaining() // 8)
            var priors = List[Float64](capacity=count)
            for _ in range(count):
                priors.append(reader.scalar[DType.float64]())
            model.priors = priors^

        var fitted = reader.boolean()
        if len(archive.payloads) != (4 if fitted else 0):
            raise Error("ModelSchemaError: unexpected GaussianNB arrays")
        if fitted:
            var features = reader.integer()
            if features < 1:
                raise Error("ModelSchemaError: invalid GaussianNB feature count")
            var epsilon = reader.scalar[Self.dtype]()
            var classes = _read_classes[Self.D](reader)
            var count = Self.D.check_target(classes)
            var domain = Self.D.from_classes(classes)
            _check_classes[Self.D](classes, count, domain, "GaussianNB")
            model.class_count_ = _read_array[Self.dtype](archive, "class_count_", [count])
            model.class_prior_ = _read_array[Self.dtype](archive, "class_prior_", [count])
            model.theta_ = _read_array[Self.dtype](archive, "theta_", [count, features])
            model.var_ = _read_array[Self.dtype](archive, "var_", [count, features])
            _check_counts_and_priors(model.class_count_, model.class_prior_, epsilon)
            model.epsilon_ = epsilon
            model.n_features_in_ = features
            model.classes_ = classes^
            model._domain = domain^
        model._metadata = _read_metadata(reader, archive)
        if version == _FEATURE_NAMES_VERSION:
            model.feature_names_in_ = _read_feature_names(reader, model.n_features_in_)
        reader.finish()
        return model^


def _check_count_priors[estimator: String](
    class_counts: DenseMatrix, log_priors: DenseMatrix,
) raises:
    var has_positive_count = False
    for c in range(class_counts.size):
        var count = class_counts.item(c)
        var prior = log_priors.item(c)
        if isnan(count) or count < 0:
            raise Error("ModelSchemaError: invalid " + estimator + " class count")
        # A negative explicit prior can produce NaN regardless of the count.
        # The current configuration can differ from the one used at fit time.
        if prior == Float64("inf"):
            raise Error("ModelSchemaError: invalid " + estimator + " log prior")
        has_positive_count = has_positive_count or count > 0
    if not has_positive_count:
        raise Error("ModelSchemaError: fitted " + estimator + " has no positive class count")


def _check_count_statistics[complement: Bool, estimator: String](
    class_counts: DenseMatrix, feature_counts: DenseMatrix,
    log_priors: DenseMatrix, log_probs: DenseMatrix,
) raises:
    _check_count_priors[estimator](class_counts, log_priors)
    for c in range(class_counts.size):
        for f in range(feature_counts.shape[1]):
            var feature_count = feature_counts.item(c, f)
            var weight = log_probs.item(c, f)
            var invalid_weight = weight < 0 if complement else weight > 0
            if isnan(feature_count) or feature_count < 0 or invalid_weight:
                raise Error("ModelSchemaError: invalid " + estimator + " feature statistics")


def _check_feature_all(feature_all: DenseMatrix, feature_counts: DenseMatrix) raises:
    var totals = sum_axis2d(feature_counts, 0)
    for f in range(feature_all.size):
        if feature_all.item(f) != totals.item(f):
            raise Error("ModelSchemaError: ComplementNB feature_all_ disagrees with feature_count_")


def _count_input_dtype_tag[dtype: DType]() -> UInt64:
    # These extra tags identify input specializations in the manifest only.
    # All numerical payloads in the count schema are Float64.
    comptime if dtype == DType.bool:
        return 7
    elif dtype == DType.int8:
        return 8
    elif dtype == DType.int16:
        return 9
    elif dtype == DType.uint8:
        return 10
    elif dtype == DType.uint16:
        return 11
    else:
        return UInt64(dtype_code[dtype]())


def _write_discrete_config(
    mut writer: BinaryWriter,
    alpha: Float64, force_alpha: Bool, fit_prior: Bool,
    class_prior: Optional[List[Float64]],
) raises:
    writer.scalar(alpha)
    writer.boolean(force_alpha)
    writer.boolean(fit_prior)
    writer.boolean(Bool(class_prior))
    if class_prior:
        ref priors = class_prior.value()
        writer.uint(UInt64(len(priors)))
        for prior in priors:
            writer.scalar(prior)


def _write_count_config(
    mut writer: BinaryWriter, mut archive: ModelArchive,
    alpha: Float64, force_alpha: Bool, fit_prior: Bool,
    class_prior: Optional[List[Float64]], alpha_values: Optional[DenseMatrix],
) raises:
    _write_discrete_config(writer, alpha, force_alpha, fit_prior, class_prior)
    writer.boolean(Bool(alpha_values))
    if alpha_values:
        ref values = alpha_values.value()
        _write_array(archive, "alpha_values", values, [values.size])


@fieldwise_init
struct _CountConfig(Movable):
    var alpha: Float64
    var force_alpha: Bool
    var fit_prior: Bool
    var class_prior: Optional[List[Float64]]
    var has_alpha_values: Bool


def _read_discrete_config(mut reader: BinaryReader[_]) raises -> _CountConfig:
    var alpha = reader.scalar[DType.float64]()
    var force_alpha = reader.boolean()
    var fit_prior = reader.boolean()
    var class_prior: Optional[List[Float64]] = None
    if reader.boolean():
        var count = reader.count(reader.remaining() // 8)
        var priors = List[Float64](capacity=count)
        for _ in range(count):
            priors.append(reader.scalar[DType.float64]())
        class_prior = priors^
    return _CountConfig(alpha, force_alpha, fit_prior, class_prior^, False)


def _read_count_config(mut reader: BinaryReader[_]) raises -> _CountConfig:
    var config = _read_discrete_config(reader)
    config.has_alpha_values = reader.boolean()
    return config^


def _read_count_alpha(archive: ModelArchive) raises -> DenseMatrix:
    var alpha = archive.array("alpha_values")
    if len(alpha.shape) != 1:
        raise Error("ModelSchemaError: invalid shape for alpha_values")
    return array(alpha.values[DType.float64](), alpha.shape)


def _write_count_warning(mut writer: BinaryWriter, warnings: List[FitWarning], estimator: String) raises:
    if len(warnings) > 1:
        raise Error("ModelSchemaError: invalid " + estimator + " fit warning")
    writer.boolean(len(warnings) == 1)
    if len(warnings) == 1:
        ref warning = warnings[0]
        if warning.category != "UserWarning" or warning.message.byte_length() == 0:
            raise Error("ModelSchemaError: invalid " + estimator + " fit warning")
        writer.string(warning.message)


def _read_count_warning(mut reader: BinaryReader[_], estimator: String) raises -> List[FitWarning]:
    if not reader.boolean():
        return []
    var message = reader.string(reader.remaining())
    if message.byte_length() == 0:
        raise Error("ModelSchemaError: invalid " + estimator + " fit warning")
    return [FitWarning("UserWarning", message^)]


struct MultinomialNBSchema[
    D: ClassDomain = IntListDomain, dtype: DType = DType.float64
](ModelSchema):
    """Native state for MultinomialNB with a built-in label domain."""

    comptime Model = MultinomialNB[Self.D, Self.dtype]
    comptime Estimator = "MultinomialNB"
    comptime TypeId = "moxie.naive_bayes.MultinomialNB"
    comptime StateVersion = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        writer.uint(_count_input_dtype_tag[Self.dtype]())
        writer.uint(_domain_tag[Self.D]())
        _write_count_config(writer, archive, model.alpha, model.force_alpha,
                            model.fit_prior, model.class_prior, model.alpha_values)

        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        var count = Self.D.check_target(model.classes_)
        var features = model.n_features_in_
        if fitted:
            if not model._domain:
                raise Error("ModelSchemaError: missing fitted " + Self.Estimator + " label domain")
            _check_classes[Self.D](model.classes_, count, model._domain.value(), Self.Estimator)
            writer.integer(features)
            _write_classes[Self.D](writer, model.classes_, count)
            _write_count_warning(writer, model.fit_warnings, Self.Estimator)
            _write_array(archive, "class_count_", model.class_count_, [count])
            _write_array(archive, "class_log_prior_", model.class_log_prior_, [count])
            _write_array(archive, "feature_count_", model.feature_count_, [count, features])
            _write_array(archive, "feature_log_prob_", model.feature_log_prob_, [count, features])
            _check_count_statistics[False, Self.Estimator](
                model.class_count_, model.feature_count_, model.class_log_prior_, model.feature_log_prob_,
            )
        elif (
            features != 0 or count != 0 or model._domain or len(model.fit_warnings) != 0
            or model.class_count_.shape.tolist() != [0] or model.class_log_prior_.shape.tolist() != [0]
            or model.feature_count_.shape.tolist() != [0, 0]
            or model.feature_log_prob_.shape.tolist() != [0, 0]
        ):
            raise Error("ModelSchemaError: unfitted " + Self.Estimator + " has learned state")
        _write_metadata(writer, archive, model._metadata, model.feature_names_in_, features)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = Self.StateVersion
        if archive.schema_version in (_WEIGHT_REQUEST_VERSION, _FEATURE_NAMES_VERSION):
            version = archive.schema_version
        archive.expect(Self.TypeId, version)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != _count_input_dtype_tag[Self.dtype]():
            raise Error("ModelSchemaError: " + Self.Estimator + " dtype mismatch")
        if reader.uint() != _domain_tag[Self.D]():
            raise Error("ModelSchemaError: " + Self.Estimator + " label domain mismatch")
        var config = _read_count_config(reader)
        var has_alpha_values = config.has_alpha_values
        var model = Self.Model(alpha=config.alpha, force_alpha=config.force_alpha,
                               fit_prior=config.fit_prior, class_prior=config.class_prior.copy())
        var fitted = reader.boolean()
        var expected_arrays = Int(has_alpha_values) + (4 if fitted else 0)
        if len(archive.payloads) != expected_arrays:
            raise Error("ModelSchemaError: unexpected " + Self.Estimator + " arrays")
        if has_alpha_values:
            model.alpha_values = _read_count_alpha(archive)
        if fitted:
            var features = reader.integer()
            if features < 1:
                raise Error("ModelSchemaError: invalid " + Self.Estimator + " feature count")
            var classes = _read_classes[Self.D](reader)
            var count = Self.D.check_target(classes)
            var domain = Self.D.from_classes(classes)
            _check_classes[Self.D](classes, count, domain, Self.Estimator)
            model.fit_warnings = _read_count_warning(reader, Self.Estimator)
            model.class_count_ = _read_array[DType.float64](archive, "class_count_", [count])
            model.class_log_prior_ = _read_array[DType.float64](archive, "class_log_prior_", [count])
            model.feature_count_ = _read_array[DType.float64](archive, "feature_count_", [count, features])
            model.feature_log_prob_ = _read_array[DType.float64](archive, "feature_log_prob_", [count, features])
            _check_count_statistics[False, Self.Estimator](
                model.class_count_, model.feature_count_, model.class_log_prior_, model.feature_log_prob_,
            )
            model.n_features_in_ = features
            model.classes_ = classes^
            model._domain = domain^
        model._metadata = _read_metadata(reader, archive)
        if version == _FEATURE_NAMES_VERSION:
            model.feature_names_in_ = _read_feature_names(reader, model.n_features_in_)
        reader.finish()
        return model^


struct ComplementNBSchema[
    D: ClassDomain = IntListDomain, dtype: DType = DType.float64
](ModelSchema):
    """Native state for ComplementNB with a built-in label domain."""

    comptime Model = ComplementNB[Self.D, Self.dtype]
    comptime Estimator = "ComplementNB"
    comptime TypeId = "moxie.naive_bayes.ComplementNB"
    comptime StateVersion = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        writer.uint(_count_input_dtype_tag[Self.dtype]())
        writer.uint(_domain_tag[Self.D]())
        _write_count_config(writer, archive, model.alpha, model.force_alpha,
                            model.fit_prior, model.class_prior, model.alpha_values)

        writer.boolean(model.norm)
        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        var count = Self.D.check_target(model.classes_)
        var features = model.n_features_in_
        if fitted:
            if not model._domain:
                raise Error("ModelSchemaError: missing fitted " + Self.Estimator + " label domain")
            _check_classes[Self.D](model.classes_, count, model._domain.value(), Self.Estimator)
            writer.integer(features)
            _write_classes[Self.D](writer, model.classes_, count)
            _write_count_warning(writer, model.fit_warnings, Self.Estimator)
            _write_array(archive, "class_count_", model.class_count_, [count])
            _write_array(archive, "class_log_prior_", model.class_log_prior_, [count])
            _write_array(archive, "feature_count_", model.feature_count_, [count, features])
            _write_array(archive, "feature_log_prob_", model.feature_log_prob_, [count, features])
            _check_count_statistics[True, Self.Estimator](
                model.class_count_, model.feature_count_, model.class_log_prior_, model.feature_log_prob_,
            )
            _write_array(archive, "feature_all_", model.feature_all_, [features])
            _check_feature_all(model.feature_all_, model.feature_count_)
        elif (
            features != 0 or count != 0 or model._domain or len(model.fit_warnings) != 0
            or model.class_count_.shape.tolist() != [0] or model.class_log_prior_.shape.tolist() != [0]
            or model.feature_count_.shape.tolist() != [0, 0]
            or model.feature_log_prob_.shape.tolist() != [0, 0]
            or model.feature_all_.shape.tolist() != [0]
        ):
            raise Error("ModelSchemaError: unfitted " + Self.Estimator + " has learned state")
        _write_metadata(writer, archive, model._metadata, model.feature_names_in_, features)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = Self.StateVersion
        if archive.schema_version in (_WEIGHT_REQUEST_VERSION, _FEATURE_NAMES_VERSION):
            version = archive.schema_version
        archive.expect(Self.TypeId, version)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != _count_input_dtype_tag[Self.dtype]():
            raise Error("ModelSchemaError: " + Self.Estimator + " dtype mismatch")
        if reader.uint() != _domain_tag[Self.D]():
            raise Error("ModelSchemaError: " + Self.Estimator + " label domain mismatch")
        var config = _read_count_config(reader)
        var has_alpha_values = config.has_alpha_values
        var model = Self.Model(alpha=config.alpha, force_alpha=config.force_alpha,
                               fit_prior=config.fit_prior, class_prior=config.class_prior.copy())
        model.norm = reader.boolean()
        var fitted = reader.boolean()
        var expected_arrays = Int(has_alpha_values) + (5 if fitted else 0)
        if len(archive.payloads) != expected_arrays:
            raise Error("ModelSchemaError: unexpected " + Self.Estimator + " arrays")
        if has_alpha_values:
            model.alpha_values = _read_count_alpha(archive)
        if fitted:
            var features = reader.integer()
            if features < 1:
                raise Error("ModelSchemaError: invalid " + Self.Estimator + " feature count")
            var classes = _read_classes[Self.D](reader)
            var count = Self.D.check_target(classes)
            var domain = Self.D.from_classes(classes)
            _check_classes[Self.D](classes, count, domain, Self.Estimator)
            model.fit_warnings = _read_count_warning(reader, Self.Estimator)
            model.class_count_ = _read_array[DType.float64](archive, "class_count_", [count])
            model.class_log_prior_ = _read_array[DType.float64](archive, "class_log_prior_", [count])
            model.feature_count_ = _read_array[DType.float64](archive, "feature_count_", [count, features])
            model.feature_log_prob_ = _read_array[DType.float64](archive, "feature_log_prob_", [count, features])
            _check_count_statistics[True, Self.Estimator](
                model.class_count_, model.feature_count_, model.class_log_prior_, model.feature_log_prob_,
            )
            model.feature_all_ = _read_array[DType.float64](archive, "feature_all_", [features])
            _check_feature_all(model.feature_all_, model.feature_count_)
            model.n_features_in_ = features
            model.classes_ = classes^
            model._domain = domain^
        model._metadata = _read_metadata(reader, archive)
        if version == _FEATURE_NAMES_VERSION:
            model.feature_names_in_ = _read_feature_names(reader, model.n_features_in_)
        reader.finish()
        return model^


struct BernoulliNBSchema[
    D: ClassDomain = IntListDomain, dtype: DType = DType.float64
](ModelSchema):
    """Native state for BernoulliNB with a built-in label domain."""

    comptime Model = BernoulliNB[Self.D, Self.dtype]
    comptime Estimator = "BernoulliNB"
    comptime TypeId = "moxie.naive_bayes.BernoulliNB"
    comptime StateVersion = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        writer.uint(_count_input_dtype_tag[Self.dtype]())
        writer.uint(_domain_tag[Self.D]())
        _write_count_config(writer, archive, model.alpha, model.force_alpha,
                            model.fit_prior, model.class_prior, model.alpha_values)

        writer.boolean(Bool(model.binarize))
        if model.binarize:
            writer.scalar(model.binarize.value())
        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        var count = Self.D.check_target(model.classes_)
        var features = model.n_features_in_
        if fitted:
            if not model._domain:
                raise Error("ModelSchemaError: missing fitted " + Self.Estimator + " label domain")
            _check_classes[Self.D](model.classes_, count, model._domain.value(), Self.Estimator)
            writer.integer(features)
            _write_classes[Self.D](writer, model.classes_, count)
            _write_count_warning(writer, model.fit_warnings, Self.Estimator)
            _write_array(archive, "class_count_", model.class_count_, [count])
            _write_array(archive, "class_log_prior_", model.class_log_prior_, [count])
            # Raw inputs with binarize=None can produce signed counts and
            # nonfinite log probabilities. Current configuration may differ from fit.
            _write_array(archive, "feature_count_", model.feature_count_, [count, features])
            _write_array(archive, "feature_log_prob_", model.feature_log_prob_, [count, features])
            _check_count_priors[Self.Estimator](model.class_count_, model.class_log_prior_)
        elif (
            features != 0 or count != 0 or model._domain or len(model.fit_warnings) != 0
            or model.class_count_.shape.tolist() != [0] or model.class_log_prior_.shape.tolist() != [0]
            or model.feature_count_.shape.tolist() != [0, 0]
            or model.feature_log_prob_.shape.tolist() != [0, 0]
        ):
            raise Error("ModelSchemaError: unfitted " + Self.Estimator + " has learned state")
        _write_metadata(writer, archive, model._metadata, model.feature_names_in_, features)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = Self.StateVersion
        if archive.schema_version in (_WEIGHT_REQUEST_VERSION, _FEATURE_NAMES_VERSION):
            version = archive.schema_version
        archive.expect(Self.TypeId, version)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != _count_input_dtype_tag[Self.dtype]():
            raise Error("ModelSchemaError: " + Self.Estimator + " dtype mismatch")
        if reader.uint() != _domain_tag[Self.D]():
            raise Error("ModelSchemaError: " + Self.Estimator + " label domain mismatch")
        var config = _read_count_config(reader)
        var has_alpha_values = config.has_alpha_values
        var binarize: Optional[Float64] = None
        if reader.boolean():
            binarize = reader.scalar[DType.float64]()
        var model = Self.Model(alpha=config.alpha, force_alpha=config.force_alpha,
                               fit_prior=config.fit_prior, class_prior=config.class_prior.copy(),
                               binarize=binarize)
        var fitted = reader.boolean()
        var expected_arrays = Int(has_alpha_values) + (4 if fitted else 0)
        if len(archive.payloads) != expected_arrays:
            raise Error("ModelSchemaError: unexpected " + Self.Estimator + " arrays")
        if has_alpha_values:
            model.alpha_values = _read_count_alpha(archive)
        if fitted:
            var features = reader.integer()
            if features < 1:
                raise Error("ModelSchemaError: invalid " + Self.Estimator + " feature count")
            var classes = _read_classes[Self.D](reader)
            var count = Self.D.check_target(classes)
            var domain = Self.D.from_classes(classes)
            _check_classes[Self.D](classes, count, domain, Self.Estimator)
            model.fit_warnings = _read_count_warning(reader, Self.Estimator)
            model.class_count_ = _read_array[DType.float64](archive, "class_count_", [count])
            model.class_log_prior_ = _read_array[DType.float64](archive, "class_log_prior_", [count])
            model.feature_count_ = _read_array[DType.float64](archive, "feature_count_", [count, features])
            model.feature_log_prob_ = _read_array[DType.float64](archive, "feature_log_prob_", [count, features])
            _check_count_priors[Self.Estimator](model.class_count_, model.class_log_prior_)
            model.n_features_in_ = features
            model.classes_ = classes^
            model._domain = domain^
        model._metadata = _read_metadata(reader, archive)
        if version == _FEATURE_NAMES_VERSION:
            model.feature_names_in_ = _read_feature_names(reader, model.n_features_in_)
        reader.finish()
        return model^


def _check_categorical_statistics(
    counts: DenseMatrix, log_probs: DenseMatrix, categories: Int64,
) raises:
    if categories < 1 or categories > Int64(counts.shape[1]):
        raise Error("ModelSchemaError: invalid CategoricalNB category count")
    for c in range(counts.shape[0]):
        for k in range(counts.shape[1]):
            var count = counts.item(c, k)
            # Zero smoothing and overflow can produce nonfinite log probabilities.
            if isnan(count) or count < 0 or log_probs.item(c, k) > 0:
                raise Error("ModelSchemaError: invalid CategoricalNB feature statistics")


struct CategoricalNBSchema[
    D: ClassDomain = IntListDomain, dtype: DType = DType.float64
](ModelSchema):
    """Native state for CategoricalNB with a built-in label domain."""

    comptime Model = CategoricalNB[Self.D, Self.dtype]
    comptime Estimator = "CategoricalNB"
    comptime TypeId = "moxie.naive_bayes.CategoricalNB"
    comptime StateVersion = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        writer.uint(_count_input_dtype_tag[Self.dtype]())
        writer.uint(_domain_tag[Self.D]())
        _write_discrete_config(
            writer, model.alpha, model.force_alpha, model.fit_prior, model.class_prior,
        )
        writer.boolean(Bool(model.min_categories))
        if model.min_categories:
            writer.integer(model.min_categories.value())
        writer.boolean(Bool(model.min_categories_values))
        if model.min_categories_values:
            ref minima = model.min_categories_values.value()
            writer.uint(UInt64(len(minima)))
            for minimum in minima:
                writer.integer(minimum)

        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        var count = Self.D.check_target(model.classes_)
        var features = model.n_features_in_
        if fitted:
            if not model._domain:
                raise Error("ModelSchemaError: missing fitted CategoricalNB label domain")
            _check_classes[Self.D](model.classes_, count, model._domain.value(), Self.Estimator)
            if len(model.category_count_) != features or len(model.feature_log_prob_) != features:
                raise Error("ModelSchemaError: invalid CategoricalNB number of feature tables")
            writer.integer(features)
            _write_classes[Self.D](writer, model.classes_, count)
            _write_count_warning(writer, model.fit_warnings, Self.Estimator)
            _write_array(archive, "class_count_", model.class_count_, [count])
            _write_array(archive, "class_log_prior_", model.class_log_prior_, [count])
            _write_array(archive, "n_categories_", model.n_categories_, [features])
            _check_count_priors[Self.Estimator](model.class_count_, model.class_log_prior_)
            # Pack ragged per-feature tables in feature and row-major order.
            # A fixed payload count allows wide inputs within the archive limits.
            var counts_writer = BinaryWriter()
            var probabilities_writer = BinaryWriter()
            var values = 0
            for f in range(features):
                ref counts = model.category_count_[f]
                ref log_probs = model.feature_log_prob_[f]
                if len(counts.shape) != 2 or counts.shape[0] != count or counts.shape != log_probs.shape:
                    raise Error("ModelSchemaError: invalid CategoricalNB feature table shape")
                var width = counts.shape[1]
                writer.integer(width)
                _check_categorical_statistics(counts, log_probs, model.n_categories_.item(f))
                for c in range(count):
                    for k in range(width):
                        counts_writer.scalar(counts.item(c, k))
                        probabilities_writer.scalar(log_probs.item(c, k))
                values += counts.size
            comptime code = dtype_code[DType.float64]()
            archive.payloads.append(ArrayPayload("category_count_", code, [values], counts_writer^.finish()))
            archive.payloads.append(ArrayPayload("feature_log_prob_", code, [values], probabilities_writer^.finish()))
        elif (
            features != 0 or count != 0 or model._domain or len(model.fit_warnings) != 0
            or model.class_count_.shape.tolist() != [0] or model.class_log_prior_.shape.tolist() != [0]
            or model.n_categories_.shape.tolist() != [0]
            or len(model.category_count_) != 0 or len(model.feature_log_prob_) != 0
        ):
            raise Error("ModelSchemaError: unfitted CategoricalNB has learned state")
        _write_metadata(writer, archive, model._metadata, model.feature_names_in_, features)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = Self.StateVersion
        if archive.schema_version in (_WEIGHT_REQUEST_VERSION, _FEATURE_NAMES_VERSION):
            version = archive.schema_version
        archive.expect(Self.TypeId, version)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != _count_input_dtype_tag[Self.dtype]():
            raise Error("ModelSchemaError: CategoricalNB dtype mismatch")
        if reader.uint() != _domain_tag[Self.D]():
            raise Error("ModelSchemaError: CategoricalNB label domain mismatch")
        var config = _read_discrete_config(reader)
        var min_categories: Optional[Int] = None
        if reader.boolean():
            min_categories = reader.integer()
        var min_categories_values: Optional[List[Int]] = None
        if reader.boolean():
            var length = reader.count(reader.remaining() // 8)
            var minima = List[Int](capacity=length)
            for _ in range(length):
                minima.append(reader.integer())
            min_categories_values = minima^
        var model = Self.Model(
            alpha=config.alpha, force_alpha=config.force_alpha, fit_prior=config.fit_prior,
            class_prior=config.class_prior.copy(), min_categories=min_categories,
            min_categories_values=min_categories_values^,
        )
        var fitted = reader.boolean()
        if fitted:
            if len(archive.payloads) != 5:
                raise Error("ModelSchemaError: unexpected CategoricalNB arrays")
            var features = reader.integer()
            if features < 1 or features > reader.remaining() // 8:
                raise Error("ModelSchemaError: invalid CategoricalNB feature count")
            var classes = _read_classes[Self.D](reader)
            var count = Self.D.check_target(classes)
            var domain = Self.D.from_classes(classes)
            _check_classes[Self.D](classes, count, domain, Self.Estimator)
            model.fit_warnings = _read_count_warning(reader, Self.Estimator)
            model.class_count_ = _read_array[DType.float64](archive, "class_count_", [count])
            model.class_log_prior_ = _read_array[DType.float64](archive, "class_log_prior_", [count])
            model.n_categories_ = _read_array[DType.int64](archive, "n_categories_", [features])
            _check_count_priors[Self.Estimator](model.class_count_, model.class_log_prior_)
            var counts_payload = archive.array("category_count_")
            var probabilities_payload = archive.array("feature_log_prob_")
            comptime code = dtype_code[DType.float64]()
            if (
                counts_payload.dtype != code or probabilities_payload.dtype != code
                or len(counts_payload.shape) != 1 or counts_payload.shape != probabilities_payload.shape
            ):
                raise Error("ModelSchemaError: invalid packed CategoricalNB feature tables")
            var counts_reader = BinaryReader(Span(counts_payload.data))
            var probabilities_reader = BinaryReader(Span(probabilities_payload.data))
            for f in range(features):
                var width = reader.integer()
                if width < 1 or width > counts_reader.remaining() // 8 // count:
                    raise Error("ModelSchemaError: invalid CategoricalNB feature table width")
                var counts = zeros[DType.float64](Shape(count, width))
                var log_probs = zeros[DType.float64](Shape(count, width))
                for c in range(count):
                    for k in range(width):
                        counts[Item(c, k)] = counts_reader.scalar[DType.float64]()
                        log_probs[Item(c, k)] = probabilities_reader.scalar[DType.float64]()
                _check_categorical_statistics(counts, log_probs, model.n_categories_.item(f))
                model.category_count_.append(counts^)
                model.feature_log_prob_.append(log_probs^)
            counts_reader.finish()
            probabilities_reader.finish()
            model.n_features_in_ = features
            model.classes_ = classes^
            model._domain = domain^
        elif len(archive.payloads) != 0:
            raise Error("ModelSchemaError: unexpected CategoricalNB arrays")
        model._metadata = _read_metadata(reader, archive)
        if version == _FEATURE_NAMES_VERSION:
            model.feature_names_in_ = _read_feature_names(reader, model.n_features_in_)
        reader.finish()
        return model^
