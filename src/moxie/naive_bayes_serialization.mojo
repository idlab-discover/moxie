"""Native archive schemas for dense Naive Bayes estimators."""

from std.math import isfinite, isnan
from std.sys import size_of
from numojo import NDArray, Item, array
from moxie._labels import StringTarget
from moxie._naive_bayes_dense import ClassDomain
from moxie.naive_bayes import GaussianNB, IntListDomain, NumericDomain, StringDomain
from moxie.serialization import BinaryReader, BinaryWriter, ModelArchive, ModelSchema, dtype_code


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


def _check_classes[D: ClassDomain](classes: D.Target, count: Int, domain: D) raises:
    if count < 1 or domain.n_classes() != count:
        raise Error("ModelSchemaError: invalid GaussianNB class count")
    var ordered = domain.class_values()
    for i in range(count):
        if not D.equal_at(classes, ordered, i):
            raise Error("ModelSchemaError: GaussianNB classes must match the sorted label domain")


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
    """Version 1 state for GaussianNB with a built-in label domain."""

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
            _check_classes[Self.D](model.classes_, count, model._domain.value())
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
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TypeId, Self.StateVersion)
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
            _check_classes[Self.D](classes, count, domain)
            model.class_count_ = _read_array[Self.dtype](archive, "class_count_", [count])
            model.class_prior_ = _read_array[Self.dtype](archive, "class_prior_", [count])
            model.theta_ = _read_array[Self.dtype](archive, "theta_", [count, features])
            model.var_ = _read_array[Self.dtype](archive, "var_", [count, features])
            _check_counts_and_priors(model.class_count_, model.class_prior_, epsilon)
            model.epsilon_ = epsilon
            model.n_features_in_ = features
            model.classes_ = classes^
            model._domain = domain^
        reader.finish()
        return model^
