"""Dummy-owned experimental state schemas for Moxie's native model container."""
import numojo as nm
from numojo import NDArray, Shape, Item
from moxie.dummy import DummyRegressor, DummyClassifier, Label
from moxie.serialization import ModelSchema, ModelArchive, BinaryWriter, BinaryReader, dtype_code


def _optional_int(mut writer: BinaryWriter, value: Optional[Int]):
    writer.boolean(Bool(value))
    if value:
        writer.integer(value.value())


def _read_optional_int(mut reader: BinaryReader[_]) raises -> Optional[Int]:
    if reader.boolean():
        return reader.integer()
    return None


def _array[dtype: DType](mut archive: ModelArchive, name: String, value: NDArray[dtype]) raises:
    # Dummy state is rank 1/2. Logical indexing honors strided constructor arrays.
    if value.ndim < 1 or value.ndim > 2:
        raise Error("ModelSchemaError: dummy array requires rank 1 or 2")
    var shape = List[Int]()
    for i in range(value.ndim):
        shape.append(value.shape[i])
    var values = List[Scalar[dtype]]()
    for i in range(value.shape[0]):
        if value.ndim == 1:
            values.append(value.item(i))
        else:
            for j in range(value.shape[1]):
                values.append(value.item(i, j))
    archive.add_array[dtype](name, shape, values)


def _read_array[dtype: DType](archive: ModelArchive, name: String) raises -> NDArray[dtype]:
    var payload = archive.array(name)
    if len(payload.shape) < 1 or len(payload.shape) > 2:
        raise Error("ModelSchemaError: dummy array requires rank 1 or 2")
    var values = payload.values[dtype]()
    var shape = Shape(payload.shape[0])
    if len(payload.shape) == 2:
        shape = Shape(payload.shape[0], payload.shape[1])
    var array = nm.zeros[dtype](shape)
    for i in range(len(values)):
        array.unsafe_set(i, values[i])
    return array^


def _features(value: Optional[Int]) raises:
    if value and value.value() < 0:
        raise Error("ModelSchemaError: invalid feature count")


struct DummyRegressorSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = DummyRegressor[Self.dtype]

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive("moxie.dummy.DummyRegressor", 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.string(model.strategy)
        writer.boolean(Bool(model.quantile))
        if model.quantile:
            writer.scalar[DType.float64](model.quantile.value())
        writer.boolean(Bool(model.constant))
        if model.constant:
            _array(archive, "constant", model.constant.value())
        writer.boolean(model.__sklearn_is_fitted__())
        _optional_int(writer, model.n_features_in_)
        _optional_int(writer, model.n_outputs_)
        if model.constant_:
            _array(archive, "constant_", model.constant_.value())
        archive.manifest = writer^.finish()
        # Validate the same schema on both sides of the boundary.
        _ = Self.decode(archive)
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.validate()
        archive.expect("moxie.dummy.DummyRegressor", 1)
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != UInt64(dtype_code[Self.dtype]()):
            raise Error("ModelSchemaError: regressor dtype mismatch")
        var model = Self.Model(strategy=reader.string())
        if reader.boolean():
            model.quantile = reader.scalar[DType.float64]()
        var expected = 0
        if reader.boolean():
            model.constant = _read_array[Self.dtype](archive, "constant")
            expected += 1
        var fitted = reader.boolean()
        model.n_features_in_ = _read_optional_int(reader)
        model.n_outputs_ = _read_optional_int(reader)
        _features(model.n_features_in_)
        if fitted:
            if not model.n_outputs_ or model.n_outputs_.value() < 1:
                raise Error("ModelSchemaError: invalid output count")
            var learned = _read_array[Self.dtype](archive, "constant_")
            if learned.ndim != 2 or learned.shape[0] != 1 or learned.shape[1] != model.n_outputs_.value():
                raise Error("ModelSchemaError: learned constant shape mismatch")
            model.constant_ = learned^
            expected += 1
        elif model.n_outputs_ or model.n_features_in_:
            raise Error("ModelSchemaError: unfitted regressor has learned metadata")
        if len(archive.payloads) != expected:
            raise Error("ModelSchemaError: unexpected arrays")
        reader.finish()
        return model^


def _labels(mut writer: BinaryWriter, labels: List[Label]) raises:
    writer.uint(UInt64(len(labels)))
    for label in labels:
        writer.uint(UInt64(label.kind))
        if label.kind == 0:
            writer.integer(label.integer)
        elif label.kind == 1:
            writer.scalar[DType.float64](label.number)
        elif label.kind == 2:
            writer.string(label.text)
        else:
            raise Error("ModelSchemaError: unsupported label kind")


def _read_labels(mut reader: BinaryReader[_]) raises -> List[Label]:
    var count = reader.count(reader.remaining() // 16)
    var result = List[Label]()
    for _ in range(count):
        var kind = reader.count(2)
        if kind == 0:
            result.append(Label(reader.integer()))
        elif kind == 1:
            result.append(Label(reader.scalar[DType.float64]()))
        else:
            result.append(Label(reader.string()))
    return result^


struct DummyClassifierSchema[multioutput: Bool = False, dtype: DType = DType.float64](ModelSchema):
    comptime Model = DummyClassifier[Self.multioutput, Self.dtype]

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive("moxie.dummy.DummyClassifier", 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.boolean(Self.multioutput)
        writer.string(model.strategy)
        _optional_int(writer, model.random_state)
        # None means the process-global RNG at inference; external borrowed RNGs
        # are not owned by this estimator and are never captured in this file.
        writer.boolean(Bool(model.constant))
        if model.constant:
            _labels(writer, model.constant.value())
        writer.boolean(model.__sklearn_is_fitted__())
        _optional_int(writer, model.n_features_in_)
        writer.string(model._strategy)
        if model.n_outputs_:
            var outputs = model.n_outputs_.value()
            if outputs < 1 or outputs > 1024:
                raise Error("ModelSchemaError: invalid output count")
            comptime if Self.multioutput:
                if len(rebind[List[List[Label]]](model.classes_)) != outputs or len(rebind[List[Int]](model.n_classes_)) != outputs or len(rebind[List[NDArray[DType.float64]]](model.class_prior_)) != outputs:
                    raise Error("ModelSchemaError: inconsistent output containers")
            else:
                if outputs != 1:
                    raise Error("ModelSchemaError: single-output classifier")
            writer.uint(UInt64(model.n_outputs_.value()))
            if not model.sparse_output_ or model.sparse_output_.value():
                raise Error("ModelSchemaError: expected dense fitted state")
            for j in range(model.n_outputs_.value()):
                comptime if Self.multioutput:
                    _labels(writer, rebind[List[List[Label]]](model.classes_)[j])
                    writer.integer(rebind[List[Int]](model.n_classes_)[j])
                    _array(archive, String("class_prior_/") + String(j), rebind[List[NDArray[DType.float64]]](model.class_prior_)[j])
                else:
                    _labels(writer, rebind[List[Label]](model.classes_))
                    writer.integer(rebind[Int](model.n_classes_))
                    _array(archive, "class_prior_/0", rebind[NDArray[DType.float64]](model.class_prior_))
        archive.manifest = writer^.finish()
        _ = Self.decode(archive)
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.validate()
        archive.expect("moxie.dummy.DummyClassifier", 1)
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != UInt64(dtype_code[Self.dtype]()) or reader.boolean() != Self.multioutput:
            raise Error("ModelSchemaError: classifier type parameters mismatch")
        var model = Self.Model(strategy=reader.string())
        model.random_state = _read_optional_int(reader)
        if reader.boolean():
            model.constant = _read_labels(reader)
        var fitted = reader.boolean()
        model.n_features_in_ = _read_optional_int(reader)
        _features(model.n_features_in_)
        model._strategy = reader.string()
        var outputs = 0
        if fitted:
            outputs = reader.count(len(archive.payloads))
            if outputs < 1 or (not Self.multioutput and outputs != 1):
                raise Error("ModelSchemaError: invalid output count")
            if model._strategy != "prior" and model._strategy != "most_frequent" and model._strategy != "uniform" and model._strategy != "stratified" and model._strategy != "constant":
                raise Error("ModelSchemaError: invalid fitted strategy")
            var classes = List[List[Label]]()
            var counts = List[Int]()
            var priors = List[NDArray[DType.float64]]()
            for j in range(outputs):
                var labels = _read_labels(reader)
                var count = reader.integer()
                if count < 1 or count != len(labels):
                    raise Error("ModelSchemaError: class count mismatch")
                for k in range(1, count):
                    if labels[k].kind != labels[0].kind or not labels[k - 1] < labels[k]:
                        raise Error("ModelSchemaError: classes must be homogeneous, sorted and unique")
                var prior = _read_array[DType.float64](archive, String("class_prior_/") + String(j))
                if prior.ndim != 1 or prior.shape[0] != count:
                    raise Error("ModelSchemaError: prior shape mismatch")
                classes.append(labels^)
                counts.append(count)
                priors.append(prior^)
            comptime if Self.multioutput:
                model.classes_ = rebind[Self.Model.Classes](classes).copy()
                model.n_classes_ = rebind[Self.Model.ClassCounts](counts).copy()
                model.class_prior_ = rebind[Self.Model.Probabilities](priors).copy()
            else:
                model.classes_ = rebind[Self.Model.Classes](classes[0]).copy()
                model.n_classes_ = rebind[Self.Model.ClassCounts](counts[0]).copy()
                model.class_prior_ = rebind[Self.Model.Probabilities](priors[0]).copy()
            model.n_outputs_ = outputs
            model.sparse_output_ = False
        elif model.n_features_in_ or model._strategy != "":
            raise Error("ModelSchemaError: unfitted classifier has learned metadata")
        if len(archive.payloads) != outputs:
            raise Error("ModelSchemaError: unexpected arrays")
        reader.finish()
        return model^
