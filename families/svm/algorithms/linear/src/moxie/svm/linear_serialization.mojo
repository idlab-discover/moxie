"""Experimental native archives for LinearSVC and LinearSVR."""

import numojo as nm
from std.math import floor, isfinite
from moxie._random import RandomState
from moxie.metadata_routing import FitScoreMetadataRequest, SampleWeightRequest
from moxie.serialization import BinaryReader, BinaryWriter, ModelArchive, ModelSchema
from moxie.svm._contracts import Dense
from moxie.svm._linear import ClassWeight, Dual, LinearSVC, LinearSVR, _linear_fit_warnings


def _label_type[Label: Copyable & Comparable & Deinitable]() -> String:
    comptime if Label == String:
        return "string"
    elif Label == Int:
        return "int"
    elif Label == Int32:
        return "int32"
    elif Label == Int64:
        return "int64"
    elif Label == Float32:
        return "float32"
    elif Label == Float64:
        return "float64"
    else:
        comptime assert False, "unsupported SVM archive label type"


def _write_labels[Label: Copyable & Comparable & Deinitable](
    mut writer: BinaryWriter, labels: List[Label]
):
    writer.uint(UInt64(len(labels)))
    for label in labels:
        comptime if Label == String:
            writer.string(rebind[String](label))
        elif Label == Float32:
            writer.scalar[DType.float32](rebind[Float32](label))
        elif Label == Float64:
            writer.scalar[DType.float64](rebind[Float64](label))
        elif Label == Int:
            writer.integer(rebind[Int](label))
        elif Label == Int32:
            writer.scalar[DType.int32](rebind[Int32](label))
        elif Label == Int64:
            writer.scalar[DType.int64](rebind[Int64](label))


def _read_labels[Label: Copyable & Comparable & Deinitable](
    mut reader: BinaryReader[_],
) raises -> List[Label]:
    var width = 4
    comptime if Label != Float32 and Label != Int32:
        width = 8
    var count = reader.count(max(0, reader.remaining() - 8) // width)
    var labels = List[Label]()
    for _ in range(count):
        var label: Label
        comptime if Label == String:
            label = rebind[Label](reader.string(reader.remaining())).copy()
        elif Label == Float32:
            label = rebind[Label](reader.scalar[DType.float32]()).copy()
        elif Label == Float64:
            label = rebind[Label](reader.scalar[DType.float64]()).copy()
        elif Label == Int:
            label = rebind[Label](reader.integer()).copy()
        elif Label == Int32:
            label = rebind[Label](reader.scalar[DType.int32]()).copy()
        elif Label == Int64:
            label = rebind[Label](reader.scalar[DType.int64]()).copy()
        else:
            raise Error("ModelSchemaError: unsupported SVM archive label type")
        labels.append(label^)
    return labels^


def _class_rows[Label: Copyable & Comparable & Deinitable](
    classes: List[Label], fitted: Bool, name: String = "LinearSVC",
) raises -> Int:
    if not fitted:
        if len(classes) != 0:
            raise Error("ModelSchemaError: unfitted " + name + " has classes")
        return 0
    if len(classes) < 2:
        raise Error("ModelSchemaError: fitted " + name + " needs at least two classes")
    for i in range(len(classes)):
        comptime if Label == Float64 or Label == Float32:
            comptime dtype = DType.float64 if Label == Float64 else DType.float32
            var label = rebind[Scalar[dtype]](classes[i])
            if not isfinite(label) or label != floor(label):
                raise Error("ModelSchemaError: non-discrete " + name + " class")
        if i > 0 and not (classes[i - 1] < classes[i]):
            raise Error("ModelSchemaError: " + name + " classes must be strictly sorted")
    return 1 if len(classes) == 2 else len(classes)


def _write_array(mut archive: ModelArchive, name: String, array: Dense, shape: List[Int]) raises:
    if array.ndim != len(shape):
        raise Error("ModelSchemaError: invalid rank for " + name)
    for i in range(len(shape)):
        if array.shape[i] != shape[i]:
            raise Error("ModelSchemaError: invalid shape for " + name)
    if array.size == 0:
        return
    var values = List[Float64](capacity=array.size)
    var width = shape[len(shape) - 1]
    var stride = Int(array.strides[len(shape) - 1])
    for i in range(array.size):
        var offset = i % width * stride
        if array.ndim == 2:
            offset += i // width * Int(array.strides[0])
        var value = array.unsafe_get(offset)
        if not isfinite(value):
            raise Error("ModelSchemaError: nonfinite values in " + name)
        values.append(value)
    archive.add_array[DType.float64](name, shape, values)


def _read_array(archive: ModelArchive, name: String, shape: List[Int]) raises -> Dense:
    var payload = archive.array(name)
    if payload.shape != shape:
        raise Error("ModelSchemaError: invalid shape for " + name)
    var values = payload.values[DType.float64]()
    var array = nm.zeros[DType.float64](nm.Shape(shape))
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error("ModelSchemaError: nonfinite values in " + name)
        array.unsafe_set(i, values[i])
    return array^


@fieldwise_init
struct _FitState(Copyable, Movable):
    var features: Int
    var iterations: Int
    var status: Int

    def validate(self) raises:
        if (
            self.features < 0
            or self.iterations < 0
            or self.status < 0
            or self.status > 1
            or (self.features == 0 and (self.iterations != 0 or self.status != 0))
        ):
            raise Error("ModelSchemaError: invalid linear SVM fit metadata")

    def write(self, mut writer: BinaryWriter) raises:
        self.validate()
        writer.integer(self.features)
        writer.integer(self.iterations)
        writer.integer(self.status)

    @staticmethod
    def read(mut reader: BinaryReader[_]) raises -> Self:
        var result = Self(reader.integer(), reader.integer(), reader.integer())
        result.validate()
        return result^


def _write_rng(mut writer: BinaryWriter, rng: RandomState) raises:
    writer.integer(rng.state_position())
    for i in range(624):
        writer.scalar[DType.uint32](rng.state_word(i))
    writer.boolean(rng.has_gauss())
    writer.scalar[DType.float64](rng.cached_gauss())


def _read_rng(mut reader: BinaryReader[_]) raises -> RandomState:
    var position = reader.integer()
    if position < 0 or position > 624:
        raise Error("ModelSchemaError: invalid MT19937 position")
    var rng = RandomState(0)
    var nonzero = False
    for i in range(624):
        var word = reader.scalar[DType.uint32]()
        rng._key[i] = word
        nonzero = nonzero or word != 0
    var has_gauss = reader.boolean()
    var gauss = reader.scalar[DType.float64]()
    if not nonzero or not isfinite(gauss) or (not has_gauss and gauss != 0):
        raise Error("ModelSchemaError: invalid MT19937 state")
    rng._pos = position
    rng._has_gauss = has_gauss
    rng._gauss = gauss
    return rng^


def _write_request(mut writer: BinaryWriter, request: SampleWeightRequest):
    var value = request.value()
    if value.isa[NoneType]():
        writer.integer(0)
    elif value.isa[Bool]():
        writer.integer(2 if value[Bool] else 1)
    else:
        writer.integer(3)
        writer.string(value[String])


def _read_request(mut reader: BinaryReader[_]) raises -> SampleWeightRequest:
    var request = SampleWeightRequest()
    var tag = reader.integer()
    if tag == 1 or tag == 2:
        request.set(tag == 2)
    elif tag == 3:
        var alias_name = reader.string(reader.remaining())
        request.set(alias_name)
        if not request.value().isa[String]():
            raise Error("ModelSchemaError: invalid linear SVM metadata alias")
    elif tag != 0:
        raise Error("ModelSchemaError: invalid linear SVM metadata request tag")
    return request^


struct LinearSVCSchema[Label: Copyable & Comparable & Deinitable = Float64](ModelSchema):
    """Version 2 state for a dense classifier with the specified label type."""

    comptime Model = LinearSVC[Self.Label]
    comptime TypeId = "moxie.svm.LinearSVC"
    comptime StateVersion = 2

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        writer.string(_label_type[Self.Label]())
        writer.string(model.penalty)
        writer.string(model.loss)
        writer.string(model.multi_class)
        writer.string(model.class_weight.mode)
        _write_labels(writer, model.class_weight.labels)
        writer.uint(UInt64(len(model.class_weight.values)))
        for value in model.class_weight.values:
            writer.scalar[DType.float64](value)
        writer.string(model.dual.value)
        writer.scalar[DType.float64](model.tol)
        writer.scalar[DType.float64](model.C)
        writer.boolean(model.fit_intercept)
        writer.scalar[DType.float64](model.intercept_scaling)
        writer.integer(model.verbose)
        writer.integer(model.random_state)
        writer.integer(model.max_iter)
        var state = _FitState(model.n_features_in_, model.n_iter_, model.fit_status_)
        state.write(writer)
        _write_rng(writer, model._rng)
        _write_labels(writer, model.classes_)
        var rows = _class_rows(model.classes_, state.features > 0)
        var coef_shape: List[Int] = [rows, state.features] if rows else [0]
        _write_array(archive, "coef_", model.coef_, coef_shape)
        _write_array(archive, "intercept_", model.intercept_, [rows])
        _write_request(writer, model._metadata.fit)
        _write_request(writer, model._metadata.score)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        if archive.schema_version == 1:
            archive.expect(Self.TypeId, 1)
        else:
            archive.expect(Self.TypeId, Self.StateVersion)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.string() != _label_type[Self.Label]():
            raise Error("ModelSchemaError: LinearSVC label type mismatch")
        var penalty = reader.string(reader.remaining())
        var loss = reader.string(reader.remaining())
        var multi_class = reader.string(reader.remaining())
        var weights = ClassWeight[Self.Label](reader.string(reader.remaining()))
        weights.labels = _read_labels[Self.Label](reader)
        var count = reader.count(max(0, reader.remaining() - 8) // 8)
        for _ in range(count):
            weights.values.append(reader.scalar[DType.float64]())
        var model = Self.Model(
            penalty=penalty,
            loss=loss,
            multi_class=multi_class,
            class_weight=weights,
            dual=Dual(reader.string(reader.remaining())),
            tol=reader.scalar[DType.float64](),
            C=reader.scalar[DType.float64](),
            fit_intercept=reader.boolean(),
            intercept_scaling=reader.scalar[DType.float64](),
            verbose=reader.integer(),
            random_state=reader.integer(),
            max_iter=reader.integer(),
        )
        var state = _FitState.read(reader)
        model._rng = _read_rng(reader)
        model.classes_ = _read_labels[Self.Label](reader)
        if archive.schema_version == Self.StateVersion:
            model._metadata = FitScoreMetadataRequest(
                _read_request(reader), _read_request(reader)
            )
        reader.finish()
        var rows = _class_rows(model.classes_, state.features > 0)
        if len(archive.payloads) != (2 if rows else 0):
            raise Error("ModelSchemaError: unexpected LinearSVC array count")
        if rows:
            model.coef_ = _read_array(archive, "coef_", [rows, state.features])
            model.intercept_ = _read_array(archive, "intercept_", [rows])
        model.n_features_in_ = state.features
        model.n_iter_ = state.iterations
        model.fit_status_ = state.status
        model.fit_warnings = _linear_fit_warnings(state.status)
        return model^


struct LinearSVRSchema(ModelSchema):
    """Version 2 state for the dense linear regressor."""

    comptime Model = LinearSVR
    comptime TypeId = "moxie.svm.LinearSVR"
    comptime StateVersion = 2

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        writer.scalar[DType.float64](model.epsilon)
        writer.string(model.loss)
        writer.string(model.dual.value)
        writer.scalar[DType.float64](model.tol)
        writer.scalar[DType.float64](model.C)
        writer.boolean(model.fit_intercept)
        writer.scalar[DType.float64](model.intercept_scaling)
        writer.integer(model.verbose)
        writer.integer(model.random_state)
        writer.integer(model.max_iter)
        var state = _FitState(model.n_features_in_, model.n_iter_, model.fit_status_)
        state.write(writer)
        _write_rng(writer, model._rng)
        _write_array(archive, "coef_", model.coef_, [state.features])
        _write_array(archive, "intercept_", model.intercept_, [1 if state.features else 0])
        _write_request(writer, model._metadata.fit)
        _write_request(writer, model._metadata.score)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        if archive.schema_version == 1:
            archive.expect(Self.TypeId, 1)
        else:
            archive.expect(Self.TypeId, Self.StateVersion)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        var model = Self.Model(
            epsilon=reader.scalar[DType.float64](),
            loss=reader.string(reader.remaining()),
            dual=Dual(reader.string(reader.remaining())),
            tol=reader.scalar[DType.float64](),
            C=reader.scalar[DType.float64](),
            fit_intercept=reader.boolean(),
            intercept_scaling=reader.scalar[DType.float64](),
            verbose=reader.integer(),
            random_state=reader.integer(),
            max_iter=reader.integer(),
        )
        var state = _FitState.read(reader)
        model._rng = _read_rng(reader)
        if archive.schema_version == Self.StateVersion:
            model._metadata = FitScoreMetadataRequest(
                _read_request(reader), _read_request(reader)
            )
        reader.finish()
        if len(archive.payloads) != (2 if state.features else 0):
            raise Error("ModelSchemaError: unexpected LinearSVR array count")
        if state.features:
            model.coef_ = _read_array(archive, "coef_", [state.features])
            model.intercept_ = _read_array(archive, "intercept_", [1])
        model.n_features_in_ = state.features
        model.n_iter_ = state.iterations
        model.fit_status_ = state.status
        model.fit_warnings = _linear_fit_warnings(state.status)
        return model^
