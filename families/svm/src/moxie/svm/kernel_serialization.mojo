"""Experimental native archives for the five dense LIBSVM estimators."""

import numojo as nm
from std.math import isfinite
from moxie.metadata_routing import FitScoreMetadataRequest, SampleWeightRequest
from moxie.serialization import BinaryReader, BinaryWriter, ModelArchive, ModelSchema
from moxie.svm._c_svm import SVC, SVR
from moxie.svm._classification import (
    Gamma, Probability, _KernelFitDiagnostics, checked_int32, support_counts,
)
from moxie.svm._contracts import Dense, KernelModel, KernelSpec, TrainingSnapshot
from moxie.svm._linear import ClassWeight
from moxie.svm._nu_svm import NuSVC, NuSVR
from moxie.svm._one_class import OneClassSVM
from moxie.svm.linear_serialization import (
    _class_rows, _label_type, _read_labels, _read_request, _read_rng,
    _write_labels, _write_request, _write_rng,
)

comptime _PublicIndices = nm.NDArray[DType.int32]


@fieldwise_init
struct _KernelConfig(Copyable, Movable):
    var kernel: String
    var degree: Int
    var gamma: Gamma
    var coef0: Float64
    var tol: Float64
    var shrinking: Bool
    var cache_size: Float64
    var verbose: Bool
    var max_iter: Int

    def write(self, mut writer: BinaryWriter):
        writer.string(self.kernel)
        writer.integer(self.degree)
        writer.string(self.gamma.mode)
        writer.scalar[DType.float64](self.gamma.value)
        writer.boolean(self.gamma.numeric)
        writer.scalar[DType.float64](self.coef0)
        writer.scalar[DType.float64](self.tol)
        writer.boolean(self.shrinking)
        writer.scalar[DType.float64](self.cache_size)
        writer.boolean(self.verbose)
        writer.integer(self.max_iter)

    @staticmethod
    def read(mut reader: BinaryReader[_]) raises -> Self:
        var kernel = reader.string(reader.remaining())
        var degree = reader.integer()
        var gamma = Gamma(reader.string(reader.remaining()))
        gamma.value = reader.scalar[DType.float64]()
        gamma.numeric = reader.boolean()
        return Self(
            kernel^, degree, gamma^, reader.scalar[DType.float64](),
            reader.scalar[DType.float64](), reader.boolean(),
            reader.scalar[DType.float64](), reader.boolean(), reader.integer(),
        )


@fieldwise_init
struct _ClassifierConfig[Label: Copyable & Comparable & Deinitable](Movable):
    var probability: Probability
    var class_weight: ClassWeight[Self.Label]
    var decision_function_shape: String
    var break_ties: Bool
    var random_state: Int

    def write(self, mut writer: BinaryWriter):
        writer.string(_label_type[Self.Label]())
        writer.string(self.probability.mode)
        writer.boolean(self.probability.boolean)
        writer.string(self.class_weight.mode)
        _write_labels(writer, self.class_weight.labels)
        writer.uint(UInt64(len(self.class_weight.values)))
        for value in self.class_weight.values:
            writer.scalar[DType.float64](value)
        writer.string(self.decision_function_shape)
        writer.boolean(self.break_ties)
        writer.integer(self.random_state)

    @staticmethod
    def read(mut reader: BinaryReader[_]) raises -> Self:
        if reader.string() != _label_type[Self.Label]():
            raise Error("ModelSchemaError: kernel SVM label type mismatch")
        var probability = Probability(reader.string(reader.remaining()))
        probability.boolean = reader.boolean()
        var weight = ClassWeight[Self.Label](reader.string(reader.remaining()))
        weight.labels = _read_labels[Self.Label](reader)
        var count = reader.count(max(0, reader.remaining() - 8) // 8)
        for _ in range(count):
            weight.values.append(reader.scalar[DType.float64]())
        return Self(
            probability^, weight^, reader.string(reader.remaining()),
            reader.boolean(), reader.integer(),
        )


def _write_array[dtype: DType](
    mut archive: ModelArchive, name: String, array: nm.NDArray[dtype],
    shape: List[Int],
) raises:
    if array.ndim != len(shape):
        raise Error("ModelSchemaError: invalid rank for " + name)
    for axis in range(len(shape)):
        if array.shape[axis] != shape[axis]:
            raise Error("ModelSchemaError: invalid shape for " + name)
    var values = List[Scalar[dtype]](capacity=array.size)
    var width = shape[len(shape) - 1]
    var stride = Int(array.strides[len(shape) - 1])
    for i in range(array.size):
        var offset = i % width * stride
        if array.ndim == 2:
            offset += i // width * Int(array.strides[0])
        var value = array.unsafe_get(offset)
        comptime if dtype == DType.float64:
            if not isfinite(value):
                raise Error("ModelSchemaError: nonfinite values in " + name)
        values.append(value)
    archive.add_array[dtype](name, shape, values)


def _payload_index(archive: ModelArchive, name: String) raises -> Int:
    for i in range(len(archive.payloads)):
        if archive.payloads[i].name == name:
            return i
    raise Error("ModelSchemaError: missing array " + name)


def _read_array[dtype: DType](
    archive: ModelArchive, name: String, shape: List[Int],
) raises -> nm.NDArray[dtype]:
    ref payload = archive.payloads[_payload_index(archive, name)]
    if payload.shape != shape:
        raise Error("ModelSchemaError: invalid shape for " + name)
    var values = payload.values[dtype]()
    var result = nm.zeros[dtype](nm.Shape(shape))
    for i in range(len(values)):
        comptime if dtype == DType.float64:
            if not isfinite(values[i]):
                raise Error("ModelSchemaError: nonfinite values in " + name)
        result.unsafe_set(i, values[i])
    return result^


def _validate_geometry(kernel: KernelSpec, rows: Int, columns: Int, gram: Bool, classes: Int, status: Int) raises:
    kernel.validate()
    if (
        kernel.kind == "callable"
        or (gram and (kernel.kind != "precomputed" or rows != columns))
        or (kernel.kind == "precomputed" and kernel.gamma != 0)
        or rows <= 0 or columns <= 0 or classes > rows
        or status < 0 or status > 1
    ):
        raise Error("ModelSchemaError: invalid kernel SVM fitted metadata")


def _validate_support(model: KernelModel, public_support: _PublicIndices, rows: Int, classes: Int) raises:
    var size = model.support.size
    var seen = Dict[Int, Bool]()
    var end = 0
    for c in range(max(1, classes)):
        var count = Int(model.n_support.unsafe_get(c * Int(model.n_support.strides[0])))
        if count < 0 or count > size - end:
            raise Error("ModelSchemaError: invalid kernel SVM support counts")
        var previous = -1
        var previous_compact = -1
        for i in range(end, end + count):
            var original = Int(model.support.unsafe_get(i * Int(model.support.strides[0])))
            var compact = Int(public_support.unsafe_get(i * Int(public_support.strides[0])))
            if (
                original >= rows or original <= previous
                or compact <= previous_compact or compact > original
                or original in seen
            ):
                raise Error("ModelSchemaError: invalid kernel SVM support index")
            seen[original] = True
            previous = original
            previous_compact = compact
        end += count
    if end != size:
        raise Error("ModelSchemaError: inconsistent kernel SVM support total")
    for i in range(model.iterations.size):
        var count = model.iterations.unsafe_get(i * Int(model.iterations.strides[0]))
        if count < 0 or (classes > 0 and count > 2147483647):
            raise Error("ModelSchemaError: invalid kernel SVM iteration count")


def _write_fit(
    mut archive: ModelArchive, mut writer: BinaryWriter,
    kernel: KernelSpec, engine: Optional[KernelModel],
    snapshot: Optional[TrainingSnapshot], public_support: _PublicIndices,
    classes: Int, diagnostics: Optional[_KernelFitDiagnostics],
) raises:
    writer.boolean(Bool(engine))
    if not engine:
        if snapshot or classes != 0 or diagnostics:
            raise Error("ModelSchemaError: inconsistent unfitted kernel SVM")
        return
    if not snapshot:
        raise Error("ModelSchemaError: fitted kernel SVM is missing training state")
    ref model = engine.value()
    ref training = snapshot.value()
    var rows = training.n_train
    var columns = training.n_features
    _validate_geometry(kernel, rows, columns, training.precomputed, classes, model.fit_status)
    writer.string(kernel.kind)
    writer.scalar[DType.float64](kernel.gamma)
    writer.integer(kernel.degree)
    writer.scalar[DType.float64](kernel.coef0)
    writer.boolean(training.precomputed)
    writer.integer(model.fit_status)
    var supports = model.support.size
    var pairs = classes * (classes - 1) // 2 if classes else 1
    var calibration = model.prob_a.size
    if supports > rows or (calibration != 0 and (classes == 0 or calibration != pairs)):
        raise Error("ModelSchemaError: invalid kernel SVM fitted array size")
    var vectors: List[Int] = [0, 0] if kernel.kind == "precomputed" else [supports, columns]
    _write_array(archive, "training", training.x, [rows, columns])
    _write_array(archive, "support_", public_support, [supports])
    _write_array(archive, "support", model.support, [supports])
    _write_array(archive, "support_vectors", model.support_vectors, vectors)
    _write_array(archive, "n_support", model.n_support, [max(1, classes)])
    _write_array(archive, "dual_coef", model.dual_coef, [max(1, classes - 1), supports])
    _write_array(archive, "rho", model.rho, [pairs])
    _write_array(archive, "iterations", model.iterations, [pairs])
    _write_array(archive, "prob_a", model.prob_a, [calibration])
    _write_array(archive, "prob_b", model.prob_b, [calibration])
    _validate_support(model, public_support, rows, classes)
    writer.boolean(Bool(diagnostics))
    if diagnostics:
        ref fit = diagnostics.value()
        _validate_diagnostics(fit, model, classes)
        writer.integer(fit.max_iter)
        writer.boolean(fit.probability_explicit)


def _validate_diagnostics(fit: _KernelFitDiagnostics, model: KernelModel, classes: Int) raises:
    var calibrated = model.prob_a.size != 0
    if (
        fit.max_iter < -1
        or (model.fit_status == 1 and fit.max_iter < 0)
        or (fit.probability_explicit and classes == 0)
        or (calibrated and not fit.probability_explicit)
    ):
        raise Error("ModelSchemaError: invalid kernel SVM fit diagnostics")
    if fit.max_iter == -1:
        return
    var reached_limit = False
    for i in range(model.iterations.size):
        var iterations = Int(model.iterations.unsafe_get(i * Int(model.iterations.strides[0])))
        if iterations > fit.max_iter:
            raise Error("ModelSchemaError: invalid kernel SVM fit iteration limit")
        reached_limit |= iterations == fit.max_iter
    # For calibrated fits, a smaller CV subproblem can also reach the cap.
    if (reached_limit and model.fit_status == 0) or (
        model.fit_status == 1 and not calibrated and not reached_limit
    ):
        raise Error("ModelSchemaError: inconsistent kernel SVM fit iteration limit")


@fieldwise_init
struct _KernelFit(Movable):
    var kernel: KernelSpec
    var engine: Optional[KernelModel]
    var snapshot: Optional[TrainingSnapshot]
    var support: _PublicIndices
    var diagnostics: Optional[_KernelFitDiagnostics]


def _read_fit(
    archive: ModelArchive, mut reader: BinaryReader[_], classes: Int, extra_arrays: Int,
) raises -> _KernelFit:
    if not reader.boolean():
        if len(archive.payloads) or classes:
            raise Error("ModelSchemaError: unexpected unfitted kernel SVM state")
        return _KernelFit(
            KernelSpec("rbf", 0, 3, 0), None, None,
            nm.zeros[DType.int32](nm.Shape(0)), None,
        )
    if len(archive.payloads) != 10 + extra_arrays:
        raise Error("ModelSchemaError: unexpected fitted kernel SVM array count")
    var kernel = KernelSpec(
        reader.string(reader.remaining()), reader.scalar[DType.float64](),
        reader.integer(), reader.scalar[DType.float64](),
    )
    var gram = reader.boolean()
    var status = reader.integer()
    var shape = archive.payloads[_payload_index(archive, "training")].shape.copy()
    if len(shape) != 2:
        raise Error("ModelSchemaError: invalid training rank")
    var rows = shape[0]
    var columns = shape[1]
    _validate_geometry(kernel, rows, columns, gram, classes, status)
    var support_shape = archive.payloads[_payload_index(archive, "support")].shape.copy()
    if len(support_shape) != 1 or support_shape[0] > rows:
        raise Error("ModelSchemaError: invalid support shape")
    var supports = support_shape[0]
    var pairs = classes * (classes - 1) // 2 if classes else 1
    var probability_shape = archive.payloads[_payload_index(archive, "prob_a")].shape.copy()
    if len(probability_shape) != 1 or (
        probability_shape[0] != 0 and (classes == 0 or probability_shape[0] != pairs)
    ):
        raise Error("ModelSchemaError: invalid calibration shape")
    var vectors: List[Int] = [0, 0] if kernel.kind == "precomputed" else [supports, columns]
    var engine = KernelModel(
        _read_array[DType.int64](archive, "support", support_shape),
        _read_array[DType.float64](archive, "support_vectors", vectors),
        _read_array[DType.int64](archive, "n_support", [max(1, classes)]),
        _read_array[DType.float64](archive, "dual_coef", [max(1, classes - 1), supports]),
        _read_array[DType.float64](archive, "rho", [pairs]),
        _read_array[DType.int64](archive, "iterations", [pairs]),
        status,
        _read_array[DType.float64](archive, "prob_a", probability_shape),
        _read_array[DType.float64](archive, "prob_b", probability_shape),
    )
    var support = _read_array[DType.int32](archive, "support_", support_shape)
    _validate_support(engine, support, rows, classes)
    var diagnostics: Optional[_KernelFitDiagnostics] = None
    if archive.schema_version >= 2 and reader.boolean():
        var fit = _KernelFitDiagnostics(reader.integer(), reader.boolean())
        _validate_diagnostics(fit, engine, classes)
        diagnostics = fit^
    var training = _read_array[DType.float64](archive, "training", shape)
    return _KernelFit(
        kernel^, engine^, TrainingSnapshot(training, gram), support^, diagnostics^,
    )


def _class_count[Label: Copyable & Comparable & Deinitable](classes: List[Label], fitted: Bool) raises -> Int:
    _ = _class_rows(classes, fitted, "kernel SVM")
    return len(classes)


def _validate_class_weights(weights: Dense) raises:
    for i in range(weights.size):
        if weights.unsafe_get(i * Int(weights.strides[0])) < 0:
            raise Error("ModelSchemaError: negative learned kernel SVM class weight")


def _read_class_weights(archive: ModelArchive, classes: Int) raises -> Dense:
    var weights = _read_array[DType.float64](archive, "class_weight_", [classes])
    _validate_class_weights(weights)
    return weights^


def _read_supervised_requests(
    mut reader: BinaryReader[_], version: Int,
) raises -> FitScoreMetadataRequest:
    if version < 3:
        return FitScoreMetadataRequest(SampleWeightRequest(), SampleWeightRequest())
    return FitScoreMetadataRequest(_read_request(reader), _read_request(reader))


struct SVCSchema[Label: Copyable & Comparable & Deinitable = Float64](ModelSchema):
    """Version 3 state for a dense SVC."""

    comptime Model = SVC[Self.Label]
    comptime TypeId = "moxie.svm.SVC"
    comptime StateVersion = 3

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        _KernelConfig(
            model.kernel.copy(), model.degree, model.gamma.copy(), model.coef0,
            model.tol, model.shrinking, model.cache_size, model.verbose, model.max_iter,
        ).write(writer)
        writer.scalar[DType.float64](model.C)
        _ClassifierConfig(
            model.probability.copy(), model.class_weight.copy(),
            model.decision_function_shape.copy(), model.break_ties, model.random_state,
        ).write(writer)
        _write_labels(writer, model.classes_)
        var classes = _class_count(model.classes_, Bool(model._model))
        _write_rng(writer, model._rng)
        _write_fit(
            archive, writer, model._kernel, model._model, model._snapshot,
            model.support_, classes, model._fit_diagnostics,
        )
        if classes:
            _write_array(archive, "class_weight_", model.class_weight_, [classes])
            _validate_class_weights(model.class_weight_)
        _write_request(writer, model._metadata.fit)
        _write_request(writer, model._metadata.score)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = archive.schema_version
        archive.expect(Self.TypeId, version if 1 <= version <= 2 else Self.StateVersion)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        var config = _KernelConfig.read(reader)
        var C = reader.scalar[DType.float64]()
        var classifier = _ClassifierConfig[Self.Label].read(reader)
        var classes = _read_labels[Self.Label](reader)
        var rng = _read_rng(reader)
        var state = _read_fit(archive, reader, len(classes), 1)
        var requests = _read_supervised_requests(reader, version)
        reader.finish()
        var model = Self.Model(
            kernel=config.kernel, degree=config.degree, gamma=config.gamma,
            coef0=config.coef0, tol=config.tol, shrinking=config.shrinking,
            cache_size=config.cache_size, verbose=config.verbose, max_iter=config.max_iter,
            C=C,
            probability=classifier.probability, class_weight=classifier.class_weight,
            decision_function_shape=classifier.decision_function_shape,
            break_ties=classifier.break_ties, random_state=classifier.random_state,
        )
        model._metadata = requests^
        model._rng = rng^
        _ = _class_count(classes, Bool(state.engine))
        model.classes_ = classes^
        if state.engine:
            ref engine = state.engine.value()
            ref snapshot = state.snapshot.value()
            model.support_vectors_ = engine.support_vectors.copy()
            model.n_support_ = support_counts(engine)
            if len(model.classes_) == 2:
                model.dual_coef_ = engine.dual_coef * Float64(-1)
                model.intercept_ = engine.rho.copy()
            else:
                model.dual_coef_ = engine.dual_coef.copy()
                model.intercept_ = engine.rho * Float64(-1)
            model.class_weight_ = _read_class_weights(archive, len(model.classes_))
            model.probA_ = engine.prob_a.copy()
            model.probB_ = engine.prob_b.copy()
            model.n_iter_ = checked_int32(engine.iterations)
            model.n_features_in_ = snapshot.n_features
            model.shape_fit_ = snapshot.x.shape.copy()
            model.fit_status_ = engine.fit_status
            if state.diagnostics:
                model.fit_warnings = state.diagnostics.value().warnings(engine.fit_status)
            swap(model._fit_diagnostics, state.diagnostics)
            model._gamma = state.kernel.gamma
            swap(model.support_, state.support)
            swap(model._kernel, state.kernel)
            swap(model._snapshot, state.snapshot)
            swap(model._model, state.engine)
        return model^


struct NuSVCSchema[Label: Copyable & Comparable & Deinitable = Float64](ModelSchema):
    """Version 3 state for a dense NuSVC."""

    comptime Model = NuSVC[Self.Label]
    comptime TypeId = "moxie.svm.NuSVC"
    comptime StateVersion = 3

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        _KernelConfig(
            model.kernel.copy(), model.degree, model.gamma.copy(), model.coef0,
            model.tol, model.shrinking, model.cache_size, model.verbose, model.max_iter,
        ).write(writer)
        writer.scalar[DType.float64](model.nu)
        _ClassifierConfig(
            model.probability.copy(), model.class_weight.copy(),
            model.decision_function_shape.copy(), model.break_ties, model.random_state,
        ).write(writer)
        _write_labels(writer, model.classes_)
        var classes = _class_count(model.classes_, Bool(model._model))
        _write_rng(writer, model._rng)
        _write_fit(
            archive, writer, model._kernel, model._model, model._snapshot,
            model.support_, classes, model._fit_diagnostics,
        )
        if classes:
            _write_array(archive, "class_weight_", model.class_weight_, [classes])
            _validate_class_weights(model.class_weight_)
        _write_request(writer, model._metadata.fit)
        _write_request(writer, model._metadata.score)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = archive.schema_version
        archive.expect(Self.TypeId, version if 1 <= version <= 2 else Self.StateVersion)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        var config = _KernelConfig.read(reader)
        var nu = reader.scalar[DType.float64]()
        var classifier = _ClassifierConfig[Self.Label].read(reader)
        var classes = _read_labels[Self.Label](reader)
        var rng = _read_rng(reader)
        var state = _read_fit(archive, reader, len(classes), 1)
        var requests = _read_supervised_requests(reader, version)
        reader.finish()
        var model = Self.Model(
            kernel=config.kernel, degree=config.degree, gamma=config.gamma,
            coef0=config.coef0, tol=config.tol, shrinking=config.shrinking,
            cache_size=config.cache_size, verbose=config.verbose, max_iter=config.max_iter,
            nu=nu,
            probability=classifier.probability, class_weight=classifier.class_weight,
            decision_function_shape=classifier.decision_function_shape,
            break_ties=classifier.break_ties, random_state=classifier.random_state,
        )
        model._metadata = requests^
        model._rng = rng^
        _ = _class_count(classes, Bool(state.engine))
        model.classes_ = classes^
        if state.engine:
            ref engine = state.engine.value()
            ref snapshot = state.snapshot.value()
            model.support_vectors_ = engine.support_vectors.copy()
            model.n_support_ = support_counts(engine)
            if len(model.classes_) == 2:
                model.dual_coef_ = engine.dual_coef * Float64(-1)
                model.intercept_ = engine.rho.copy()
            else:
                model.dual_coef_ = engine.dual_coef.copy()
                model.intercept_ = engine.rho * Float64(-1)
            model.class_weight_ = _read_class_weights(archive, len(model.classes_))
            model.probA_ = engine.prob_a.copy()
            model.probB_ = engine.prob_b.copy()
            model.n_iter_ = checked_int32(engine.iterations)
            model.n_features_in_ = snapshot.n_features
            model.shape_fit_ = snapshot.x.shape.copy()
            model.fit_status_ = engine.fit_status
            if state.diagnostics:
                model.fit_warnings = state.diagnostics.value().warnings(engine.fit_status)
            swap(model._fit_diagnostics, state.diagnostics)
            model._gamma = state.kernel.gamma
            swap(model.support_, state.support)
            swap(model._kernel, state.kernel)
            swap(model._snapshot, state.snapshot)
            swap(model._model, state.engine)
        return model^


struct SVRSchema(ModelSchema):
    """Version 3 state for a dense SVR."""

    comptime Model = SVR
    comptime TypeId = "moxie.svm.SVR"
    comptime StateVersion = 3

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        _KernelConfig(
            model.kernel.copy(), model.degree, model.gamma.copy(), model.coef0,
            model.tol, model.shrinking, model.cache_size, model.verbose, model.max_iter,
        ).write(writer)
        writer.scalar[DType.float64](model.C)
        writer.scalar[DType.float64](model.epsilon)
        _write_rng(writer, model._rng)
        _write_fit(
            archive, writer, model._kernel, model._model, model._snapshot,
            model.support_, 0, model._fit_diagnostics,
        )
        _write_request(writer, model._metadata.fit)
        _write_request(writer, model._metadata.score)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = archive.schema_version
        archive.expect(Self.TypeId, version if 1 <= version <= 2 else Self.StateVersion)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        var config = _KernelConfig.read(reader)
        var C = reader.scalar[DType.float64]()
        var epsilon = reader.scalar[DType.float64]()
        var rng = _read_rng(reader)
        var state = _read_fit(archive, reader, 0, 0)
        var requests = _read_supervised_requests(reader, version)
        reader.finish()
        var model = Self.Model(
            kernel=config.kernel, degree=config.degree, gamma=config.gamma,
            coef0=config.coef0, tol=config.tol, shrinking=config.shrinking,
            cache_size=config.cache_size, verbose=config.verbose, max_iter=config.max_iter,
            C=C, epsilon=epsilon,
        )
        model._metadata = requests^
        model._rng = rng^
        if state.engine:
            ref engine = state.engine.value()
            ref snapshot = state.snapshot.value()
            model.support_vectors_ = engine.support_vectors.copy()
            model.n_support_ = support_counts(engine)
            model.dual_coef_ = engine.dual_coef.copy()
            model.intercept_ = engine.rho * Float64(-1)
            model.n_iter_ = Int(engine.iterations.unsafe_get(0))
            model.n_features_in_ = snapshot.n_features
            model.shape_fit_ = snapshot.x.shape.copy()
            model.fit_status_ = engine.fit_status
            if state.diagnostics:
                model.fit_warnings = state.diagnostics.value().warnings(engine.fit_status)
            swap(model._fit_diagnostics, state.diagnostics)
            model._gamma = state.kernel.gamma
            swap(model.support_, state.support)
            swap(model._kernel, state.kernel)
            swap(model._snapshot, state.snapshot)
            swap(model._model, state.engine)
        return model^


struct NuSVRSchema(ModelSchema):
    """Version 3 state for a dense NuSVR."""

    comptime Model = NuSVR
    comptime TypeId = "moxie.svm.NuSVR"
    comptime StateVersion = 3

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        _KernelConfig(
            model.kernel.copy(), model.degree, model.gamma.copy(), model.coef0,
            model.tol, model.shrinking, model.cache_size, model.verbose, model.max_iter,
        ).write(writer)
        writer.scalar[DType.float64](model.C)
        writer.scalar[DType.float64](model.nu)
        _write_rng(writer, model._rng)
        _write_fit(
            archive, writer, model._kernel, model._model, model._snapshot,
            model.support_, 0, model._fit_diagnostics,
        )
        _write_request(writer, model._metadata.fit)
        _write_request(writer, model._metadata.score)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = archive.schema_version
        archive.expect(Self.TypeId, version if 1 <= version <= 2 else Self.StateVersion)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        var config = _KernelConfig.read(reader)
        var C = reader.scalar[DType.float64]()
        var nu = reader.scalar[DType.float64]()
        var rng = _read_rng(reader)
        var state = _read_fit(archive, reader, 0, 0)
        var requests = _read_supervised_requests(reader, version)
        reader.finish()
        var model = Self.Model(
            kernel=config.kernel, degree=config.degree, gamma=config.gamma,
            coef0=config.coef0, tol=config.tol, shrinking=config.shrinking,
            cache_size=config.cache_size, verbose=config.verbose, max_iter=config.max_iter,
            C=C, nu=nu,
        )
        model._metadata = requests^
        model._rng = rng^
        if state.engine:
            ref engine = state.engine.value()
            ref snapshot = state.snapshot.value()
            model.support_vectors_ = engine.support_vectors.copy()
            model.n_support_ = support_counts(engine)
            model.dual_coef_ = engine.dual_coef.copy()
            model.intercept_ = engine.rho * Float64(-1)
            model.n_iter_ = Int(engine.iterations.unsafe_get(0))
            model.n_features_in_ = snapshot.n_features
            model.shape_fit_ = snapshot.x.shape.copy()
            model.fit_status_ = engine.fit_status
            if state.diagnostics:
                model.fit_warnings = state.diagnostics.value().warnings(engine.fit_status)
            swap(model._fit_diagnostics, state.diagnostics)
            model._gamma = state.kernel.gamma
            swap(model.support_, state.support)
            swap(model._kernel, state.kernel)
            swap(model._snapshot, state.snapshot)
            swap(model._model, state.engine)
        return model^


struct OneClassSVMSchema(ModelSchema):
    """Version 3 state for a dense OneClassSVM."""

    comptime Model = OneClassSVM
    comptime TypeId = "moxie.svm.OneClassSVM"
    comptime StateVersion = 3

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        _KernelConfig(
            model.kernel.copy(), model.degree, model.gamma.copy(), model.coef0,
            model.tol, model.shrinking, model.cache_size, model.verbose, model.max_iter,
        ).write(writer)
        writer.scalar[DType.float64](model.nu)
        _write_rng(writer, model._rng)
        _write_fit(
            archive, writer, model._kernel, model._model, model._snapshot,
            model.support_, 0, model._fit_diagnostics,
        )
        if model._model:
            _write_array(archive, "offset_", model.offset_, [1])
        _write_request(writer, model._fit_request)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var version = archive.schema_version
        archive.expect(Self.TypeId, version if 1 <= version <= 2 else Self.StateVersion)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        var config = _KernelConfig.read(reader)
        var nu = reader.scalar[DType.float64]()
        var rng = _read_rng(reader)
        var state = _read_fit(archive, reader, 0, 1)
        var request = SampleWeightRequest()
        if version >= 3:
            request = _read_request(reader)
        reader.finish()
        var model = Self.Model(
            kernel=config.kernel, degree=config.degree, gamma=config.gamma,
            coef0=config.coef0, tol=config.tol, shrinking=config.shrinking,
            cache_size=config.cache_size, verbose=config.verbose, max_iter=config.max_iter,
            nu=nu,
        )
        model._fit_request = request^
        model._rng = rng^
        if state.engine:
            ref engine = state.engine.value()
            ref snapshot = state.snapshot.value()
            model.support_vectors_ = engine.support_vectors.copy()
            model.n_support_ = support_counts(engine)
            model.dual_coef_ = engine.dual_coef.copy()
            model.intercept_ = engine.rho * Float64(-1)
            model.n_iter_ = Int(engine.iterations.unsafe_get(0))
            model.offset_ = _read_array[DType.float64](archive, "offset_", [1])
            model.n_features_in_ = snapshot.n_features
            model.shape_fit_ = snapshot.x.shape.copy()
            model.fit_status_ = engine.fit_status
            if state.diagnostics:
                model.fit_warnings = state.diagnostics.value().warnings(engine.fit_status)
            swap(model._fit_diagnostics, state.diagnostics)
            model._gamma = state.kernel.gamma
            swap(model.support_, state.support)
            swap(model._kernel, state.kernel)
            swap(model._snapshot, state.snapshot)
            swap(model._model, state.engine)
        return model^
