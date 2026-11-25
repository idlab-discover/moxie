# SPDX-License-Identifier: BSD-3-Clause
"""Experimental native persistence for calibration with built-in dense classifier adapters."""

from numojo import Item, Shape
from std.math import isfinite
from std.utils import Variant
from ._calibration_adapter import adapt_classifier
from ._calibration_algorithms import Array, _IsotonicCalibration, _SigmoidCalibration
from ._calibration_cv import (
    CalibratedClassifierCV,
    _CalibratedClassifier,
)
from ._calibration_estimator import (
    CalibrationEstimator,
    ClassifierCalibrationAdapter,
    _adapt_linear_svc,
    validate_integral_labels,
)
from ._calibration_splits import CalibrationCV, CalibrationSplit
from .discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from .discriminant_analysis_serialization import (
    LinearDiscriminantAnalysisSchema,
    QuadraticDiscriminantAnalysisSchema,
)
from .frozen import FrozenEstimator
from .isotonic_serialization import IsotonicRegressionSchema
from .model_selection import KFold, LeaveOneOut, StratifiedKFold
from .naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from .naive_bayes_serialization import (
    BernoulliNBSchema,
    CategoricalNBSchema,
    ComplementNBSchema,
    GaussianNBSchema,
    MultinomialNBSchema,
)
from .serialization import BinaryReader, BinaryWriter, ModelArchive, ModelSchema
from .svm import LinearSVC, SVC
from .svm.kernel_serialization import SVCSchema
from .svm.linear_serialization import LinearSVCSchema


def _write_optional_int(mut writer: BinaryWriter, value: Optional[Int]):
    writer.boolean(Bool(value))
    if value:
        writer.integer(value.or_else(0))


def _read_optional_int(mut reader: BinaryReader[_]) raises -> Optional[Int]:
    if reader.boolean():
        return reader.integer()
    return None


def _write_indices(mut writer: BinaryWriter, indices: List[Int]):
    writer.uint(UInt64(len(indices)))
    for index in indices:
        writer.integer(index)


def _read_indices(mut reader: BinaryReader[_]) raises -> List[Int]:
    var count = reader.count(max(0, reader.remaining() - 8) // 8)
    var indices = List[Int](capacity=count)
    for _ in range(count):
        indices.append(reader.integer())
    return indices^


def _write_cv(mut writer: BinaryWriter, cv: CalibrationCV):
    if cv.isa[type_of(None)]():
        writer.uint(0)
    elif cv.isa[Int]():
        writer.uint(1)
        writer.integer(cv[Int])
    elif cv.isa[List[CalibrationSplit]]():
        writer.uint(2)
        ref splits = cv[List[CalibrationSplit]]
        writer.uint(UInt64(len(splits)))
        for split in splits:
            _write_indices(writer, split.train)
            _write_indices(writer, split.test)
    elif cv.isa[KFold]():
        writer.uint(3)
        ref splitter = cv[KFold]
        writer.integer(splitter.n_splits)
        writer.boolean(splitter.shuffle)
        _write_optional_int(writer, splitter.random_state)
    elif cv.isa[StratifiedKFold]():
        writer.uint(4)
        ref splitter = cv[StratifiedKFold]
        writer.integer(splitter.n_splits)
        writer.boolean(splitter.shuffle)
        _write_optional_int(writer, splitter.random_state)
    else:
        writer.uint(5)


def _read_cv(mut reader: BinaryReader[_]) raises -> CalibrationCV:
    var kind = reader.count(5)
    if kind == 0:
        return None
    if kind == 1:
        return reader.integer()
    if kind == 2:
        var count = reader.count(max(0, reader.remaining() - 8) // 16)
        var splits = List[CalibrationSplit](capacity=count)
        for _ in range(count):
            splits.append(
                CalibrationSplit(_read_indices(reader), _read_indices(reader))
            )
        return splits^
    if kind == 5:
        return LeaveOneOut()
    var folds = reader.integer()
    var shuffle = reader.boolean()
    var seed = _read_optional_int(reader)
    if kind == 3:
        return KFold(folds, shuffle=shuffle, random_state=seed)
    return StratifiedKFold(folds, shuffle=shuffle, random_state=seed)


def _write_child(
    mut writer: BinaryWriter,
    mut archive: ModelArchive,
    prefix: String,
    var child: ModelArchive,
) raises:
    child.validate()
    writer.string(child.type_id)
    writer.integer(child.schema_version)
    writer.bytes(Span(child.manifest))
    writer.uint(UInt64(len(child.payloads)))
    while len(child.payloads) > 0:
        var payload = child.payloads.pop(0)
        payload.name = prefix + payload.name
        archive.payloads.append(payload^)


def _read_child(
    mut reader: BinaryReader[_],
    archive: ModelArchive,
    prefix: String,
    mut array_index: Int,
) raises -> ModelArchive:
    var child = ModelArchive(reader.string(4096), reader.integer())
    child.manifest.extend(reader.bytes(reader.remaining()))
    var count = reader.count(len(archive.payloads) - array_index)
    for _ in range(count):
        var payload = archive.payloads[array_index].copy()
        if not payload.name.startswith(prefix):
            raise Error("ModelSchemaError: unexpected calibration child array")
        var local_name = String(payload.name.removeprefix(prefix))
        payload.name = local_name^
        child.payloads.append(payload^)
        array_index += 1
    return child^


def _write_classes(mut archive: ModelArchive, name: String, classes: Array) raises:
    var values = List[Float64](capacity=classes.size)
    for i in range(classes.size):
        values.append(classes.item(i))
    var shape: List[Int] = [len(values)]
    archive.add_array[DType.float64](name, shape, values)


def _read_classes(
    archive: ModelArchive, name: String, mut array_index: Int
) raises -> Array:
    if array_index == len(archive.payloads):
        raise Error("ModelSchemaError: missing calibration class array")
    ref payload = archive.payloads[array_index]
    if payload.name != name or len(payload.shape) != 1:
        raise Error("ModelSchemaError: invalid calibration class array")
    var values = payload.values[DType.float64]()
    var classes = Array(Shape(len(values)))
    for i in range(len(values)):
        classes[Item(i)] = values[i]
    array_index += 1
    return classes^


def _validate_classes(classes: Array) raises:
    if classes.ndim != 1:
        raise Error("ModelSchemaError: calibration classes must be a vector")
    for i in range(classes.size):
        if not isfinite(classes.item(i)):
            raise Error("ModelSchemaError: calibration classes must be finite")
        if i > 0 and classes.item(i) <= classes.item(i - 1):
            raise Error(
                "ModelSchemaError: calibration classes must be sorted and unique"
            )
    validate_integral_labels(classes)


trait _AdapterCodec:
    comptime Adapter: CalibrationEstimator
    comptime TypeId: StaticString

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive: ...

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter: ...

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int: ...


struct _LinearSVCCodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[LinearSVC[]]
    comptime TypeId: StaticString = "moxie.calibration.CalibratedClassifierCV.LinearSVC"

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in LinearSVC adapter"
            )
        return LinearSVCSchema[].encode(adapter.estimator())

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        return _adapt_linear_svc(LinearSVCSchema[].decode(archive))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().n_features_in_


struct _SVCCodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[SVC[]]
    comptime TypeId: StaticString = "moxie.calibration.CalibratedClassifierCV.SVC"

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in SVC adapter"
            )
        return SVCSchema[].encode(adapter.estimator())

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        return adapt_classifier(SVCSchema[].decode(archive))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().n_features_in_


struct _GaussianNBCodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[GaussianNB[]]
    comptime TypeId: StaticString = (
        "moxie.calibration.CalibratedClassifierCV.GaussianNB"
    )

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in GaussianNB adapter"
            )
        return GaussianNBSchema[].encode(adapter.estimator())

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        return adapt_classifier(GaussianNBSchema[].decode(archive))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().n_features_in_


struct _FrozenGaussianNBCodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[FrozenEstimator[GaussianNB[]]]
    comptime TypeId: StaticString = "moxie.calibration.CalibratedClassifierCV.FrozenGaussianNB"

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in frozen GaussianNB adapter"
            )
        return GaussianNBSchema[].encode(adapter.estimator().estimator())

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        return adapt_classifier(FrozenEstimator(GaussianNBSchema[].decode(archive)))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().estimator().n_features_in_


struct _MultinomialNBCodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[MultinomialNB[]]
    comptime TypeId: StaticString = (
        "moxie.calibration.CalibratedClassifierCV.MultinomialNB"
    )

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in MultinomialNB adapter"
            )
        return MultinomialNBSchema[].encode(adapter.estimator())

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        return adapt_classifier(MultinomialNBSchema[].decode(archive))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().n_features_in_


struct _ComplementNBCodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[ComplementNB[]]
    comptime TypeId: StaticString = (
        "moxie.calibration.CalibratedClassifierCV.ComplementNB"
    )

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in ComplementNB adapter"
            )
        return ComplementNBSchema[].encode(adapter.estimator())

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        return adapt_classifier(ComplementNBSchema[].decode(archive))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().n_features_in_


struct _BernoulliNBCodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[BernoulliNB[]]
    comptime TypeId: StaticString = (
        "moxie.calibration.CalibratedClassifierCV.BernoulliNB"
    )

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in BernoulliNB adapter"
            )
        return BernoulliNBSchema[].encode(adapter.estimator())

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        return adapt_classifier(BernoulliNBSchema[].decode(archive))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().n_features_in_


struct _CategoricalNBCodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[CategoricalNB[]]
    comptime TypeId: StaticString = (
        "moxie.calibration.CalibratedClassifierCV.CategoricalNB"
    )

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in CategoricalNB adapter"
            )
        return CategoricalNBSchema[].encode(adapter.estimator())

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        return adapt_classifier(CategoricalNBSchema[].decode(archive))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().n_features_in_


struct _LDACodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[LinearDiscriminantAnalysis[]]
    comptime TypeId: StaticString = (
        "moxie.calibration.CalibratedClassifierCV.LinearDiscriminantAnalysis"
    )

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in LDA adapter"
            )
        if adapter.estimator().covariance_estimator:
            raise Error("ModelSchemaError: calibrated LDA requires covariance_estimator=None")
        return LinearDiscriminantAnalysisSchema[].encode(adapter.estimator())

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        return adapt_classifier(LinearDiscriminantAnalysisSchema[].decode(archive))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().n_features_in_


struct _FrozenLDACodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[FrozenEstimator[LinearDiscriminantAnalysis[]]]
    comptime TypeId: StaticString = (
        "moxie.calibration.CalibratedClassifierCV.FrozenLinearDiscriminantAnalysis"
    )

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in frozen LDA adapter"
            )
        ref estimator = adapter.estimator().estimator()
        if estimator.covariance_estimator:
            raise Error("ModelSchemaError: calibrated LDA requires covariance_estimator=None")
        return LinearDiscriminantAnalysisSchema[].encode(estimator)

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        var estimator = LinearDiscriminantAnalysisSchema[].decode(archive)
        return adapt_classifier(FrozenEstimator(estimator^))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().estimator().n_features_in_


struct _QDACodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[QuadraticDiscriminantAnalysis[]]
    comptime TypeId: StaticString = (
        "moxie.calibration.CalibratedClassifierCV.QuadraticDiscriminantAnalysis"
    )

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in QDA adapter"
            )
        if adapter.estimator().covariance_estimator:
            raise Error("ModelSchemaError: calibrated QDA requires covariance_estimator=None")
        return QuadraticDiscriminantAnalysisSchema[].encode(adapter.estimator())

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        return adapt_classifier(QuadraticDiscriminantAnalysisSchema[].decode(archive))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().n_features_in_


struct _FrozenQDACodec(_AdapterCodec):
    comptime Adapter = ClassifierCalibrationAdapter[FrozenEstimator[QuadraticDiscriminantAnalysis[]]]
    comptime TypeId: StaticString = (
        "moxie.calibration.CalibratedClassifierCV.FrozenQuadraticDiscriminantAnalysis"
    )

    @staticmethod
    def encode(adapter: Self.Adapter) raises -> ModelArchive:
        if not adapter._builtin_reader:
            raise Error(
                "ModelSchemaError: calibration requires the built-in frozen QDA adapter"
            )
        ref estimator = adapter.estimator().estimator()
        if estimator.covariance_estimator:
            raise Error("ModelSchemaError: calibrated QDA requires covariance_estimator=None")
        return QuadraticDiscriminantAnalysisSchema[].encode(estimator)

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Adapter:
        var estimator = QuadraticDiscriminantAnalysisSchema[].decode(archive)
        return adapt_classifier(FrozenEstimator(estimator^))

    @staticmethod
    def feature_count(adapter: Self.Adapter) -> Int:
        return adapter.estimator().estimator().n_features_in_


def _check_pair[Codec: _AdapterCodec](
    pair: _CalibratedClassifier[Codec.Adapter], classes: Array, features: Int
) raises:
    if Codec.feature_count(pair.estimator) != features:
        raise Error("ModelSchemaError: calibration child feature count mismatch")
    var child_classes = pair.estimator.class_labels()
    _validate_classes(child_classes)
    var positions = List[Int](capacity=child_classes.size)
    var index = 0
    for i in range(child_classes.size):
        var label = child_classes.item(i)
        while index < classes.size and classes.item(index) < label:
            index += 1
        if index == classes.size or classes.item(index) != label:
            raise Error("ModelSchemaError: unknown calibration child class")
        positions.append(index)
    if pair.classes.ndim != 1 or pair.classes.size != classes.size:
        raise Error("ModelSchemaError: calibration child classes mismatch")
    for i in range(classes.size):
        if pair.classes.item(i) != classes.item(i):
            raise Error("ModelSchemaError: calibration child classes mismatch")
    if pair.method == "temperature":
        if (
            child_classes.size != classes.size
            or pair.temperature.n_classes_ != classes.size
            or not isfinite(pair.temperature.beta_)
            or pair.temperature.beta_ <= 0
            or len(pair.class_indices) != 0
            or len(pair.sigmoid) != 0
            or len(pair.isotonic) != 0
        ):
            raise Error("ModelSchemaError: invalid temperature calibration state")
        return

    var count = 1 if child_classes.size == 2 else child_classes.size
    if len(pair.class_indices) != count:
        raise Error("ModelSchemaError: incorrect number of calibration columns")
    for i in range(count):
        var expected = 1 if classes.size == 2 else positions[i]
        if pair.class_indices[i] != expected:
            raise Error("ModelSchemaError: calibration column order mismatch")
    if pair.method == "sigmoid":
        if len(pair.sigmoid) != count or len(pair.isotonic) != 0:
            raise Error("ModelSchemaError: incorrect number of sigmoid calibrators")
        for sigmoid in pair.sigmoid:
            if (
                not sigmoid.fitted
                or not isfinite(sigmoid.a_)
                or not isfinite(sigmoid.b_)
            ):
                raise Error("ModelSchemaError: invalid sigmoid coefficients")
    elif pair.method == "isotonic":
        if len(pair.isotonic) != count or len(pair.sigmoid) != 0:
            raise Error("ModelSchemaError: incorrect number of isotonic calibrators")
        for isotonic in pair.isotonic:
            ref model = isotonic._model
            if (
                not model.__sklearn_is_fitted__()
                or not model.increasing_
                or model.out_of_bounds != "clip"
                or model._bounds_error
                or isotonic.X_thresholds_.ndim != 1
                or isotonic.y_thresholds_.ndim != 1
                or len(model.X_thresholds_) != isotonic.X_thresholds_.size
                or len(model.y_thresholds_) != isotonic.y_thresholds_.size
            ):
                raise Error("ModelSchemaError: invalid isotonic calibration state")
            for i in range(len(model.X_thresholds_)):
                var probability = model.y_thresholds_[i]
                if (
                    probability < 0
                    or probability > 1
                    or model.X_thresholds_[i] != isotonic.X_thresholds_.item(i)
                    or probability != isotonic.y_thresholds_.item(i)
                ):
                    raise Error(
                        "ModelSchemaError: invalid isotonic calibration thresholds"
                    )
    else:
        raise Error("ModelSchemaError: unknown fitted calibration method")


def _write_pair[Codec: _AdapterCodec](
    mut writer: BinaryWriter,
    mut archive: ModelArchive,
    pair: _CalibratedClassifier[Codec.Adapter],
    prefix: String,
) raises:
    writer.string(pair.method)
    _write_classes(archive, prefix + "classes_", pair.classes)
    _write_child(
        writer, archive, prefix + "estimator/", Codec.encode(pair.estimator)
    )
    if pair.method == "temperature":
        writer.scalar[DType.float64](pair.temperature.beta_)
        writer.integer(pair.temperature.n_classes_)
        return
    _write_indices(writer, pair.class_indices)
    if pair.method == "sigmoid":
        for sigmoid in pair.sigmoid:
            writer.scalar[DType.float64](sigmoid.a_)
            writer.scalar[DType.float64](sigmoid.b_)
    else:
        for i in range(len(pair.isotonic)):
            _write_child(
                writer,
                archive,
                prefix + "isotonic/" + String(i) + "/",
                IsotonicRegressionSchema[].encode(pair.isotonic[i]._model),
            )


def _read_pair[Codec: _AdapterCodec](
    mut reader: BinaryReader[_],
    archive: ModelArchive,
    prefix: String,
    mut array_index: Int,
) raises -> _CalibratedClassifier[Codec.Adapter]:
    var method = reader.string(reader.remaining())
    if method != "sigmoid" and method != "isotonic" and method != "temperature":
        raise Error("ModelSchemaError: unknown fitted calibration method")
    var classes = _read_classes(archive, prefix + "classes_", array_index)
    var estimator = Codec.decode(
        _read_child(reader, archive, prefix + "estimator/", array_index)
    )
    var pair = _CalibratedClassifier(estimator^, classes^, method.copy())
    if method == "temperature":
        pair.temperature.beta_ = reader.scalar[DType.float64]()
        pair.temperature.n_classes_ = reader.integer()
        return pair^
    pair.class_indices = _read_indices(reader)
    if method == "sigmoid":
        for _ in range(len(pair.class_indices)):
            var sigmoid = _SigmoidCalibration()
            sigmoid.a_ = reader.scalar[DType.float64]()
            sigmoid.b_ = reader.scalar[DType.float64]()
            sigmoid.fitted = True
            pair.sigmoid.append(sigmoid^)
    elif method == "isotonic":
        for i in range(len(pair.class_indices)):
            var isotonic = _IsotonicCalibration()
            isotonic._model = IsotonicRegressionSchema[].decode(
                _read_child(
                    reader, archive, prefix + "isotonic/" + String(i) + "/", array_index
                )
            )
            ref model = isotonic._model
            isotonic.X_thresholds_ = Array(Shape(len(model.X_thresholds_)))
            isotonic.y_thresholds_ = Array(Shape(len(model.y_thresholds_)))
            for j in range(len(model.X_thresholds_)):
                isotonic.X_thresholds_[Item(j)] = model.X_thresholds_[j]
                isotonic.y_thresholds_[Item(j)] = model.y_thresholds_[j]
            pair.isotonic.append(isotonic^)
    return pair^


struct _CalibrationSchema[Codec: _AdapterCodec](ModelSchema):
    comptime Model = CalibratedClassifierCV[Self.Codec.Adapter]
    comptime TypeId = Self.Codec.TypeId
    comptime StateVersion = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        _validate_classes(model.classes_)
        var count = len(model.calibrated_classifiers_)
        Self._check_state(model.n_features_in_, model.classes_.size, count)

        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var writer = BinaryWriter()
        writer.string(model.method)
        _write_cv(writer, model.cv)
        _write_optional_int(writer, model.n_jobs)
        writer.boolean(model.ensemble.isa[Bool]())
        if model.ensemble.isa[Bool]():
            writer.boolean(model.ensemble[Bool])
        else:
            writer.string(model.ensemble[String])
        writer.integer(model.n_features_in_)
        writer.uint(UInt64(count))

        _write_classes(archive, "classes_", model.classes_)
        _write_child(writer, archive, "estimator/", Self.Codec.encode(model.estimator))
        for i in range(count):
            ref pair = model.calibrated_classifiers_[i]
            _check_pair[Self.Codec](pair, model.classes_, model.n_features_in_)
            _write_pair[Self.Codec](writer, archive, pair, "fold/" + String(i) + "/")
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TypeId, Self.StateVersion)
        archive.validate()
        var array_index = 0
        var classes = _read_classes(archive, "classes_", array_index)
        _validate_classes(classes)
        var reader = BinaryReader(Span(archive.manifest))
        var method = reader.string(reader.remaining())
        var cv = _read_cv(reader)
        var n_jobs = _read_optional_int(reader)
        var ensemble: Variant[Bool, String]
        if reader.boolean():
            ensemble = reader.boolean()
        else:
            ensemble = reader.string(reader.remaining())
        var features = reader.integer()
        var count = reader.count((len(archive.payloads) - array_index) // 3)
        Self._check_state(features, classes.size, count)

        var child = _read_child(reader, archive, "estimator/", array_index)
        var estimator = Self.Codec.decode(child)
        var model = Self.Model(
            estimator^, method=method^, cv=cv^, n_jobs=n_jobs, ensemble=ensemble^
        )
        for i in range(count):
            var pair = _read_pair[Self.Codec](
                reader, archive, "fold/" + String(i) + "/", array_index
            )
            _check_pair[Self.Codec](pair, classes, features)
            model.calibrated_classifiers_.append(pair^)
        reader.finish()
        if array_index != len(archive.payloads):
            raise Error("ModelSchemaError: unexpected calibration arrays")
        model.classes_ = classes^
        model.n_features_in_ = features
        return model^

    @staticmethod
    def _check_state(features: Int, classes: Int, count: Int) raises:
        if (count == 0 and (features != 0 or classes != 0)) or (
            count > 0 and (features < 1 or classes < 2)
        ):
            raise Error("ModelSchemaError: invalid calibration fitted state")


# Each concrete schema has a distinct archive identifier.
comptime CalibratedClassifierCVSchema = _CalibrationSchema[_LinearSVCCodec]
comptime CalibratedSVCClassifierCVSchema = _CalibrationSchema[_SVCCodec]
comptime CalibratedGaussianNBClassifierCVSchema = _CalibrationSchema[_GaussianNBCodec]
comptime CalibratedFrozenGaussianNBClassifierCVSchema = _CalibrationSchema[_FrozenGaussianNBCodec]
comptime CalibratedMultinomialNBClassifierCVSchema = _CalibrationSchema[_MultinomialNBCodec]
comptime CalibratedComplementNBClassifierCVSchema = _CalibrationSchema[_ComplementNBCodec]
comptime CalibratedBernoulliNBClassifierCVSchema = _CalibrationSchema[_BernoulliNBCodec]
comptime CalibratedCategoricalNBClassifierCVSchema = _CalibrationSchema[_CategoricalNBCodec]
comptime CalibratedLDAClassifierCVSchema = _CalibrationSchema[_LDACodec]
comptime CalibratedQDAClassifierCVSchema = _CalibrationSchema[_QDACodec]
comptime CalibratedFrozenLDAClassifierCVSchema = _CalibrationSchema[_FrozenLDACodec]
comptime CalibratedFrozenQDAClassifierCVSchema = _CalibrationSchema[_FrozenQDACodec]
