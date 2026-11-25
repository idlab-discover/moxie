# SPDX-License-Identifier: BSD-3-Clause
"""Fitted and unfitted native files for built-in calibration adapters."""
import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.utils import Variant
from moxie.base import clone
from moxie.calibration import (
    CalibratedClassifierCV,
    CalibrationCV,
    CalibrationEstimator,
    CalibrationSplit,
    ClassifierCalibrationAdapter,
    adapt_classifier,
)
from moxie.calibration_serialization import (
    CalibratedClassifierCVSchema,
    CalibratedGaussianNBClassifierCVSchema,
    CalibratedLDAClassifierCVSchema,
    CalibratedQDAClassifierCVSchema,
    CalibratedSVCClassifierCVSchema,
)
from moxie.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    NoCovarianceEstimator,
    QuadraticDiscriminantAnalysis,
)
from moxie.model_selection import KFold, LeaveOneOut, StratifiedKFold
from moxie.naive_bayes import GaussianNB
from moxie.serialization import (
    ModelArchive, ModelSchema, dumps, loads, load_model, save_model,
)
from moxie.svm import LinearSVC, SVC

comptime Schema = CalibratedClassifierCVSchema
comptime Dense = nm.NDArray[DType.float64]


def assert_bad(archive: ModelArchive, expected: String) raises:
    with assert_raises(contains=expected):
        _ = Schema.decode(loads(dumps(archive)))


def assert_equal_predictions[E: CalibrationEstimator](
    original: CalibratedClassifierCV[E], restored: CalibratedClassifierCV[E],
    X: Dense, y: Dense,
) raises:
    var expected = original.predict_proba(X)
    var actual = restored.predict_proba(X)
    assert_equal(actual.shape[0], expected.shape[0])
    assert_equal(actual.shape[1], expected.shape[1])
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            assert_equal(actual.item(i, j), expected.item(i, j))
    var actual_labels = restored.predict(X)
    var expected_labels = original.predict(X)
    for i in range(actual_labels.size):
        assert_equal(actual_labels.item(i), expected_labels.item(i))
    assert_equal(restored.score(X, y), original.score(X, y))


def test_svc(X: Dense, y: Dense) raises:
    comptime SVCSchema = CalibratedSVCClassifierCVSchema
    var file = "builds/calibration-svc-model.moxie"
    var model = CalibratedClassifierCV(
        adapt_classifier(SVC(C=0.5, random_state=3)),
        cv=3, method="temperature", ensemble=False,
    )
    save_model[SVCSchema](model, file)
    var unfitted = load_model[SVCSchema](file)
    assert_false(unfitted.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.predict(X)
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = load_model[Schema](file)

    _ = model.fit(X, y)
    _ = model.set_params(estimator=adapt_classifier(SVC(C=2.0, random_state=7)))
    save_model[SVCSchema](model, file)
    var restored = load_model[SVCSchema](file)
    assert_equal(restored.estimator.estimator().C, 2.0)
    assert_equal(restored.calibrated_classifiers_[0].estimator.estimator().C, 0.5)
    assert_equal_predictions(model, restored, X, y)
    assert_false(clone(restored).__sklearn_is_fitted__())

    var bad = SVCSchema.encode(restored)
    bad.payloads[len(bad.payloads) - 1].name = "fold/0/estimator/unknown"
    with assert_raises(contains="ModelSchemaError"):
        _ = SVCSchema.decode(loads(dumps(bad)))

    _ = restored.fit(X, y)
    _ = model.fit(X, y)
    assert_equal(restored.calibrated_classifiers_[0].estimator.estimator().C, 2.0)
    assert_equal_predictions(model, restored, X, y)

    var ovo = CalibratedClassifierCV(
        adapt_classifier(SVC(decision_function_shape="ovo")), cv=3,
    )
    with assert_raises(contains="multiclass SVC calibration requires ovr scores"):
        _ = ovo.fit(X, y)


def check_unfitted[E: CalibrationEstimator, //, Schema: ModelSchema](
    model: CalibratedClassifierCV[E], X: Dense,
) raises where Schema.Model == CalibratedClassifierCV[E]:
    var archive = Schema.encode(rebind[Schema.Model](model))
    var restored = rebind_var[CalibratedClassifierCV[E]](
        Schema.decode(loads(dumps(archive)))
    )
    assert_false(restored.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = restored.predict(X)
    assert_equal(dumps(Schema.encode(rebind[Schema.Model](restored))), dumps(archive))


def test_discriminant_configuration(X: Dense) raises:
    var lda = CalibratedClassifierCV(
        adapt_classifier(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        method="temperature", cv=3,
    )
    check_unfitted[CalibratedLDAClassifierCVSchema](lda, X)
    var qda = CalibratedClassifierCV(
        adapt_classifier(QuadraticDiscriminantAnalysis(reg_param=0.25, store_covariance=True)),
        method="isotonic", cv=3,
    )
    check_unfitted[CalibratedQDAClassifierCVSchema](qda, X)
    var archive = CalibratedLDAClassifierCVSchema.encode(lda)
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = CalibratedQDAClassifierCVSchema.decode(loads(dumps(archive)))

    var unsupported_lda = CalibratedClassifierCV(
        adapt_classifier(
            LinearDiscriminantAnalysis(covariance_estimator=NoCovarianceEstimator())
        ),
    )
    with assert_raises(contains="requires covariance_estimator=None"):
        _ = CalibratedLDAClassifierCVSchema.encode(unsupported_lda)

    var unsupported_qda = CalibratedClassifierCV(
        adapt_classifier(
            QuadraticDiscriminantAnalysis(covariance_estimator=NoCovarianceEstimator())
        ),
    )
    with assert_raises(contains="requires covariance_estimator=None"):
        _ = CalibratedQDAClassifierCVSchema.encode(unsupported_qda)


def test_gaussian(X: Dense, y: Dense) raises:
    comptime GSchema = CalibratedGaussianNBClassifierCVSchema
    var file = "builds/calibration-gaussian-model.moxie"
    var priors: List[Float64] = [0.2, 0.3, 0.5]
    var model = CalibratedClassifierCV(
        adapt_classifier(GaussianNB(priors=priors^, var_smoothing=0.1)),
        method="sigmoid", cv=3, ensemble=True,
    )
    check_unfitted[GSchema](model, X)
    _ = model.fit(X, y)
    _ = model.set_params(
        estimator=adapt_classifier(GaussianNB(var_smoothing=0.01)),
        method="temperature",
    )
    save_model[GSchema](model, file)
    var restored = load_model[GSchema](file)
    assert_equal(restored.estimator.estimator().var_smoothing, 0.01)
    assert_equal(restored.calibrated_classifiers_[0].estimator.estimator().var_smoothing, 0.1)
    assert_equal(restored.calibrated_classifiers_[0].method, "sigmoid")
    assert_equal_predictions(model, restored, X, y)
    assert_false(clone(restored).__sklearn_is_fitted__())

    var malformed = GSchema.encode(restored)
    for ref payload in malformed.payloads:
        if payload.name == "fold/0/estimator/theta_":
            payload.name = "fold/0/estimator/unknown"
    with assert_raises(contains="theta_"):
        _ = GSchema.decode(loads(dumps(malformed)))

    _ = restored.fit(X, y)
    _ = model.fit(X, y)
    assert_equal(restored.calibrated_classifiers_[0].estimator.estimator().var_smoothing, 0.01)
    assert_equal(restored.calibrated_classifiers_[0].method, "temperature")
    assert_equal_predictions(model, restored, X, y)

    def read_classes(model: GaussianNB[]) raises -> Dense:
        return Dense(nm.Shape(0))

    var custom = CalibratedClassifierCV(
        ClassifierCalibrationAdapter(GaussianNB(), read_classes, "GaussianNB")
    )
    with assert_raises(contains="requires the built-in GaussianNB adapter"):
        _ = GSchema.encode(custom)


def test_cv_configuration() raises:
    var splits: List[CalibrationSplit] = [
        CalibrationSplit([1, 3, 4], [0, 2]),
        CalibrationSplit([0, 2], [1, 3, 4]),
    ]
    var variants: List[CalibrationCV] = [
        None, 7, splits^, KFold(2, shuffle=True, random_state=63),
        StratifiedKFold(3, shuffle=True, random_state=0), LeaveOneOut(),
    ]
    for i in range(len(variants)):
        var model = CalibratedClassifierCV(cv=variants[i].copy())
        var archive = Schema.encode(model)
        var restored = Schema.decode(loads(dumps(archive)))
        # Verify each configuration tag, splitter setting and index survives.
        var rewritten = Schema.encode(restored)
        assert_equal(archive.manifest, rewritten.manifest)
        assert_false(restored.__sklearn_is_fitted__())
        assert_equal(restored.classes_.size, 0)
    def read_classes(model: LinearSVC[]) raises -> Dense:
        return Dense(nm.Shape(0))

    var custom = CalibratedClassifierCV(
        ClassifierCalibrationAdapter(LinearSVC(), read_classes, "LinearSVC")
    )
    with assert_raises(contains="requires the built-in LinearSVC adapter"):
        _ = Schema.encode(custom)


def main() raises:
    test_cv_configuration()
    var X = Dense(nm.Shape(42, 2))
    var y = Dense(nm.Shape(42))
    for i in range(42):
        var label = Float64(i % 3)
        X[nm.Item(i, 0)] = label + Float64(i % 5) / 7.0
        X[nm.Item(i, 1)] = Float64(i % 11) / 13.0
        y[nm.Item(i)] = label
    test_svc(X, y)
    test_discriminant_configuration(X)
    test_gaussian(X, y)
    var file = "builds/calibration-test-model.moxie"
    var model = CalibratedClassifierCV(
        adapt_classifier(LinearSVC(C=0.5, random_state=0)),
        method="isotonic", cv=StratifiedKFold(3, shuffle=True, random_state=42),
        ensemble=True,
    )
    save_model[Schema](model, file)
    var unfitted = load_model[Schema](file)
    assert_false(unfitted.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.predict(X)

    _ = model.fit(X, y)
    _ = model.set_params(
        estimator=adapt_classifier(LinearSVC(C=3.0, random_state=17)),
        method="temperature", cv=CalibrationCV(Int(2)), n_jobs=1,
        ensemble=Variant[Bool, String](False),
    )
    save_model[Schema](model, file)
    var restored = load_model[Schema](file)
    assert_true(restored.__sklearn_is_fitted__())
    assert_equal(restored.n_features_in_, 2)
    assert_equal(restored.method, "temperature")
    assert_equal(restored.cv[Int], 2)
    assert_equal(restored.n_jobs.or_else(0), 1)
    assert_false(restored.ensemble[Bool])
    assert_equal(restored.estimator.estimator().C, 3.0)
    assert_equal(restored.estimator.estimator().random_state, 17)
    assert_equal(len(restored.calibrated_classifiers_), 3)
    assert_equal(restored.calibrated_classifiers_[0].method, "isotonic")
    assert_equal(restored.calibrated_classifiers_[0].estimator.estimator().C, 0.5)
    assert_equal_predictions(model, restored, X, y)

    var archive = Schema.encode(restored)
    var bad = archive.copy()
    bad.schema_version += 1
    assert_bad(bad, "unsupported estimator type or state schema")
    bad = archive.copy()
    bad.payloads[0].name = "unknown"
    assert_bad(bad, "invalid calibration class array")
    bad = archive.copy()
    bad.payloads[1].name = "fold/0/wrong_classes"
    assert_bad(bad, "invalid calibration class array")
    bad = archive.copy()
    _ = bad.manifest.pop()
    assert_bad(bad, "ModelFormatError")
    bad = archive.copy()
    bad.manifest.append(0)
    assert_bad(bad, "trailing data")
    bad = archive.copy()
    bad.add_array[DType.float64]("extra", List[Int]([0]), List[Float64]())
    assert_bad(bad, "unexpected calibration arrays")

    var inconsistent = restored.copy()
    inconsistent.n_features_in_ += 1
    with assert_raises(contains="child feature count mismatch"):
        _ = Schema.encode(inconsistent)
    inconsistent = restored.copy()
    inconsistent.calibrated_classifiers_[0].class_indices[0] = 2
    with assert_raises(contains="column order mismatch"):
        _ = Schema.encode(inconsistent)

    var cloned = clone(restored)
    assert_false(cloned.__sklearn_is_fitted__())
    _ = restored.fit(X, y)
    _ = model.fit(X, y)
    assert_equal(restored.calibrated_classifiers_[0].method, "temperature")
    assert_equal(len(restored.calibrated_classifiers_), 1)
    assert_equal_predictions(model, restored, X, y)
    assert_equal(load_model[Schema](file).calibrated_classifiers_[0].method, "isotonic")
    print("PASS: built-in calibration adapter model files and malformed state")
