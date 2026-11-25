# SPDX-License-Identifier: BSD-3-Clause
"""Fitted and unfitted native files for LinearSVC calibration."""
import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.utils import Variant
from moxie.base import clone
from moxie.calibration import (
    CalibratedClassifierCV,
    CalibrationCV,
    CalibrationSplit,
    ClassifierCalibrationAdapter,
    adapt_classifier,
)
from moxie.calibration_serialization import CalibratedClassifierCVSchema
from moxie.model_selection import KFold, LeaveOneOut, StratifiedKFold
from moxie.serialization import (
    ModelArchive, dumps, loads, load_model, save_model,
)
from moxie.svm import LinearSVC

comptime Schema = CalibratedClassifierCVSchema
comptime Dense = nm.NDArray[DType.float64]


def assert_bad(archive: ModelArchive, expected: String) raises:
    with assert_raises(contains=expected):
        _ = Schema.decode(loads(dumps(archive)))


def assert_equal_predictions(
    original: Schema.Model, restored: Schema.Model, X: Dense, y: Dense
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
    print("PASS: calibrated LinearSVC native model files and malformed state")
