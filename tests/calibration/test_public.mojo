"""Isolated consumer using production classifiers through calibration."""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.sys import argv
from moxie.base import clone, SupervisedEstimator
from moxie.calibration import (
    adapt_classifier, CalibratedClassifierCV, CalibrationSplit,
    FrozenCalibrationEstimator, calibration_curve, CalibrationDisplay,
)
from moxie.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from moxie.frozen import FrozenEstimator
from moxie.svm import LinearSVC

comptime Array = nm.NDArray[DType.float64]


def generic_fit[E: SupervisedEstimator](mut model: E, X: E.Input, y: E.Target) raises:
    _ = model.fit(X, y)


def test_default_classifier(X: Array, y: Array) raises:
    var model = CalibratedClassifierCV()
    assert_equal(model.get_params().estimator.get_params().random_state, 0)
    assert_false(model.estimator.uses_probabilities())
    with assert_raises():
        _ = model.predict_proba(X)
    generic_fit(model, X, y)
    assert_equal(len(model.calibrated_classifiers_), 5)
    assert_false(model.estimator.__sklearn_is_fitted__())
    assert_true(model.calibrated_classifiers_[0].estimator.estimator().__sklearn_is_fitted__())
    var first = model.predict_proba(X)
    var explicit_none = CalibratedClassifierCV(estimator=None)
    var explicit = CalibratedClassifierCV(adapt_classifier(LinearSVC(random_state=0)))
    var cloned = clone(model)
    assert_false(cloned.__sklearn_is_fitted__())
    assert_equal(cloned.get_params().estimator.get_params().random_state, 0)
    _ = model.fit(X, y)
    _ = explicit_none.fit(X, y)
    _ = explicit.fit(X, y)
    _ = cloned.fit(X, y)
    var repeated = model.predict_proba(X)
    var none_probability = explicit_none.predict_proba(X)
    var explicit_probability = explicit.predict_proba(X)
    var cloned_probability = cloned.predict_proba(X)
    for i in range(X.shape[0]):
        for j in range(2):
            assert_equal(repeated.item(i, j), first.item(i, j))
            assert_equal(none_probability.item(i, j), first.item(i, j))
            assert_equal(explicit_probability.item(i, j), first.item(i, j))
            assert_equal(cloned_probability.item(i, j), first.item(i, j))


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(30, 2))
    var y = nm.zeros[DType.float64](nm.Shape(30))
    for i in range(30):
        X[nm.Item(i, 0)] = Float64(i % 2) + Float64(i % 7) / 4
        X[nm.Item(i, 1)] = Float64(i % 11) / 3
        y[nm.Item(i)] = -3.0 if i % 2 == 0 else 7.0
    test_default_classifier(X, y)
    var base = adapt_classifier(LinearDiscriminantAnalysis())
    with assert_raises():
        _ = base.class_labels()
    var invalid = y.copy()
    invalid[nm.Item(0)] = -3.5
    with assert_raises():
        base.fit(X, invalid)
    assert_false(base.__sklearn_is_fitted__())

    var qda = adapt_classifier(QuadraticDiscriminantAnalysis())
    qda.fit(X, y)
    var qda_update = qda.copy()
    var qda_params = qda_update.get_params()
    qda_params.tol = 1e-5
    _ = qda_update.set_params(qda_params)
    assert_true(qda.__sklearn_is_fitted__())
    assert_false(qda_update.__sklearn_is_fitted__())
    assert_equal(qda_update.get_params().tol, 1e-5)

    var unfitted = FrozenCalibrationEstimator(FrozenEstimator(base.copy()))
    with assert_raises():
        unfitted.fit(X, y)
    var frozen_lda = CalibratedClassifierCV(
        adapt_classifier(FrozenEstimator(LinearDiscriminantAnalysis()))
    )
    with assert_raises(contains="NotFittedError"):
        _ = frozen_lda.fit(X, y)
    assert_false(frozen_lda.__sklearn_is_fitted__())
    var splits = List[CalibrationSplit]()
    for fold in range(3):
        var train = List[Int]()
        var test = List[Int]()
        for i in range(30):
            if i % 3 == fold:
                test.append(i)
            else:
                train.append(i)
        splits.append(CalibrationSplit(train^, test^))
    var methods: List[String] = ["sigmoid", "isotonic", "temperature"]
    for method in methods:
        var model = CalibratedClassifierCV(adapt_classifier(LinearDiscriminantAnalysis()), method=method.copy(), cv=splits.copy(), ensemble=False)
        generic_fit(model, X, y)
        assert_equal(len(model.calibrated_classifiers_), 1)
        var params = model.get_params()
        var probabilities = model.set_params(params).fit(X, y).predict_proba(X)
        assert_equal(probabilities.shape[1], 2)
        assert_equal(model.classes_.item(0), -3)
        assert_equal(model.classes_.item(1), 7)
        for i in range(30):
            assert_true(abs(probabilities.item(i, 0) + probabilities.item(i, 1) - 1) < 1e-12)
        var before = model.predict_proba(X)
        with assert_raises():
            _ = model.fit(X, y, nm.ones[DType.float64](nm.Shape(30)))
        var after = model.predict_proba(X)
        assert_equal(before.item(0, 1), after.item(0, 1))
        assert_false(clone(model).__sklearn_is_fitted__())
        base.fit(X, y)
        var replacement = base.copy()
        var base_params = replacement.get_params()
        base_params.tol = 1e-5
        _ = replacement.set_params(base_params)
        assert_equal(base.get_params().tol, 1e-4)
        assert_equal(replacement.get_params().tol, 1e-5)
        assert_true(replacement.__sklearn_is_fitted__())
        assert_false(clone(base).__sklearn_is_fitted__())
        var base_display = CalibrationDisplay.from_estimator(base, X, y)
        assert_equal(base_display.estimator_name.value(), "LinearDiscriminantAnalysis")
        var frozen = FrozenCalibrationEstimator(FrozenEstimator(base.copy()))
        var copied = clone(frozen)
        assert_true(copied.__sklearn_is_fitted__())
        var frozen_model = CalibratedClassifierCV(frozen^, method=method.copy(), cv=splits.copy())
        _ = frozen_model.fit(X, y, nm.ones[DType.float64](nm.Shape(30)))
        assert_equal(len(frozen_model.calibrated_classifiers_), 1)
        var original = copied.response(X)
        var changed_y = y.copy()
        for i in range(30):
            changed_y[nm.Item(i)] = -3
        _ = copied.fit(X, changed_y)
        assert_equal(copied.response(X).item(0, 0), original.item(0, 0))
        var display = CalibrationDisplay.from_estimator(model, X, y)
        var args = argv()
        var path = args[1] if len(args) > 1 else "builds/calibration-consumer.svg"
        display.figure_.value().savefig(path)
        var p = nm.zeros[DType.float64](nm.Shape(30))
        for i in range(30):
            p[nm.Item(i)] = probabilities.item(i, 1)
        var curve = calibration_curve(y, p, pos_label=7.0, n_bins="cube_root")
        assert_true(curve[0].size > 0)
    print("PASS: default LinearSVC, production LDA, fluent contracts, frozen ownership, splits, curves and SVG")
