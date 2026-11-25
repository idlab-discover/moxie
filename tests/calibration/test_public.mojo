"""Isolated consumer using production classifiers through calibration."""
import numojo as nm
from std.math import nan
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
from moxie.linear_model import LogisticRegression
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from moxie.svm import LinearSVC

comptime Array = nm.NDArray[DType.float64]


def generic_fit[E: SupervisedEstimator](mut model: E, X: E.Input, y: E.Target) raises:
    _ = model.fit(X, y)


def test_generic_frozen_classifier(X: Array, y: Array) raises:
    var pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    var frozen_unfitted = FrozenEstimator(pipeline.copy())
    var model = CalibratedClassifierCV(adapt_classifier(frozen_unfitted.copy()), cv=3)
    with assert_raises(contains="NotFittedError"):
        _ = model.fit(X, y)
    assert_false(model.__sklearn_is_fitted__())

    _ = pipeline.fit(X, y.astype[DType.int64]())
    var frozen = FrozenEstimator(pipeline^)
    var adapter = adapt_classifier(frozen.copy())
    assert_true(adapter.is_frozen())
    assert_false(adapter.uses_probabilities())
    assert_true(clone(adapter).__sklearn_is_fitted__())
    var scores = frozen.decision_function(X)
    var classes = frozen.get_classes()
    var params = model.get_params()
    params.estimator = adapter.copy()
    _ = model.set_params(params)

    _ = model.fit(X, y)
    var first = model.predict_proba(X)
    _ = model.fit(X, y)
    var second = model.predict_proba(X)
    assert_equal(len(model.calibrated_classifiers_), 1)
    assert_equal(first.shape, nm.Shape(X.shape[0], classes.size))
    var after = frozen.decision_function(X)
    for i in range(scores.size):
        assert_equal(after.item(i), scores.item(i))
    for i in range(first.size):
        assert_equal(second.item(i), first.item(i))
    for i in range(classes.size):
        assert_equal(model.classes_.item(i), Float64(classes.item(i)))
    assert_false(frozen_unfitted.__sklearn_is_fitted__())


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


def test_float32_features(X: Array, y: Array) raises:
    var rows = X.shape[0]
    var owner = nm.full[DType.float32](nm.Shape(rows, 4), Float32(999))
    var strided = owner.view_with_layout(
        nm.Shape(rows, 2), nm.NDArrayStrides(-4, 2), 4 * (rows - 1)
    )
    var column_major = nm.NDArray[DType.float32](X.shape, order="F")
    var weight = Array(nm.Shape(rows))
    for i in range(rows):
        weight[nm.Item(i)] = Float64(i + 1)
        for j in range(2):
            var value = Float32(X.item(i, j))
            strided[nm.Item(i, j)] = value
            column_major[nm.Item(i, j)] = value

    var empty = nm.zeros[DType.float32](nm.Shape(0, 2))
    var model = CalibratedClassifierCV(cv=3, ensemble=False)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict_proba(empty)
    var expected_model = clone(model)
    var x64 = column_major.astype[DType.float64]()
    var expected = expected_model.fit(x64, y, weight.copy()).predict_proba(x64)
    var actual = model.fit(strided, y, sample_weight=weight.copy()).predict_proba(column_major)
    var predictions = model.predict(strided)
    var expected_predictions = expected_model.predict(x64)
    assert_equal(model.score(strided, y, weight.copy()), expected_model.score(x64, y, weight^))
    assert_equal(model.n_features_in_, 2)
    assert_equal(model.classes_.item(0), -3)
    assert_equal(model.classes_.item(1), 7)
    assert_false(clone(model).__sklearn_is_fitted__())

    with assert_raises(contains="nonempty matrix"):
        _ = model.fit(empty, y)
    var wrong_width = nm.ones[DType.float32](nm.Shape(rows, 1))
    with assert_raises(contains="inconsistent feature count"):
        _ = model.predict(wrong_width)
    var invalid = column_major.copy()
    invalid[nm.Item(0, 0)] = nan[DType.float32]()
    with assert_raises(contains="input must be finite"):
        _ = model.fit(invalid, y)
    var preserved = model.predict_proba(strided)
    for i in range(rows):
        assert_equal(predictions.item(i), expected_predictions.item(i))
        for j in range(2):
            assert_equal(actual.item(i, j), expected.item(i, j))
            assert_equal(preserved.item(i, j), expected.item(i, j))

    var refit = model.fit(column_major, y).predict_proba(strided)
    var expected_refit = expected_model.fit(x64, y).predict_proba(x64)
    for i in range(rows):
        for j in range(2):
            assert_equal(refit.item(i, j), expected_refit.item(i, j))


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(30, 2))
    var y = nm.zeros[DType.float64](nm.Shape(30))
    for i in range(30):
        X[nm.Item(i, 0)] = Float64(i % 2) + Float64(i % 7) / 4
        X[nm.Item(i, 1)] = Float64(i % 11) / 3
        y[nm.Item(i)] = -3.0 if i % 2 == 0 else 7.0
    test_default_classifier(X, y)
    test_generic_frozen_classifier(X, y)
    test_float32_features(X, y)
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
