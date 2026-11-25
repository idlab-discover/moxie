"""Train and calibrate StandardScaler → SVC as a single fold estimator."""
from numojo import NDArray, Item, Shape, ones
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from moxie.base import ClassifierMixin, clone
from moxie.calibration import CalibratedClassifierCV, adapt_classifier
from moxie.pipeline import Pipeline, make_pipeline
from moxie.preprocessing import StandardScaler
from moxie.svm import SVC
from parity_io import encode

comptime Dense = NDArray[DType.float64]
comptime ScaledSVC = Pipeline[StandardScaler[], SVC[]]


def main() raises:
    comptime assert conforms_to(ScaledSVC, ClassifierMixin)
    comptime assert conforms_to(Pipeline[StandardScaler[], ScaledSVC], ClassifierMixin)
    var X = Dense(Shape(36, 2))
    var y = Dense(Shape(36))
    for i in range(36):
        var label = Float64(i % 2)
        X[Item(i, 0)] = label * 1.5 + Float64((i * 7) % 11) / 4
        X[Item(i, 1)] = Float64((i * 5) % 13) / 2 - label
        y[Item(i)] = 9.0 if label == 0.0 else -3.0

    var query = Dense(Shape(8, 2))
    for i in range(8):
        query[Item(i, 0)] = Float64(i) / 2
        query[Item(i, 1)] = Float64((i * 7) % 9) / 2

    for ensemble in [True, False]:
        var estimator = make_pipeline(StandardScaler(), SVC(tol=1e-8))
        var model = CalibratedClassifierCV(adapt_classifier(estimator^), cv=3, ensemble=ensemble)
        with assert_raises():
            _ = model.predict_proba(query)
        _ = model.fit(X, y)
        assert_false(model.estimator.__sklearn_is_fitted__())
        assert_equal(len(model.calibrated_classifiers_), 3 if ensemble else 1)
        assert_equal(model.classes_.item(0), -3.0)
        assert_equal(model.classes_.item(1), 9.0)
        var means = Dense(Shape(len(model.calibrated_classifiers_), 2))
        for i in range(means.shape[0]):
            ref fitted = model.calibrated_classifiers_[i].estimator.estimator()
            ref mean = fitted.transformer.mean_.value()
            for j in range(means.shape[1]):
                means[Item(i, j)] = mean.item(j)
        var probabilities = model.predict_proba(query)
        assert_equal(probabilities.ndim, 2)
        assert_equal(probabilities.shape[0], 8)
        assert_equal(probabilities.shape[1], 2)
        with assert_raises():
            _ = model.fit(X, y, sample_weight=ones[DType.float64](Shape(36)))
        assert_equal(model.classes_.item(0), -3.0)
        var fresh = clone(model)
        assert_false(fresh.__sklearn_is_fitted__())
        _ = fresh.fit(X, y)
        var cloned = fresh.predict_proba(query)
        for i in range(probabilities.size):
            assert_equal(probabilities.item(i), cloned.item(i))
        var labels = model.predict(query)
        var shifted = y + 2.0
        _ = model.fit(X, shifted)
        assert_equal(model.classes_.item(0), -1.0)
        assert_equal(model.classes_.item(1), 11.0)
        assert_equal(fresh.classes_.item(0), -3.0)
        var refit = model.predict(query)
        for i in range(refit.size):
            assert_equal(refit.item(i), labels.item(i) + 2.0)
            assert_true(labels.item(i) == -3.0 or labels.item(i) == 9.0)
        var name = "ensemble" if ensemble else "pooled"
        print(String(
            "result|", name, "|", encode("means", means), "|",
            encode("probabilities", probabilities), "|",
            encode("predictions", labels), "|", encode("refit_classes", model.classes_),
        ))
