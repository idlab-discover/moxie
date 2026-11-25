"""Native contract and differential fixture for dense Naive Bayes calibration."""
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from moxie.base import clone
from moxie.calibration import (
    CalibratedClassifierCV,
    CalibrationEstimator,
    FrozenCalibrationEstimator,
    adapt_classifier,
)
from moxie.frozen import FrozenEstimator
from moxie.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from parity_io import Fixture, ResultLine, fixture_lines

def compare[E: CalibrationEstimator](
    var adapter: E, name: String, fixture: Fixture, frozen: Bool = False,
) raises:
    var X = fixture.array[DType.float64]("X")
    var y = fixture.array[DType.float64]("y")
    var weights = fixture.array[DType.float64]("weights")
    var query = fixture.array[DType.float64]("query")
    assert_true(adapter.uses_probabilities())
    assert_equal(adapter.is_frozen(), frozen)
    assert_equal(clone(adapter).__sklearn_is_fitted__(), frozen)
    if not frozen:
        with assert_raises():
            _ = adapter.class_labels()
        with assert_raises():
            _ = adapter.response(query)
        var invalid = y.copy()
        invalid.unsafe_set(0, 1.5)
        with assert_raises():
            adapter.fit(X, invalid)
        assert_false(adapter.__sklearn_is_fitted__())

    for method in ["sigmoid", "isotonic"]:
        var model = CalibratedClassifierCV(adapter.copy(), method=method, cv=2)
        _ = model.fit(X, y, sample_weight=weights.copy())
        assert_equal(len(model.calibrated_classifiers_), 1 if frozen else 2)
        assert_equal(model.estimator.__sklearn_is_fitted__(), frozen)
        var result = ResultLine(String(name, "/", method))
        result.add("classes", model.classes_)
        result.add("probabilities", model.predict_proba(query))
        result.add("predictions", model.predict(query))
        if frozen:
            var base = adapter.response(query)
            var fitted = model.calibrated_classifiers_[0].estimator.response(query)
            for i in range(base.shape[0]):
                for j in range(base.shape[1]):
                    assert_equal(base.item(i, j), fitted.item(i, j))
            result.add("base_probabilities", fitted)
        result.emit()


def main() raises:
    var lines = fixture_lines()
    assert_equal(len(lines), 1)
    var fixture = Fixture(lines[0])
    compare(adapt_classifier(GaussianNB()), "GaussianNB", fixture)
    compare(adapt_classifier(MultinomialNB()), "MultinomialNB", fixture)
    compare(adapt_classifier(ComplementNB()), "ComplementNB", fixture)
    compare(adapt_classifier(BernoulliNB()), "BernoulliNB", fixture)
    compare(adapt_classifier(CategoricalNB()), "CategoricalNB", fixture)

    var train = fixture.array[DType.float64]("train")
    var train_y = fixture.array[DType.float64]("train_y")
    var gaussian = adapt_classifier(GaussianNB())
    gaussian.fit(train, train_y)
    compare(
        adapt_classifier(FrozenEstimator(gaussian.estimator().copy())),
        "GaussianNB/frozen", fixture, frozen=True,
    )
    compare(
        FrozenCalibrationEstimator(FrozenEstimator(gaussian^)),
        "GaussianNB/frozen_adapter", fixture, frozen=True,
    )
    var multinomial = adapt_classifier(MultinomialNB())
    multinomial.fit(train, train_y)
    compare(
        adapt_classifier(FrozenEstimator(multinomial.estimator().copy())),
        "MultinomialNB/frozen", fixture, frozen=True,
    )
    var complement = adapt_classifier(ComplementNB())
    complement.fit(train, train_y)
    compare(
        adapt_classifier(FrozenEstimator(complement.estimator().copy())),
        "ComplementNB/frozen", fixture, frozen=True,
    )
    var bernoulli = adapt_classifier(BernoulliNB())
    bernoulli.fit(train, train_y)
    compare(
        adapt_classifier(FrozenEstimator(bernoulli.estimator().copy())),
        "BernoulliNB/frozen", fixture, frozen=True,
    )
    var categorical = adapt_classifier(CategoricalNB())
    categorical.fit(train, train_y)
    compare(
        adapt_classifier(FrozenEstimator(categorical.estimator().copy())),
        "CategoricalNB/frozen", fixture, frozen=True,
    )
