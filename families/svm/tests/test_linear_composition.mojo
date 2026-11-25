"""Dense LinearSVC through the public calibration, selection and Pipeline APIs."""
from numojo import Item, NDArray, Shape
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie.base import clone
from moxie.calibration import CalibratedClassifierCV, adapt_classifier
from moxie.feature_selection import (
    BuiltinScorer, LinearSVCImportance, SelectFromModel, SelectKBest,
)
from moxie.pipeline import make_pipeline
from moxie.svm import LinearSVC


comptime Dense = NDArray[DType.float64]
comptime Labels = List[Float64]


def classifier() raises -> LinearSVC[]:
    return LinearSVC(C=0.7, dual=False, tol=1e-9, random_state=0)


def check_calibration(X: Dense, y: Dense, query: Dense) raises:
    var model = CalibratedClassifierCV(adapt_classifier(classifier()), cv=3, ensemble=True)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict_proba(query)
    _ = model.fit(X, y)
    assert_equal(len(model.calibrated_classifiers_), 3)
    assert_false(model.estimator.__sklearn_is_fitted__())
    for fold in model.calibrated_classifiers_:
        assert_true(fold.estimator.estimator().__sklearn_is_fitted__())
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.estimator.get_params().C, 0.7)
    var probabilities = model.predict_proba(query)
    var labels = model.predict(query)
    assert_equal(probabilities.shape, Shape(query.shape[0], 2))
    assert_equal(labels.shape, Shape(query.shape[0]))
    for j in range(model.classes_.size):
        print("calibration_class", model.classes_.item(j))
    for i in range(query.shape[0]):
        assert_true(abs(probabilities.item(i, 0) + probabilities.item(i, 1) - 1.0) < 1e-12)
        print("calibration_prediction", labels.item(i))
        for j in range(2):
            print("calibration_probability", probabilities.item(i, j))


def check_model_selection(X: Dense, y: Labels, query: Dense) raises:
    var model = SelectFromModel(LinearSVCImportance(classifier()))
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(query)
    _ = model.fit(X, y.copy())
    assert_false(model.get_params().estimator.__sklearn_is_fitted__())
    assert_true(model.estimator_().__sklearn_is_fitted__())
    var support = model.get_support(indices=True)[List[Int]].copy()
    assert_equal(support, [0])
    var reduced = model.transform(query)
    assert_equal(reduced.shape, Shape(query.shape[0], len(support)))
    for j in support:
        print("model_feature", j)
    for i in range(reduced.size):
        print("model_transform", reduced.item(i))


def check_univariate_pipeline(X: Dense, y: Labels, query: Dense) raises:
    comptime Scorer = BuiltinScorer[DType.float64, DType.float64, Labels]
    var model = make_pipeline(SelectKBest(Scorer("f_classif"), k=2), classifier())
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(query)
    _ = model.fit(X, y)
    assert_equal(model.estimator.n_features_in_, 2)
    var support = model.transformer.get_support(indices=True)[List[Int]].copy()
    assert_equal(support, [0, 1])
    var predictions = model.predict(query)
    var decisions = model.decision_function(query)
    assert_equal(len(predictions), query.shape[0])
    assert_equal(decisions.shape, Shape(query.shape[0]))
    for j in support:
        print("pipeline_feature", j)
    for i in range(len(predictions)):
        print("pipeline_prediction", predictions[i])
        print("pipeline_decision", decisions.item(i))


def main() raises:
    var X = Dense(Shape(18, 3))
    var numeric = Dense(Shape(18))
    var labels = Labels(capacity=18)
    for i in range(18):
        var side = Float64(2 * (i % 2) - 1)
        X[Item(i, 0)] = 1.5 * side + Float64(i % 5 - 2) / 4
        X[Item(i, 1)] = 0.75 * side + Float64(i % 7 - 3) / 4
        X[Item(i, 2)] = Float64(i % 3 - 1) / 2
        var label = 2.0 + 5.0 * side
        numeric[Item(i)] = label
        labels.append(label)

    var query = Dense(Shape(7, 3))
    for i in range(7):
        query[Item(i, 0)] = Float64(i - 3) / 2 + 0.125
        query[Item(i, 1)] = Float64(i % 3 - 1) / 4
        query[Item(i, 2)] = Float64(i % 2) / 2

    check_calibration(X, numeric, query)
    check_model_selection(X, labels, query)
    check_univariate_pipeline(X, labels, query)
