"""External consumer of all five public classifiers and fluent contracts."""
import numojo as nm
from std.testing import assert_equal, assert_true
from moxie.base import WeightedSupervisedEstimator
from moxie.naive_bayes import (
    GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB,
    NaiveBayesClassifier, NumericDomain, StringDomain,
)
from moxie._labels import StringTarget
from moxie._naive_bayes_dense import dense_from_lists


def configured_fit[E: NaiveBayesClassifier & WeightedSupervisedEstimator](
    mut model: E, params: E.Params, X: E.Input, y: E.Target, weights: E.SampleWeight,
) raises -> E.Prediction:
    return model.set_params(params).fit(X, y, sample_weight=weights).predict(X)


def incremental[E: NaiveBayesClassifier](
    mut model: E, X: E.Input, y: E.Target, var classes: E.Classes,
) raises -> E.Prediction where conforms_to(E.Classes, Deinitable):
    return model.partial_fit(X, y, classes^).partial_fit(X, y).predict(X)


def main() raises:
    var X: List[List[Float64]] = [[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
    var y: List[Int] = [7, 7, -2, -2]
    var classes: List[Int] = [-2, 7]
    var weights: List[Float64] = [1.0, 2.0, 1.0, 2.0]
    var gaussian = GaussianNB()
    var multinomial = MultinomialNB()
    var complement = ComplementNB()
    var bernoulli = BernoulliNB()
    var categorical = CategoricalNB()
    assert_equal(incremental(gaussian, X, y, classes.copy()), y)
    assert_equal(configured_fit(gaussian, gaussian.get_params(), X, y, weights), y)
    _ = gaussian.predict_proba(X)
    _ = gaussian.predict_log_proba(X)
    _ = gaussian.predict_joint_log_proba(X)
    assert_equal(gaussian.score(X, y, sample_weight=weights.copy()), 1.0)
    assert_equal(incremental(multinomial, X, y, classes.copy()), y)
    assert_equal(configured_fit(multinomial, multinomial.get_params(), X, y, weights), y)
    _ = multinomial.predict_proba(X)
    _ = multinomial.predict_log_proba(X)
    _ = multinomial.predict_joint_log_proba(X)
    assert_equal(multinomial.score(X, y, sample_weight=weights.copy()), 1.0)
    assert_equal(incremental(complement, X, y, classes.copy()), y)
    assert_equal(configured_fit(complement, complement.get_params(), X, y, weights), y)
    _ = complement.predict_proba(X)
    _ = complement.predict_log_proba(X)
    _ = complement.predict_joint_log_proba(X)
    assert_equal(complement.score(X, y, sample_weight=weights.copy()), 1.0)
    assert_equal(incremental(bernoulli, X, y, classes.copy()), y)
    assert_equal(configured_fit(bernoulli, bernoulli.get_params(), X, y, weights), y)
    _ = bernoulli.predict_proba(X)
    _ = bernoulli.predict_log_proba(X)
    _ = bernoulli.predict_joint_log_proba(X)
    assert_equal(bernoulli.score(X, y, sample_weight=weights.copy()), 1.0)
    assert_equal(incremental(categorical, X, y, classes.copy()), y)
    assert_equal(configured_fit(categorical, categorical.get_params(), X, y, weights), y)
    _ = categorical.predict_proba(X)
    _ = categorical.predict_log_proba(X)
    _ = categorical.predict_joint_log_proba(X)
    assert_equal(categorical.score(X, y, sample_weight=weights.copy()), 1.0)
    var dense = dense_from_lists(X)
    _ = gaussian.fit(dense, y).partial_fit(dense, y)
    _ = multinomial.fit(dense, y).partial_fit(dense, y)
    _ = complement.fit(dense, y).partial_fit(dense, y)
    _ = bernoulli.fit(dense, y).partial_fit(dense, y)
    _ = categorical.fit(dense, y).partial_fit(dense, y)
    _ = gaussian.predict_proba(dense)
    _ = multinomial.predict_proba(dense)
    _ = complement.predict_proba(dense)
    _ = bernoulli.predict_proba(dense)
    _ = categorical.predict_proba(dense)
    var strings = StringTarget(List[String](["z", "z", "a", "a"]))
    var text = BernoulliNB[StringDomain]()
    _ = text.fit(dense, strings).partial_fit(dense, strings)
    assert_equal(text.predict(dense).get(0), "z")
    var numeric = nm.array[DType.int64]("[7, 7, -2, -2]")
    var typed = GaussianNB[NumericDomain[DType.int64], DType.float32]()
    _ = typed.fit(dense, numeric).partial_fit(dense, numeric)
    assert_equal(typed.predict(dense).item(0), 7)
    print("PASS: all five estimators through public and generic fluent APIs")
