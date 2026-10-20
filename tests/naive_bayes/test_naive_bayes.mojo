"""Estimator lifecycle, validation, and independently checkable NB fixtures."""
from numojo import Item
from std.math import exp, isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.naive_bayes import (
    NaiveBayesClassifier,
    GaussianNB,
    MultinomialNB,
    ComplementNB,
    BernoulliNB,
    CategoricalNB,
    CategoricalNBParams,
)


def lifecycle(
    mut model: GaussianNB[],
    X: List[List[Float64]],
    y: List[Int],
    other_y: List[Int],
    var classes: List[Int],
    bad_X: List[List[Float64]],
    bad_y: List[Int],
) raises:
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.predict_proba(X)
    with assert_raises():
        _ = model.predict_log_proba(X)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.partial_fit(X, y, classes=classes^)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.score(X, y), 1.0)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var fitted_copy = model.copy()
    # A regular fit replaces the previous classes and statistics.
    _ = model.fit(X, other_y)
    assert_equal(model.score(X, other_y), 1.0)
    assert_equal(fitted_copy.score(X, y), 1.0)
    with assert_raises():
        _ = model.predict(bad_X)
    with assert_raises():
        _ = model.fit(X, bad_y)
    with assert_raises():
        _ = model.partial_fit(bad_X, other_y)
    with assert_raises():
        _ = model.partial_fit(X, y)
    # Failed updates leave the previous fitted model usable.
    assert_equal(model.score(X, other_y), 1.0)
    _ = model.set_params(model.get_params())
    _ = model.fit(X, other_y)
    assert_equal(model.score(X, other_y), 1.0)


def lifecycle(
    mut model: MultinomialNB[],
    X: List[List[Float64]],
    y: List[Int],
    other_y: List[Int],
    var classes: List[Int],
    bad_X: List[List[Float64]],
    bad_y: List[Int],
) raises:
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.predict_proba(X)
    with assert_raises():
        _ = model.predict_log_proba(X)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.partial_fit(X, y, classes=classes^)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.score(X, y), 1.0)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var fitted_copy = model.copy()
    # A regular fit replaces the previous classes and statistics.
    _ = model.fit(X, other_y)
    assert_equal(model.score(X, other_y), 1.0)
    assert_equal(fitted_copy.score(X, y), 1.0)
    with assert_raises():
        _ = model.predict(bad_X)
    with assert_raises():
        _ = model.fit(X, bad_y)
    with assert_raises():
        _ = model.partial_fit(bad_X, other_y)
    with assert_raises():
        _ = model.partial_fit(X, y)
    # Failed updates leave the previous fitted model usable.
    assert_equal(model.score(X, other_y), 1.0)
    _ = model.set_params(model.get_params())
    _ = model.fit(X, other_y)
    assert_equal(model.score(X, other_y), 1.0)


def lifecycle(
    mut model: ComplementNB[],
    X: List[List[Float64]],
    y: List[Int],
    other_y: List[Int],
    var classes: List[Int],
    bad_X: List[List[Float64]],
    bad_y: List[Int],
) raises:
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.predict_proba(X)
    with assert_raises():
        _ = model.predict_log_proba(X)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.partial_fit(X, y, classes=classes^)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.score(X, y), 1.0)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var fitted_copy = model.copy()
    # A regular fit replaces the previous classes and statistics.
    _ = model.fit(X, other_y)
    assert_equal(model.score(X, other_y), 1.0)
    assert_equal(fitted_copy.score(X, y), 1.0)
    with assert_raises():
        _ = model.predict(bad_X)
    with assert_raises():
        _ = model.fit(X, bad_y)
    with assert_raises():
        _ = model.partial_fit(bad_X, other_y)
    with assert_raises():
        _ = model.partial_fit(X, y)
    # Failed updates leave the previous fitted model usable.
    assert_equal(model.score(X, other_y), 1.0)
    _ = model.set_params(model.get_params())
    _ = model.fit(X, other_y)
    assert_equal(model.score(X, other_y), 1.0)


def lifecycle(
    mut model: BernoulliNB[],
    X: List[List[Float64]],
    y: List[Int],
    other_y: List[Int],
    var classes: List[Int],
    bad_X: List[List[Float64]],
    bad_y: List[Int],
) raises:
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.predict_proba(X)
    with assert_raises():
        _ = model.predict_log_proba(X)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.partial_fit(X, y, classes=classes^)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.score(X, y), 1.0)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var fitted_copy = model.copy()
    # A regular fit replaces the previous classes and statistics.
    _ = model.fit(X, other_y)
    assert_equal(model.score(X, other_y), 1.0)
    assert_equal(fitted_copy.score(X, y), 1.0)
    with assert_raises():
        _ = model.predict(bad_X)
    with assert_raises():
        _ = model.fit(X, bad_y)
    with assert_raises():
        _ = model.partial_fit(bad_X, other_y)
    with assert_raises():
        _ = model.partial_fit(X, y)
    # Failed updates leave the previous fitted model usable.
    assert_equal(model.score(X, other_y), 1.0)
    _ = model.set_params(model.get_params())
    _ = model.fit(X, other_y)
    assert_equal(model.score(X, other_y), 1.0)


def lifecycle(
    mut model: CategoricalNB[],
    X: List[List[Float64]],
    y: List[Int],
    other_y: List[Int],
    var classes: List[Int],
    bad_X: List[List[Float64]],
    bad_y: List[Int],
) raises:
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.predict_proba(X)
    with assert_raises():
        _ = model.predict_log_proba(X)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.partial_fit(X, y, classes=classes^)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.score(X, y), 1.0)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var fitted_copy = model.copy()
    # A regular fit replaces the previous classes and statistics.
    _ = model.fit(X, other_y)
    assert_equal(model.score(X, other_y), 1.0)
    assert_equal(fitted_copy.score(X, y), 1.0)
    with assert_raises():
        _ = model.predict(bad_X)
    with assert_raises():
        _ = model.fit(X, bad_y)
    with assert_raises():
        _ = model.partial_fit(bad_X, other_y)
    with assert_raises():
        _ = model.partial_fit(X, y)
    # Failed updates leave the previous fitted model usable.
    assert_equal(model.score(X, other_y), 1.0)
    _ = model.set_params(model.get_params())
    _ = model.fit(X, other_y)
    assert_equal(model.score(X, other_y), 1.0)


def test_lifecycle() raises:
    var X: List[List[Float64]] = [
        [0.0, 1.0],
        [0.0, 2.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ]
    var y: List[Int] = [2, 2, 9, 9]
    var other: List[Int] = [-3, -3, 7, 7]
    var classes: List[Int] = [9, 2]
    var bad_X: List[List[Float64]] = [[1.0]]
    var bad_y: List[Int] = [2]
    var gaussian = GaussianNB()
    lifecycle(gaussian, X, y, other, classes.copy(), bad_X, bad_y)
    var multinomial = MultinomialNB()
    lifecycle(multinomial, X, y, other, classes.copy(), bad_X, bad_y)
    var complement = ComplementNB()
    lifecycle(complement, X, y, other, classes.copy(), bad_X, bad_y)
    var bernoulli = BernoulliNB()
    lifecycle(bernoulli, X, y, other, classes.copy(), bad_X, bad_y)
    var categorical = CategoricalNB()
    lifecycle(categorical, X, y, other, classes.copy(), bad_X, bad_y)


def test_counts_and_params() raises:
    var X: List[List[Float64]] = [[2.0, 0.0], [0.0, 2.0]]
    var y: List[Int] = [9, 2]
    var weights: List[Float64] = [2.0, 1.0]
    var model = MultinomialNB()
    _ = model.fit(X, y, sample_weight=weights)
    assert_equal(model.classes_[0], 2)
    assert_equal(model.class_count_.item(0), 1.0)
    assert_equal(model.class_count_.item(1), 2.0)
    assert_equal(model.feature_count_.item(1, 0), 4.0)
    assert_true(abs(exp(model.feature_log_prob_.item(1, 0)) - 5.0 / 6.0) < 1e-7)
    assert_true(abs(exp(model.class_log_prior_[1]) - 2.0 / 3.0) < 1e-7)
    var prior: List[Float64] = [0.25, 0.75]
    var configured = MultinomialNB(class_prior=prior^)
    var params = configured.get_params()
    params.class_prior.value()[0] = 0.5
    assert_equal(configured.class_prior.value()[0], 0.25)
    _ = configured.set_params(params)
    assert_equal(configured.class_prior.value()[0], 0.5)
    _ = configured.set_params(alpha=0.5, fit_prior=False)
    assert_equal(configured.alpha, 0.5)
    assert_false(configured.fit_prior)
    _ = configured.set_params(
        class_prior=Optional[Optional[List[Float64]]](Optional[List[Float64]]())
    )
    assert_false(configured.class_prior)

    var gaussian = GaussianNB()
    _ = gaussian.fit(X, y)
    _ = gaussian.set_params(var_smoothing=0.1)
    assert_equal(clone(gaussian).var_smoothing, 0.1)
    var complement = ComplementNB()
    _ = complement.set_params(norm=True)
    assert_true(clone(complement).norm)
    var bernoulli = BernoulliNB()
    _ = bernoulli.set_params(
        binarize=Optional[Optional[Float64]](Optional[Float64]())
    )
    assert_false(clone(bernoulli).binarize)
    var categorical = CategoricalNB()
    _ = categorical.set_params(
        min_categories=Optional[Optional[Int]](Optional[Int](4))
    )
    _ = categorical.fit(X, y)
    assert_equal(categorical.category_count_[0].shape[1], 4)


def test_validation() raises:
    var X: List[List[Float64]] = [[1.0, 2.0], [2.0, 1.0]]
    var y: List[Int] = [0, 1]
    var model = MultinomialNB()
    var negative: List[List[Float64]] = [[-1.0, 2.0], [2.0, 1.0]]
    with assert_raises():
        _ = model.fit(negative, y)
    var ragged: List[List[Float64]] = [[1.0], [2.0, 3.0]]
    with assert_raises():
        _ = model.fit(ragged, y)
    var invalid: List[List[Float64]] = [[Float64("nan"), 1.0], [2.0, 1.0]]
    with assert_raises():
        _ = model.fit(invalid, y)
    var empty = List[List[Float64]]()
    with assert_raises():
        _ = model.fit(empty, y)
    var negative_weights: List[Float64] = [-1.0, 2.0]
    with assert_raises():
        _ = model.fit(X, y, sample_weight=negative_weights)
    var zeros: List[Float64] = [0.0, 0.0]
    with assert_raises():
        _ = model.fit(X, y, sample_weight=zeros)
    var short_weights: List[Float64] = [1.0]
    with assert_raises():
        _ = model.fit(X, y, sample_weight=short_weights)
    _ = model.set_params(alpha=-1.0)
    with assert_raises():
        _ = model.fit(X, y)
    assert_false(model.__sklearn_is_fitted__())
    _ = model.set_params(alpha=1.0)
    _ = model.fit(X, y)
    var different_classes: List[Int] = [0, 1, 2]
    with assert_raises():
        _ = model.partial_fit(X, y, classes=different_classes^)

    var prior: List[Float64] = [0.1, 0.1]
    var gaussian = GaussianNB(priors=prior^)
    with assert_raises():
        _ = gaussian.fit(X, y)
    var categorical = CategoricalNB()
    var fractional: List[List[Float64]] = [[0.5, 1.0], [1.0, 0.0]]
    _ = categorical.fit(fractional, y)
    assert_equal(categorical.category_count_[0].item(0, 0), 1.0)
    _ = categorical.fit(X, y)
    var unknown: List[List[Float64]] = [[3.0, 1.0]]
    with assert_raises():
        _ = categorical.predict(unknown)


def test_degenerate_likelihood() raises:
    # Zero smoothing can legitimately produce NaNs. Preserve them through
    # normalization and exponentiation instead of turning them into infinity.
    var X: List[List[Float64]] = [[1.0, 0.0], [0.0, 1.0]]
    var y: List[Int] = [0, 1]
    var query: List[List[Float64]] = [[0.0, 0.0]]
    var model = MultinomialNB(alpha=0.0)
    _ = model.fit(X, y)
    var probabilities = model.predict_proba(query)
    assert_true(isnan(probabilities[0][0]))
    assert_true(isnan(probabilities[0][1]))
    var constant: List[List[Float64]] = [[1.0], [1.0]]
    var gaussian = GaussianNB()
    _ = gaussian.fit(constant, y)
    var gaussian_probabilities = gaussian.predict_proba(constant)
    assert_true(isnan(gaussian_probabilities[0][0]))


def test_categorical_minima() raises:
    var legacy_params = CategoricalNBParams(1.0, True, True, None, None)
    assert_false(Bool(legacy_params.min_categories_values))
    var X: List[List[Float64]] = [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]
    var y: List[Int] = [1, 1, 2, 2]
    var minima: List[Int] = [3, 4]
    var model = CategoricalNB(min_categories_values=minima.copy(), fit_prior=False)
    minima[0] = 9
    _ = model.fit(X, y)
    assert_equal(model.n_categories_[0], 3)
    assert_equal(model.n_categories_[1], 4)
    assert_equal(model.category_count_[0].shape[1], 3)
    assert_equal(model.category_count_[1].shape[1], 4)
    var query: List[List[Float64]] = [[0.0, 3.0]]
    assert_equal(model.predict(query)[0], 1)
    var params = model.get_params()
    params.min_categories_values.value()[0] = 5
    assert_equal(model.min_categories_values.value()[0], 3)
    var cloned = clone(model)
    assert_false(cloned.__sklearn_is_fitted__())
    cloned.min_categories_values.value()[0] = 7
    assert_equal(model.min_categories_values.value()[0], 3)
    var copied = model.copy()
    copied.min_categories_values.value()[1] = 8
    copied.category_count_[0][Item(0, 0)] = 100.0
    assert_equal(model.min_categories_values.value()[1], 4)
    assert_equal(model.category_count_[0].item(0, 0), 2.0)
    _ = model.set_params(params)
    params.min_categories_values.value()[0] = 6
    _ = model.partial_fit(X, y)
    assert_equal(model.n_categories_[0], 5)
    assert_equal(model.class_count_.item(0), 4.0)
    # Smaller per-batch minima do not discard earlier category counts.
    var smaller: List[Int] = [0, 2]
    _ = model.set_params(min_categories_values=Optional[Optional[List[Int]]](smaller.copy()))
    _ = model.partial_fit(X, y)
    assert_equal(model.n_categories_[0], 2)
    assert_equal(model.category_count_[0].shape[1], 5)
    _ = model.fit(X, y)
    assert_equal(model.category_count_[0].shape[1], 2)
    # Bad configuration fails without replacing fitted state.
    var bad_minima: List[List[Int]] = [[], [3, 4, 5], [3, -1]]
    for bad in bad_minima:
        _ = model.set_params(min_categories_values=Optional[Optional[List[Int]]](bad.copy()))
        with assert_raises():
            _ = model.partial_fit(X, y)
        with assert_raises():
            _ = model.fit(X, y)
        assert_equal(model.class_count_.item(0), 2.0)
        assert_equal(model.category_count_[0].shape[1], 2)
        assert_equal(model.predict(X)[0], 1)
    # A scalar update clears the vector; explicit None restores inference.
    _ = model.set_params(min_categories=Optional[Optional[Int]](6))
    assert_false(Bool(model.min_categories_values))
    _ = model.fit(X, y)
    assert_equal(model.n_categories_[0], 6)
    assert_equal(model.n_categories_[1], 6)
    _ = model.set_params(min_categories=Optional[Optional[Int]](Optional[Int](None)))
    _ = model.fit(X, y)
    assert_equal(model.n_categories_[0], 2)
    # A vector supplied alongside a scalar has precedence.
    _ = model.set_params(min_categories=Optional[Optional[Int]](10),
                     min_categories_values=Optional[Optional[List[Int]]](smaller.copy()))
    _ = model.fit(X, y)
    assert_equal(model.n_categories_[0], 2)
    _ = model.set_params(min_categories_values=Optional[Optional[List[Int]]](Optional[List[Int]](None)))
    _ = model.fit(X, y)
    assert_equal(model.n_categories_[0], 10)


def main() raises:
    test_categorical_minima()
    test_lifecycle()
    test_counts_and_params()
    test_validation()
    test_degenerate_likelihood()
    print("Naive Bayes lifecycle, counts, and validation checks passed")
