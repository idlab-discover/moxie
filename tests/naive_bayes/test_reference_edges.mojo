"""Edge behaviors with expected values from the pinned sklearn reference.

Values were produced with scikit-learn 4358542a1bd39af2e7e1cbace8b9fed067473c5b
(`sklearn/naive_bayes.py`). Float tolerances are 1e-12 relative: every value is
a short closed-form expression (a few logs and sums), so only libm ulps differ.
"""
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_raises
from moxie.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB


def close(actual: Float64, expected: Float64) raises:
    if abs(actual - expected) > 1e-12 * (1.0 + abs(expected)):
        print("Mismatch: actual", actual, "expected", expected)
    assert_true(abs(actual - expected) <= 1e-12 * (1.0 + abs(expected)))


def data() -> List[List[Float64]]:
    return [[1.0, 0, 2], [0, 3, 1], [2, 1, 0], [0, 0, 4]]


def labels() -> List[Int]:
    return [0, 1, 0, 1]


def test_count_models_predict_negative_values() raises:
    # sklearn checks nonnegativity only while counting, not in predict*.
    var negative: List[List[Float64]] = [[-1.0, 0, 2]]
    var multinomial = MultinomialNB()
    _ = multinomial.fit(data(), labels())
    var joint = multinomial.predict_joint_log_proba(negative)
    close(joint[0][0], -2.0794415416798357)
    close(joint[0][1], 0.49247648509779396)
    var complement = ComplementNB()
    _ = complement.fit(data(), labels())
    var scores = complement.predict_joint_log_proba(negative)
    close(scores[0][0], -1.1856236656577395)
    close(scores[0][1], 1.3862943611198908)
    var rejecting = MultinomialNB()
    with assert_raises(contains="nonnegative"):
        _ = rejecting.fit(negative, List[Int]([0]))


def test_unnormalized_class_prior_is_logged_verbatim() raises:
    var model = MultinomialNB(class_prior=List[Float64]([2.0, 6.0]))
    _ = model.fit(data(), labels())
    close(model.class_log_prior_[0], 0.6931471805599453)
    close(model.class_log_prior_[1], 1.791759469228055)


def test_partial_fit_after_fit_accumulates() raises:
    var model = MultinomialNB()
    _ = model.fit(data(), labels())
    var first: List[List[Float64]] = [[1.0, 0, 2], [0, 3, 1]]
    _ = model.partial_fit(first, List[Int]([0, 1]))
    assert_equal(model.class_count_.item(0), 3.0)
    assert_equal(model.class_count_.item(1), 3.0)
    assert_equal(model.feature_count_.item(0, 0), 4.0)
    assert_equal(model.feature_count_.item(0, 2), 4.0)
    assert_equal(model.feature_count_.item(1, 1), 6.0)
    assert_equal(model.feature_count_.item(1, 2), 6.0)
    # A different class set after fit raises, as _check_partial_fit_first_call does.
    with assert_raises(contains="classes"):
        _ = model.partial_fit(data(), labels(), classes=List[Int]([0, 1, 2]))
    var gaussian = GaussianNB()
    _ = gaussian.fit(data(), labels())
    with assert_raises(contains="classes"):
        _ = gaussian.partial_fit(data(), labels(), classes=List[Int]([0, 1, 2]))


def test_gaussian_constant_feature_uses_epsilon() raises:
    var X: List[List[Float64]] = [[1.0, 5], [2, 5], [3, 5], [4, 5]]
    var model = GaussianNB()
    _ = model.fit(X, labels())
    close(model.var_.item(0, 0), 1.00000000125)
    close(model.var_.item(0, 1), 1.25e-09)
    close(model.epsilon_, 1.25e-09)
    var query: List[List[Float64]] = [[2.5, 5.0]]
    var joint = model.predict_joint_log_proba(query)
    close(joint[0][0], 7.594036895378061)
    close(joint[0][1], 7.594036895378061)
    # Exact tie: argmax picks the first class.
    assert_equal(model.predict(query)[0], 0)


def test_gaussian_zero_variance_propagates_nan() raises:
    # All-constant X gives epsilon_ == 0 and var_ == 0; sklearn returns NaN
    # likelihoods/probabilities and np.argmax picks the first NaN (class 0).
    var X: List[List[Float64]] = [[3.0, 3], [3, 3], [3, 3], [3, 3]]
    var model = GaussianNB()
    _ = model.fit(X, labels())
    assert_equal(model.var_.item(0, 0), 0.0)
    assert_equal(model.epsilon_, 0.0)
    var query: List[List[Float64]] = [[3.0, 3], [2, 3]]
    var joint = model.predict_joint_log_proba(query)
    var proba = model.predict_proba(query)
    for i in range(2):
        for c in range(2):
            assert_true(isnan(joint[i][c]))
            assert_true(isnan(proba[i][c]))
    var predicted = model.predict(query)
    assert_equal(predicted[0], 0)
    assert_equal(predicted[1], 0)
    # One training sample: zero variance and NaN probabilities as well.
    var one_row: List[List[Float64]] = [[1.0, 0, 2]]
    var single = GaussianNB()
    _ = single.fit(one_row, List[Int]([0]))
    assert_equal(single.var_.item(0, 0), 0.0)
    assert_true(isnan(single.predict_proba(data())[1][0]))


def test_gaussian_priors_with_absent_class() raises:
    var model = GaussianNB(priors=List[Float64]([0.2, 0.3, 0.5]))
    _ = model.partial_fit(data(), labels(), classes=List[Int]([0, 1, 2]))
    close(model.class_prior_.item(2), 0.5)
    assert_equal(model.class_count_.item(2), 0.0)
    close(model.var_.item(2, 0), 2.1875e-09)
    close(model.var_.item(1, 1), 2.2500000021875)
    var proba = model.predict_proba(data())
    close(proba[1][0], 3.1580109467692992e-11)
    close(proba[3][0], 5.7098168679108235e-08)
    assert_equal(proba[3][2], 0.0)


def test_parameter_validation_matches_reference() raises:
    var codes: List[List[Float64]] = [[0.0, 1], [1, 0], [2, 1], [0, 0]]
    var unseen: List[List[Float64]] = [[3.0, 0]]
    var no_minimum = CategoricalNB(min_categories=0)
    with assert_raises(contains="min_categories"):
        _ = no_minimum.fit(codes, labels())
    var negative_smoothing = GaussianNB(var_smoothing=-1.0)
    with assert_raises(contains="var_smoothing"):
        _ = negative_smoothing.fit(data(), labels())
    var unnormalized = GaussianNB(priors=List[Float64]([0.4, 0.4]))
    with assert_raises(contains="sum"):
        _ = unnormalized.fit(data(), labels())
    var negative_alpha = MultinomialNB(alpha=-1.0)
    with assert_raises(contains="alpha"):
        _ = negative_alpha.fit(data(), labels())
    var categorical = CategoricalNB()
    _ = categorical.fit(codes, labels())
    # sklearn raises IndexError for a category beyond the allocated columns.
    with assert_raises(contains="category"):
        _ = categorical.predict(unseen)


def test_documented_deviations() raises:
    # family.md: explicit priors must be nonnegative (sklearn logs them to NaN).
    var negative_prior = MultinomialNB(class_prior=List[Float64]([-0.5, 1.5]))
    with assert_raises(contains="priors"):
        _ = negative_prior.fit(data(), labels())
    # family.md: unknown incremental labels raise (sklearn's label_binarize
    # silently drops the row from the counts).
    var row: List[List[Float64]] = [[1.0, 0, 2]]
    var model = MultinomialNB()
    _ = model.partial_fit(data(), labels(), classes=List[Int]([0, 1]))
    with assert_raises(contains="unseen label"):
        _ = model.partial_fit(row, List[Int]([5]))
    assert_equal(model.class_count_.item(0), 2.0)


def main() raises:
    test_count_models_predict_negative_values()
    test_unnormalized_class_prior_is_logged_verbatim()
    test_partial_fit_after_fit_accumulates()
    test_gaussian_constant_feature_uses_epsilon()
    test_gaussian_zero_variance_propagates_nan()
    test_gaussian_priors_with_absent_class()
    test_parameter_validation_matches_reference()
    test_documented_deviations()
    print("PASS: Naive Bayes reference edge values and documented deviations")
