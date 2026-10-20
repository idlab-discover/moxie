"""Dense dtype, scalar-weight, storage and transactional API contracts."""
import numojo as nm
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true, assert_raises
from moxie._naive_bayes_dense import validate_matrix, owned_snapshot
from moxie.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB


@no_inline
def check_dtype[dtype: DType]() raises:
    var X = nm.zeros[dtype](Shape(4, 2))
    X[Item(0, 1)] = 1
    X[Item(1, 0)] = 1
    X[Item(2, 0)] = 1
    X[Item(3, 1)] = 1
    var y: List[Int] = [9, 9, -2, -2]
    var weights: List[Float64] = [2.5, 2.5, 2.5, 2.5]
    var strided = X.view_with_layout(Shape(4, 2), nm.NDArrayStrides(-2, -1), 7)
    var snapshot = owned_snapshot(strided)
    X[Item(0, 0)] = 1
    assert_equal(snapshot.item(3, 1), SIMD[dtype, 1](0))
    X[Item(0, 0)] = 0
    var copied = strided.copy()
    var rank_one = nm.zeros[dtype](Shape(2))
    var empty = nm.zeros[dtype](Shape(0, 2))
    with assert_raises():
        _ = validate_matrix(rank_one)
    with assert_raises():
        _ = validate_matrix(empty)

    var gaussian = GaussianNB()
    var gaussian_vector = GaussianNB()
    _ = gaussian.fit(strided, y, sample_weight=2.5)
    _ = gaussian_vector.fit(copied, y, sample_weight=weights.copy())
    assert_equal(gaussian.class_count_.item(0), 5.0)
    assert_equal(gaussian.predict(strided), gaussian_vector.predict(copied))
    assert_equal(gaussian.score(strided, y), gaussian_vector.score(copied, y))
    _ = gaussian.predict_proba(strided)
    _ = gaussian.predict_log_proba(strided)
    _ = gaussian.predict_joint_log_proba(strided)
    _ = gaussian.partial_fit(strided, y, sample_weight=0.5)
    assert_equal(gaussian.class_count_.item(0), 6.0)
    var gaussian_incremental = GaussianNB()
    _ = gaussian_incremental.partial_fit(strided, y, classes=y.copy(), sample_weight=2.5)
    assert_equal(gaussian_incremental.class_count_.item(0), 5.0)
    for invalid in List[Float64]([0.0, -1.0, Float64("nan"), Float64("inf"), 1e308]):
        with assert_raises():
            _ = gaussian.fit(strided, y, sample_weight=invalid)
        with assert_raises():
            _ = gaussian.partial_fit(strided, y, sample_weight=invalid)
        assert_equal(gaussian.class_count_.item(0), 6.0)
    with assert_raises():
        _ = gaussian.fit(rank_one, y, sample_weight=1.0)
    with assert_raises():
        _ = gaussian.fit(empty, y, sample_weight=1.0)
    _ = gaussian.fit(strided, y, sample_weight=1.0)
    assert_equal(gaussian.class_count_.item(0), 2.0)

    var multinomial = MultinomialNB()
    var multinomial_vector = MultinomialNB()
    _ = multinomial.fit(strided, y, sample_weight=2.5)
    _ = multinomial_vector.fit(copied, y, sample_weight=weights.copy())
    assert_equal(multinomial.class_count_.item(0), 5.0)
    assert_equal(multinomial.predict(strided), multinomial_vector.predict(copied))
    assert_equal(multinomial.score(strided, y), multinomial_vector.score(copied, y))
    _ = multinomial.predict_proba(strided)
    _ = multinomial.predict_log_proba(strided)
    _ = multinomial.predict_joint_log_proba(strided)
    _ = multinomial.partial_fit(strided, y, sample_weight=0.5)
    assert_equal(multinomial.class_count_.item(0), 6.0)
    var multinomial_incremental = MultinomialNB()
    _ = multinomial_incremental.partial_fit(strided, y, classes=y.copy(), sample_weight=2.5)
    assert_equal(multinomial_incremental.class_count_.item(0), 5.0)
    for invalid in List[Float64]([0.0, -1.0, Float64("nan"), Float64("inf"), 1e308]):
        with assert_raises():
            _ = multinomial.fit(strided, y, sample_weight=invalid)
        with assert_raises():
            _ = multinomial.partial_fit(strided, y, sample_weight=invalid)
        assert_equal(multinomial.class_count_.item(0), 6.0)
    with assert_raises():
        _ = multinomial.fit(rank_one, y, sample_weight=1.0)
    with assert_raises():
        _ = multinomial.fit(empty, y, sample_weight=1.0)
    _ = multinomial.fit(strided, y, sample_weight=1.0)
    assert_equal(multinomial.class_count_.item(0), 2.0)

    var complement = ComplementNB()
    var complement_vector = ComplementNB()
    _ = complement.fit(strided, y, sample_weight=2.5)
    _ = complement_vector.fit(copied, y, sample_weight=weights.copy())
    assert_equal(complement.class_count_.item(0), 5.0)
    assert_equal(complement.predict(strided), complement_vector.predict(copied))
    assert_equal(complement.score(strided, y), complement_vector.score(copied, y))
    _ = complement.predict_proba(strided)
    _ = complement.predict_log_proba(strided)
    _ = complement.predict_joint_log_proba(strided)
    _ = complement.partial_fit(strided, y, sample_weight=0.5)
    assert_equal(complement.class_count_.item(0), 6.0)
    var complement_incremental = ComplementNB()
    _ = complement_incremental.partial_fit(strided, y, classes=y.copy(), sample_weight=2.5)
    assert_equal(complement_incremental.class_count_.item(0), 5.0)
    for invalid in List[Float64]([0.0, -1.0, Float64("nan"), Float64("inf"), 1e308]):
        with assert_raises():
            _ = complement.fit(strided, y, sample_weight=invalid)
        with assert_raises():
            _ = complement.partial_fit(strided, y, sample_weight=invalid)
        assert_equal(complement.class_count_.item(0), 6.0)
    with assert_raises():
        _ = complement.fit(rank_one, y, sample_weight=1.0)
    with assert_raises():
        _ = complement.fit(empty, y, sample_weight=1.0)
    _ = complement.fit(strided, y, sample_weight=1.0)
    assert_equal(complement.class_count_.item(0), 2.0)

    var bernoulli = BernoulliNB()
    var bernoulli_vector = BernoulliNB()
    _ = bernoulli.fit(strided, y, sample_weight=2.5)
    _ = bernoulli_vector.fit(copied, y, sample_weight=weights.copy())
    assert_equal(bernoulli.class_count_.item(0), 5.0)
    assert_equal(bernoulli.predict(strided), bernoulli_vector.predict(copied))
    assert_equal(bernoulli.score(strided, y), bernoulli_vector.score(copied, y))
    _ = bernoulli.predict_proba(strided)
    _ = bernoulli.predict_log_proba(strided)
    _ = bernoulli.predict_joint_log_proba(strided)
    _ = bernoulli.partial_fit(strided, y, sample_weight=0.5)
    assert_equal(bernoulli.class_count_.item(0), 6.0)
    var bernoulli_incremental = BernoulliNB()
    _ = bernoulli_incremental.partial_fit(strided, y, classes=y.copy(), sample_weight=2.5)
    assert_equal(bernoulli_incremental.class_count_.item(0), 5.0)
    for invalid in List[Float64]([0.0, -1.0, Float64("nan"), Float64("inf"), 1e308]):
        with assert_raises():
            _ = bernoulli.fit(strided, y, sample_weight=invalid)
        with assert_raises():
            _ = bernoulli.partial_fit(strided, y, sample_weight=invalid)
        assert_equal(bernoulli.class_count_.item(0), 6.0)
    with assert_raises():
        _ = bernoulli.fit(rank_one, y, sample_weight=1.0)
    with assert_raises():
        _ = bernoulli.fit(empty, y, sample_weight=1.0)
    _ = bernoulli.fit(strided, y, sample_weight=1.0)
    assert_equal(bernoulli.class_count_.item(0), 2.0)

    var categorical = CategoricalNB()
    var categorical_vector = CategoricalNB()
    _ = categorical.fit(strided, y, sample_weight=2.5)
    _ = categorical_vector.fit(copied, y, sample_weight=weights.copy())
    assert_equal(categorical.class_count_.item(0), 5.0)
    assert_equal(categorical.predict(strided), categorical_vector.predict(copied))
    assert_equal(categorical.score(strided, y), categorical_vector.score(copied, y))
    _ = categorical.predict_proba(strided)
    _ = categorical.predict_log_proba(strided)
    _ = categorical.predict_joint_log_proba(strided)
    _ = categorical.partial_fit(strided, y, sample_weight=0.5)
    assert_equal(categorical.class_count_.item(0), 6.0)
    var categorical_incremental = CategoricalNB()
    _ = categorical_incremental.partial_fit(strided, y, classes=y.copy(), sample_weight=2.5)
    assert_equal(categorical_incremental.class_count_.item(0), 5.0)
    for invalid in List[Float64]([0.0, -1.0, Float64("nan"), Float64("inf"), 1e308]):
        with assert_raises():
            _ = categorical.fit(strided, y, sample_weight=invalid)
        with assert_raises():
            _ = categorical.partial_fit(strided, y, sample_weight=invalid)
        assert_equal(categorical.class_count_.item(0), 6.0)
    with assert_raises():
        _ = categorical.fit(rank_one, y, sample_weight=1.0)
    with assert_raises():
        _ = categorical.fit(empty, y, sample_weight=1.0)
    _ = categorical.fit(strided, y, sample_weight=1.0)
    assert_equal(categorical.class_count_.item(0), 2.0)

    # Integer/bool thresholds promote to float, including outside the dtype range.
    for threshold in List[Float64]([0.5, 1.5, 256.5]):
        var binary = BernoulliNB(binarize=threshold)
        _ = binary.fit(X, y)
        var expected = Float64(0)
        if threshold < 1:
            expected = 1
        assert_equal(binary.feature_count_.item(0, 0), expected)


def check_lists() raises:
    var X: List[List[Float64]] = [[0.0, 1.0], [1.0, 0.0]]
    var y: List[Int] = [9, -2]
    var gaussian = GaussianNB()
    _ = gaussian.fit(X, y, sample_weight=2.0)
    assert_equal(gaussian.class_count_.item(0), 2.0)
    _ = gaussian.partial_fit(X, y, sample_weight=3.0)
    assert_equal(gaussian.class_count_.item(0), 5.0)
    var gaussian_fresh = GaussianNB()
    _ = gaussian_fresh.partial_fit(X, y, classes=y.copy(), sample_weight=2.0)
    assert_equal(gaussian_fresh.class_count_.item(0), 2.0)
    var multinomial = MultinomialNB()
    _ = multinomial.fit(X, y, sample_weight=2.0)
    assert_equal(multinomial.class_count_.item(0), 2.0)
    _ = multinomial.partial_fit(X, y, sample_weight=3.0)
    assert_equal(multinomial.class_count_.item(0), 5.0)
    var multinomial_fresh = MultinomialNB()
    _ = multinomial_fresh.partial_fit(X, y, classes=y.copy(), sample_weight=2.0)
    assert_equal(multinomial_fresh.class_count_.item(0), 2.0)
    var complement = ComplementNB()
    _ = complement.fit(X, y, sample_weight=2.0)
    assert_equal(complement.class_count_.item(0), 2.0)
    _ = complement.partial_fit(X, y, sample_weight=3.0)
    assert_equal(complement.class_count_.item(0), 5.0)
    var complement_fresh = ComplementNB()
    _ = complement_fresh.partial_fit(X, y, classes=y.copy(), sample_weight=2.0)
    assert_equal(complement_fresh.class_count_.item(0), 2.0)
    var bernoulli = BernoulliNB()
    _ = bernoulli.fit(X, y, sample_weight=2.0)
    assert_equal(bernoulli.class_count_.item(0), 2.0)
    _ = bernoulli.partial_fit(X, y, sample_weight=3.0)
    assert_equal(bernoulli.class_count_.item(0), 5.0)
    var bernoulli_fresh = BernoulliNB()
    _ = bernoulli_fresh.partial_fit(X, y, classes=y.copy(), sample_weight=2.0)
    assert_equal(bernoulli_fresh.class_count_.item(0), 2.0)
    var categorical = CategoricalNB()
    _ = categorical.fit(X, y, sample_weight=2.0)
    assert_equal(categorical.class_count_.item(0), 2.0)
    _ = categorical.partial_fit(X, y, sample_weight=3.0)
    assert_equal(categorical.class_count_.item(0), 5.0)
    var categorical_fresh = CategoricalNB()
    _ = categorical_fresh.partial_fit(X, y, classes=y.copy(), sample_weight=2.0)
    assert_equal(categorical_fresh.class_count_.item(0), 2.0)


def main() raises:
    comptime dtypes = (DType.bool, DType.int8, DType.int16, DType.int32,
                          DType.int64, DType.uint8, DType.uint16, DType.uint32,
                          DType.uint64, DType.float32, DType.float64)
    comptime for i in range(11):
        check_dtype[dtypes[i]]()
    check_lists()
    var negative = nm.zeros[DType.int64](Shape(1, 1))
    negative[Item(0, 0)] = -1
    with assert_raises():
        _ = validate_matrix(negative, nonnegative=True)
    print("PASS: 11 feature dtypes, scalar weights, strides and rollback")
