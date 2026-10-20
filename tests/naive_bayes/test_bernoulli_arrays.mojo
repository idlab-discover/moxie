"""Native Bernoulli input, state, and lifecycle contracts."""
import numojo as nm
from numojo import Shape, Item
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.naive_bayes import BernoulliNB, BernoulliNBParams
from moxie._naive_bayes_dense import dense_from_lists


def check_dtype[dtype: DType]() raises:
    var data: List[List[Float64]] = [[0.0, 1.0, 2.0], [-1.0, 0.5, 3.0], [2.0, 0.0, 0.0], [1.0, 0.75, 0.0]]
    var X = dense_from_lists(data).astype[dtype]()
    var y: List[Int] = [9, 9, -2, -2]
    var weights: List[Float64] = [0.5, 2.0, 1.0, 3.0]
    var boundary = nm.zeros[dtype](Shape(1, 1))
    boundary[Item(0, 0)] = 0.1
    var threshold_model = BernoulliNB(binarize=0.1)
    _ = threshold_model.fit(boundary, List[Int]([1]))
    assert_equal(threshold_model.feature_count_.item(0, 0), 0.0)
    var model = BernoulliNB(binarize=0.5)
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.predict_proba(X)
    with assert_raises():
        _ = model.predict_log_proba(X)
    with assert_raises():
        _ = model.predict_joint_log_proba(X)
    with assert_raises():
        _ = model.score(X, y)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.fit(X, y, sample_weight=weights.copy())
    assert_equal(model.classes_, List[Int]([-2, 9]))
    assert_equal(model.class_count_.item(0), 4.0)
    assert_equal(model.class_count_.item(1), 2.5)
    assert_equal(model.feature_count_.item(0, 0), 4.0)
    assert_equal(model.feature_count_.item(0, 1), 3.0)
    assert_equal(model.feature_count_.item(0, 2), 0.0)
    assert_equal(model.feature_count_.item(1, 0), 0.0)
    assert_equal(model.feature_count_.item(1, 1), 0.5)
    assert_equal(model.feature_count_.item(1, 2), 2.5)
    assert_equal(model.score(X, y, weights.copy()), 1.0)
    var probabilities = model.predict_proba(X)
    assert_equal(probabilities.shape[0], 4)
    assert_equal(probabilities.shape[1], 2)
    for i in range(4):
        assert_true(abs(probabilities.item(i, 0) + probabilities.item(i, 1) - 1.0) < 1e-12)
    var as_list = model.predict_proba(data)
    for i in range(4):
        for c in range(2):
            assert_equal(probabilities.item(i, c), as_list[i][c])
    var copied = model.copy()
    copied.feature_count_[Item(0, 0)] = 100.0
    copied.class_count_[Item(0)] = 100.0
    copied.feature_log_prob_[Item(0, 0)] = 100.0
    assert_equal(model.feature_count_.item(0, 0), 4.0)
    assert_equal(model.class_count_.item(0), 4.0)
    assert_true(model.feature_log_prob_.item(0, 0) < 0)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.feature_count_.size, 0)
    var first = X.view_with_layout(Shape(2, 3), nm.NDArrayStrides(3, 1), 0)
    var second = X.view_with_layout(Shape(2, 3), nm.NDArrayStrides(3, 1), 6)
    _ = fresh.partial_fit(first, List[Int]([9, 9]), classes=List[Int]([9, -2]),
                      sample_weight=List[Float64]([0.5, 2.0]))
    _ = fresh.partial_fit(second, List[Int]([-2, -2]), sample_weight=List[Float64]([1.0, 3.0]))
    for c in range(2):
        for f in range(3):
            assert_equal(fresh.feature_count_.item(c, f), model.feature_count_.item(c, f))
            assert_equal(fresh.feature_log_prob_.item(c, f), model.feature_log_prob_.item(c, f))
    # Noncontiguous input with negative strides and an offset.
    var reversed = X.view_with_layout(Shape(4, 3), nm.NDArrayStrides(-3, -1), 11)
    _ = fresh.fit(reversed, List[Int]([-2, -2, 9, 9]), sample_weight=List[Float64]([3.0, 1.0, 2.0, 0.5]))
    for c in range(2):
        for f in range(3):
            assert_equal(fresh.feature_count_.item(c, f), model.feature_count_.item(c, 2 - f))
    assert_equal(fresh.predict(reversed), List[Int]([-2, -2, 9, 9]))
    X[Item(0, 0)] = 99.0
    assert_equal(fresh.feature_count_.item(1, 2), 0.0)
    X[Item(0, 0)] = 0.0
    # Invalid calls preserve all fitted counts and predictions.
    with assert_raises():
        _ = model.partial_fit(X, List[Int]([1, 1, 1, 1]))
    with assert_raises():
        _ = model.partial_fit(X, y, classes=List[Int]([-2, 9, 10]))
    with assert_raises():
        _ = model.fit(X, List[Int]([1]))
    with assert_raises():
        _ = model.fit(X, y, sample_weight=List[Float64]([0.0, 0.0, 0.0, 0.0]))
    with assert_raises():
        _ = model.fit(X, y, sample_weight=List[Float64]([-1.0, 1.0, 1.0, 1.0]))
    with assert_raises():
        _ = model.score(X, List[Int]([9]))
    var invalid = X.copy()
    invalid[Item(0, 0)] = SIMD[dtype, 1](Float64("nan"))
    with assert_raises():
        _ = model.fit(invalid, y)
    invalid[Item(0, 0)] = SIMD[dtype, 1](Float64("inf"))
    with assert_raises():
        _ = model.predict(invalid)
    with assert_raises():
        _ = model.predict(nm.zeros[dtype](Shape(1, 2)))
    with assert_raises():
        _ = model.fit(nm.zeros[dtype](Shape(4)), y)
    with assert_raises():
        _ = model.partial_fit(nm.zeros[dtype](Shape(0, 3)), List[Int]())
    with assert_raises():
        _ = model.fit(nm.zeros[dtype](Shape(4, 0)), y)
    assert_equal(model.class_count_.item(0), 4.0)
    assert_equal(model.score(X, y), 1.0)
    _ = model.set_params(alpha=-1.0)
    with assert_raises():
        _ = model.partial_fit(X, y)
    assert_equal(model.class_count_.item(0), 4.0)
    _ = model.set_params(alpha=1.0)
    _ = model.fit(X, List[Int]([1, 1, 1, 1]))
    assert_equal(model.class_count_.size, 1)
    assert_equal(model.class_count_.item(0), 4.0)



def check_vector_configuration() raises:
    var values = nm.full[DType.float64](Shape(3), 0.5)
    var model = BernoulliNB(alpha_values=values.view())
    values[Item(0)] = 99.0
    assert_equal(model.alpha_values.value().item(0), 0.5)
    var params = model.get_params()
    params.alpha_values.value()[Item(0)] = 2.0
    assert_equal(model.alpha_values.value().item(0), 0.5)
    _ = model.set_params(params)
    params.alpha_values.value()[Item(0)] = 3.0
    assert_equal(model.alpha_values.value().item(0), 2.0)
    var fresh = clone(model)
    fresh.alpha_values.value()[Item(0)] = 4.0
    assert_equal(model.alpha_values.value().item(0), 2.0)
    assert_false(fresh.__sklearn_is_fitted__())
    var copied = model.copy()
    copied.alpha_values.value()[Item(0)] = 5.0
    assert_equal(model.alpha_values.value().item(0), 2.0)
    _ = model.set_params(alpha=0.25)
    assert_false(Bool(model.alpha_values))
    _ = model.set_params(alpha_values=Optional[Optional[nm.NDArray[DType.float64]]](values.copy()))
    values[Item(0)] = 100.0
    assert_equal(model.alpha_values.value().item(0), 99.0)
    _ = model.set_params(alpha_values=Optional[Optional[nm.NDArray[DType.float64]]](Optional[nm.NDArray[DType.float64]](None)))
    assert_false(Bool(model.alpha_values))
    assert_equal(model.alpha, 0.25)
    # Preserve the original five positional fields of the public Params record.
    _ = model.set_params(BernoulliNBParams(1.0, True, True, None, 0.0))
    assert_false(Bool(model.alpha_values))


def check_vector_fit[dtype: DType]() raises:
    var data: List[List[Float64]] = [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    var X = dense_from_lists(data).astype[dtype]()
    var y: List[Int] = [9, 9, -2, -2]
    var weights: List[Float64] = [0.5, 2.0, 1.0, 3.0]
    var values = nm.zeros[DType.float64](Shape(3))
    values[Item(0)] = 0.25
    values[Item(1)] = 0.5
    values[Item(2)] = 2.0
    # Vector precedence includes ignoring an invalid inactive scalar.
    var model = BernoulliNB(alpha=-1.0, alpha_values=values.copy())
    _ = model.fit(X, y, sample_weight=weights.copy())
    for f in range(3):
        var scalar = BernoulliNB(alpha=values.item(f))
        _ = scalar.fit(X, y, sample_weight=weights.copy())
        for c in range(2):
            assert_equal(model.feature_log_prob_.item(c, f), scalar.feature_log_prob_.item(c, f))
    var fresh = clone(model)
    var first = X.view_with_layout(Shape(2, 3), nm.NDArrayStrides(3, 1), 0)
    var second = X.view_with_layout(Shape(2, 3), nm.NDArrayStrides(3, 1), 6)
    _ = fresh.partial_fit(first, List[Int]([9, 9]), classes=List[Int]([-2, 9]), sample_weight=List[Float64]([0.5, 2.0]))
    _ = fresh.partial_fit(second, List[Int]([-2, -2]), sample_weight=List[Float64]([1.0, 3.0]))
    var lists = clone(model)
    _ = lists.fit(data, y, sample_weight=weights.copy())
    for c in range(2):
        for f in range(3):
            assert_equal(fresh.feature_log_prob_.item(c, f), model.feature_log_prob_.item(c, f))
            assert_equal(lists.feature_log_prob_.item(c, f), model.feature_log_prob_.item(c, f))
    var before = model.predict_proba(X)
    # Shape, finite-value and nonnegative validation is transactional.
    var bad_values = List[nm.NDArray[DType.float64]]()
    bad_values.append(nm.zeros[DType.float64](Shape(0)))
    bad_values.append(nm.zeros[DType.float64](Shape(2)))
    bad_values.append(nm.zeros[DType.float64](Shape(1, 3)))
    for bad in List[Float64]([-1.0, Float64("nan"), Float64("inf")]):
        var invalid = values.copy()
        invalid[Item(1)] = bad
        bad_values.append(invalid^)
    for invalid in bad_values:
        _ = model.set_params(alpha_values=Optional[Optional[nm.NDArray[DType.float64]]](invalid.copy()))
        with assert_raises():
            _ = model.partial_fit(X, y)
        with assert_raises():
            _ = model.fit(X, y)
        assert_equal(model.class_count_.item(0), 4.0)
        var after = model.predict_proba(X)
        for i in range(4):
            for c in range(2):
                assert_equal(after.item(i, c), before.item(i, c))
    # A strided configuration is copied by logical feature order.
    var reverse = values.view_with_layout(Shape(3), nm.NDArrayStrides(-1), 2)
    _ = model.set_params(alpha_values=Optional[Optional[nm.NDArray[DType.float64]]](reverse.copy()))
    _ = model.fit(X, y, sample_weight=weights.copy())
    for f in range(3):
        var scalar = BernoulliNB(alpha=values.item(2 - f))
        _ = scalar.fit(X, y, sample_weight=weights.copy())
        for c in range(2):
            assert_equal(model.feature_log_prob_.item(c, f), scalar.feature_log_prob_.item(c, f))
    # Elementwise flooring leaves constructor configuration unchanged.
    values[Item(0)] = 0.0
    values[Item(1)] = 1e-12
    _ = model.set_params(alpha_values=Optional[Optional[nm.NDArray[DType.float64]]](values.copy()), force_alpha=False)
    _ = model.fit(X, y)
    for f in range(3):
        var scalar = BernoulliNB(alpha=values.item(f), force_alpha=False)
        _ = scalar.fit(X, y)
        for c in range(2):
            assert_equal(model.feature_log_prob_.item(c, f), scalar.feature_log_prob_.item(c, f))
    assert_equal(model.alpha_values.value().item(0), 0.0)
    assert_equal(model.alpha_values.value().item(1), 1e-12)
    _ = model.set_params(alpha=0.5)
    _ = model.fit(X, y)
    assert_false(Bool(model.alpha_values))


def main() raises:
    check_vector_fit[DType.float32]()
    check_vector_fit[DType.float64]()
    check_vector_configuration()
    check_dtype[DType.float32]()
    check_dtype[DType.float64]()
    var X = dense_from_lists([[0.0, 1.0], [1.0, 0.0]])
    var model = BernoulliNB(alpha=0.0, binarize=None)
    _ = model.fit(X, List[Int]([1, 2]))
    var probabilities = model.predict_proba(X)
    assert_true(isnan(probabilities.item(0, 0)))
    assert_true(isnan(probabilities.item(0, 1)))
    print("PASS: Bernoulli Float32/Float64 arrays, strides, ownership and lifecycle")
