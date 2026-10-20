"""Categorical array ownership, layout, validation and category growth."""
import numojo as nm
from numojo import Shape, Item
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.naive_bayes import CategoricalNB
from moxie.naive_bayes import CategoricalNB as ListCategoricalNB
from moxie._naive_bayes_dense import dense_from_lists


def check_dtype[dtype: DType]() raises:
    var data: List[List[Float64]] = [[3.0, 2.0], [1.0, 0.0], [0.0, 0.0], [0.0, 4.0]]
    var X = dense_from_lists(data).astype[dtype]()
    var y: List[Int] = [9, -2, 9, -2]
    var weights: List[Float64] = [0.5, 2.0, 1.0, 3.0]
    var model = CategoricalNB()
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
    assert_equal(model.class_count_.item(0), 5.0)
    assert_equal(model.class_count_.item(1), 1.5)
    assert_equal(model.n_categories_, List[Int]([4, 5]))
    assert_equal(model.category_count_[0].shape[0], 2)
    assert_equal(model.category_count_[0].shape[1], 4)
    assert_equal(model.category_count_[0].item(0, 1), 2.0)
    assert_equal(model.category_count_[1].item(0, 4), 3.0)
    assert_equal(model.category_count_[0].item(1, 3), 0.5)
    var p = model.predict_proba(X)
    var list_p = model.predict_proba(data)
    assert_equal(p.shape[0], 4)
    assert_equal(p.shape[1], 2)
    for i in range(4):
        assert_true(abs(p.item(i, 0) + p.item(i, 1) - 1.0) < 1e-12)
        for c in range(2):
            assert_equal(p.item(i, c), list_p[i][c])
    var copied = model.copy()
    copied.category_count_[0][Item(0, 1)] = 100.0
    copied.feature_log_prob_[0][Item(0, 1)] = 100.0
    copied.class_count_[Item(0)] = 100.0
    assert_equal(model.category_count_[0].item(0, 1), 2.0)
    assert_equal(model.class_count_.item(0), 5.0)
    assert_true(model.feature_log_prob_[0].item(0, 1) < 0)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(len(fresh.category_count_), 0)
    var first = X.view_with_layout(Shape(2, 2), nm.NDArrayStrides(2, 1), 0)
    var second = X.view_with_layout(Shape(2, 2), nm.NDArrayStrides(2, 1), 4)
    _ = fresh.partial_fit(first, List[Int]([9, -2]), classes=List[Int]([9, -2]),
                      sample_weight=List[Float64]([0.5, 2.0]))
    _ = fresh.partial_fit(second, List[Int]([9, -2]), sample_weight=List[Float64]([1.0, 3.0]))
    assert_equal(fresh.n_categories_, List[Int]([1, 5]))
    assert_equal(fresh.category_count_[0].shape[1], 4)
    for f in range(2):
        for c in range(2):
            for k in range(model.category_count_[f].shape[1]):
                assert_equal(fresh.category_count_[f].item(c, k), model.category_count_[f].item(c, k))
                assert_equal(fresh.feature_log_prob_[f].item(c, k), model.feature_log_prob_[f].item(c, k))
    var reversed = X.view_with_layout(Shape(4, 2), nm.NDArrayStrides(-2, -1), 7)
    _ = fresh.fit(reversed, List[Int]([-2, 9, -2, 9]), sample_weight=List[Float64]([3.0, 1.0, 2.0, 0.5]))
    for f in range(2):
        for c in range(2):
            for k in range(model.category_count_[f].shape[1]):
                assert_equal(fresh.category_count_[1 - f].item(c, k), model.category_count_[f].item(c, k))
    X[Item(0, 0)] = 99.0
    assert_equal(model.category_count_[0].shape[1], 4)
    X[Item(0, 0)] = 3.0
    var before = model.predict(X)
    var invalid = X.copy()
    for value in List[Float64]([-1.0, Float64("nan"), Float64("inf"), 1e30]):
        invalid[Item(0, 0)] = SIMD[dtype, 1](value)
        with assert_raises():
            _ = model.partial_fit(invalid, y)
        with assert_raises():
            _ = model.predict(invalid)
    invalid[Item(0, 0)] = 8.0
    with assert_raises():
        _ = model.predict(invalid)
    with assert_raises():
        _ = model.fit(X, List[Int]([1]))
    with assert_raises():
        _ = model.partial_fit(X, List[Int]([1, 1, 1, 1]))
    with assert_raises():
        _ = model.partial_fit(X, y, classes=List[Int]([-2, 9, 10]))
    with assert_raises():
        _ = model.fit(X, y, sample_weight=List[Float64]([-1.0, 1.0, 1.0, 1.0]))
    with assert_raises():
        _ = model.fit(X, y, sample_weight=List[Float64]([0.0, 0.0, 0.0, 0.0]))
    with assert_raises():
        _ = model.predict(nm.zeros[dtype](Shape(1, 3)))
    with assert_raises():
        _ = model.fit(nm.zeros[dtype](Shape(4)), y)
    with assert_raises():
        _ = model.fit(nm.zeros[dtype](Shape(0, 2)), List[Int]())
    with assert_raises():
        _ = model.fit(nm.zeros[dtype](Shape(4, 0)), y)
    assert_equal(model.class_count_.item(0), 5.0)
    assert_equal(model.predict(X), before)
    _ = model.set_params(alpha=-1.0)
    with assert_raises():
        _ = model.partial_fit(X, y)
    assert_equal(model.class_count_.item(0), 5.0)
    _ = model.set_params(alpha=1.0)
    _ = model.fit(second, List[Int]([1, 1]))
    assert_equal(model.class_count_.size, 1)
    assert_equal(model.category_count_[0].shape[1], 1)
    assert_equal(model.class_count_.item(0), 2.0)


def check_conversion[dtype: DType]() raises:
    var values: List[List[Float64]] = [[-0.9, 1.9], [1.9, -0.1]]
    var X = dense_from_lists(values).astype[dtype]()
    var y: List[Int] = [0, 1]
    var model = CategoricalNB(min_categories_values=List[Int]([3]))
    _ = model.fit(X, y)
    assert_equal(model.n_categories_, List[Int]([3, 3]))
    assert_equal(model.category_count_[0].item(0, 0), 1.0)
    assert_equal(model.category_count_[1].item(1, 0), 1.0)
    assert_equal(model.predict(X), y)
    assert_equal(model.predict(values), y)
    _ = model.partial_fit(X, y)
    assert_equal(model.category_count_[0].item(0, 0), 2.0)
    # The same conversion applies to List fitting and to strided arrays.
    _ = model.fit(values, y)
    assert_equal(model.predict(X), y)
    var reversed = X.view_with_layout(Shape(2, 2), nm.NDArrayStrides(-2, 1), 2)
    assert_equal(model.predict(reversed), List[Int]([1, 0]))
    # Conversion does not mutate or retain caller storage.
    assert_equal(X.item(0, 0), SIMD[dtype, 1](-0.9))
    for invalid in List[Float64]([-1.0, -1.9, Float64("nan"), Float64("inf"), 9223372036854775808.0]):
        var bad = X.copy()
        bad[Item(0, 0)] = SIMD[dtype, 1](invalid)
        with assert_raises():
            _ = model.fit(bad, y)
        with assert_raises():
            _ = model.partial_fit(bad, y)
        with assert_raises():
            _ = model.predict(bad)
        assert_equal(model.class_count_.item(0), 1.0)
        assert_equal(model.predict(X), y)


def main() raises:
    # The preserved engine also consumes the shared minimum contract.
    var list_model = ListCategoricalNB(min_categories_values=List[Int]([3]))
    var X: List[List[Float64]] = [[0.0, 1.0], [1.0, 0.0]]
    var y: List[Int] = [0, 1]
    _ = list_model.fit(X, y)
    assert_equal(list_model.n_categories_, List[Int]([3, 3]))
    _ = list_model.partial_fit(X, y)
    assert_equal(list_model.category_count_[1].item(0, 1), 2.0)
    assert_equal(list_model.predict(X), y)
    check_dtype[DType.float32]()
    check_dtype[DType.float64]()
    check_conversion[DType.float32]()
    check_conversion[DType.float64]()
    print("PASS: categorical arrays, growth, strides, ownership and validation")
