import numojo as nm
from std.math import nan, inf, isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.feature_selection import VarianceThreshold

def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(3, 4))
    var data: List[Int] = [0, 2, 0, 3, 0, 1, 4, 3, 0, 1, 1, 3]
    for i in range(12):
        X.unsafe_set(i, SIMD[dtype, 1](data[i]))
    var selector = VarianceThreshold[dtype]()
    with assert_raises():
        _ = selector.get_support()
    selector.fit_named(X, None, [String("a"), String("b"), String("c"), String("d")])
    assert_equal(selector.n_features_in_(), 4)
    assert_equal(selector.get_feature_names_out()[0], "b")
    assert_equal(selector.get_support(True)[List[Int]][1], 2)
    var selected: nm.NDArray[dtype] = selector.transform(X)
    assert_equal(selected.shape[1], 2)
    assert_equal(selected.unsafe_get(3), 4)
    var restored: nm.NDArray[dtype] = selector.inverse_transform(selected)
    assert_equal(restored.unsafe_get(3), 0)
    assert_equal(restored.unsafe_get(6), 4)
    var variances: nm.NDArray[DType.float64] = selector.variances_()
    variances.unsafe_set(2, -1)
    assert_true(selector.variances_().unsafe_get(2) > 0)
    var duplicate = clone(selector)
    assert_false(duplicate.__sklearn_is_fitted__())
    _ = selector.set_params(threshold=0.4)
    assert_equal(selector.transform(X).shape[1], 1)
    _ = selector.set_params(threshold=100)
    assert_equal(selector.transform(X).shape[1], 0)
    _ = selector.set_params(threshold=0)
    var constant = nm.ones[dtype](nm.Shape(3, 4))
    with assert_raises():
        _ = selector.fit(constant)
    assert_equal(selector.get_support(True)[List[Int]][1], 2)
    X.unsafe_set(0, nan[dtype]())
    X.unsafe_set(4, nan[dtype]())
    X.unsafe_set(8, nan[dtype]())
    X.unsafe_set(1, nan[dtype]())
    _ = selector.fit(X)
    assert_true(isnan(selector.variances_().unsafe_get(0)))
    assert_equal(selector.get_support(True)[List[Int]][0], 2)
    X.unsafe_set(2, nan[dtype]())
    assert_true(isnan(selector.transform(X).unsafe_get(0)))
    with assert_raises():
        _ = selector.inverse_transform(selector.transform(X))
    X.unsafe_set(0, inf[dtype]())
    with assert_raises():
        _ = selector.fit(X)
    with assert_raises():
        _ = selector.transform(X)
    var empty = nm.zeros[dtype](nm.Shape(0, 4))
    with assert_raises():
        _ = selector.fit(empty)
    var single = nm.ones[dtype](nm.Shape(1, 4))
    with assert_raises():
        _ = selector.fit(single)
    var invalid = VarianceThreshold[dtype](-1)
    with assert_raises():
        _ = invalid.fit(constant)
    _ = invalid.set_params(threshold=nan[DType.float64]())
    with assert_raises():
        _ = invalid.fit(constant)
    var two = nm.zeros[dtype](nm.Shape(2, 1))
    two.unsafe_set(1, 1)
    var equality = VarianceThreshold[dtype](0.25)
    with assert_raises():
        _ = equality.fit(two)
    _ = equality.set_params(threshold=0.24)
    assert_equal(equality.fit_transform(two).shape[1], 1)

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("variance selector contracts passed")
