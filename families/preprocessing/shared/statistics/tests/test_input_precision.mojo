from std.testing import assert_equal, assert_true, assert_raises
from std.math import isnan
from numojo import zeros, Shape, NDArrayStrides
from moxie.preprocessing._moments import nanmean_at, nanmean_std_at, INPUT_PRECISION_API_VERSION

def check[dtype: DType]() raises:
    var x = zeros[dtype](Shape(3, 1))
    x.unsafe_set(0, 100000000)
    x.unsafe_set(1, 100000008)
    x.unsafe_set(2, 100000016)
    var first = nanmean_std_at(x, 0, 0)
    x.unsafe_set(1, 100000016)
    x.unsafe_set(2, 100000008)
    var second = nanmean_std_at(x, -2, 0)
    assert_equal(first.count, 3)
    comptime if dtype == DType.float32:
        assert_equal(first.stddev, Float64(10.32795524597168).cast[dtype]())
        assert_equal(second.stddev, Float64(6.531972885131836).cast[dtype]())
    else:
        assert_equal(first.stddev, second.stddev)
    x.unsafe_set(0, 1)
    x.unsafe_set(1, Float64("nan").cast[dtype]())
    x.unsafe_set(2, 3)
    var missing = nanmean_std_at(x, 0, 0)
    assert_equal(missing.count, 2)
    assert_equal(missing.mean, 2)
    assert_equal(missing.stddev, 1)
    var row = x.view_with_layout(Shape(1, 3), NDArrayStrides(3, -1), 2)
    assert_equal(nanmean_at(row, 1, 0), 2)
    x.unsafe_set(0, Float64("nan").cast[dtype]())
    x.unsafe_set(2, Float64("nan").cast[dtype]())
    var missing_all = nanmean_std_at(x, 0, 0)
    assert_equal(missing_all.count, 0)
    assert_true(isnan(missing_all.mean))
    assert_true(isnan(missing_all.stddev))
    var empty = zeros[dtype](Shape(0, 2))
    assert_true(isnan(nanmean_at(empty, 0, 1)))
    with assert_raises():
        _ = nanmean_at(x, 2, 0)
    with assert_raises():
        _ = nanmean_at(x, 0, 1)
    # Correction intermediates may contain Inf even though public fit rejects it.
    x.unsafe_set(0, Float64("inf").cast[dtype]())
    x.unsafe_set(1, 2)
    x.unsafe_set(2, 3)
    var overflow = nanmean_std_at(x, 0, 0)
    assert_equal(overflow.count, 3)
    assert_equal(overflow.mean, Float64("inf").cast[dtype]())
    assert_true(isnan(overflow.stddev))
    x.unsafe_set(2, Float64("-inf").cast[dtype]())
    assert_true(isnan(nanmean_at(x, 0, 0)))
    # Repeat/zero-stride views are borrowed; no copy is needed.
    x.unsafe_set(0, 7)
    var repeat = x.view_with_layout(Shape(10, 2), NDArrayStrides(0, 0), 0)
    assert_equal(nanmean_at(repeat, 0, 1), 7)
    assert_equal(nanmean_std_at(repeat, 1, 9).stddev, 0)
    var vector = zeros[dtype](Shape(3))
    with assert_raises():
        _ = nanmean_at(vector, 0, 0)

def main() raises:
    assert_equal(INPUT_PRECISION_API_VERSION, 1)
    check[DType.float32]()
    check[DType.float64]()
    print('PASS input-precision API, collision, NaNs, empty reductions and errors')
