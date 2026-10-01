"""Numerical edge and layout observations, emitted for the Python oracle."""
import numojo as nm
from moxie._dense import sum_axis2d, row_squared_norms
from std.testing import assert_equal, assert_raises, assert_true
from std.sys import argv
from std.math import isfinite


def edges[dtype: DType]() raises:
    var a = nm.arange[dtype](12).reshape(nm.Shape(3, 4))
    var sliced = a[0:3, 0:4:2]
    sliced[nm.Item(0, 0)] = 99
    assert_equal(a.item(0, 0), 0)
    var transposed = a.T()
    transposed[nm.Item(0, 0)] = 88
    assert_equal(a.item(0, 0), 0)
    var wide = a.astype[DType.float64]()
    wide[nm.Item(0, 0)] = 77
    assert_equal(a.item(0, 0), 0)
    var strided = a.view_with_layout(nm.Shape(3, 2), nm.NDArrayStrides(4, 2), 0)
    assert_true(not strided.is_c_contiguous())
    strided[nm.Item(1, 1)] = 42
    assert_equal(a.item(1, 2), 42)
    a[nm.Item(1, 2)] = 6
    var independent = strided.copy()
    independent[nm.Item(1, 1)] = 43
    assert_equal(a.item(1, 2), 6)
    var negative = sum_axis2d(a, -1)
    assert_equal(negative.item(0), 6)
    with assert_raises():
        _ = sum_axis2d(a, 2)
    with assert_raises():
        _ = row_squared_norms(nm.arange[dtype](3))
    with assert_raises():
        _ = nm.variance(a, axis=1, ddof=3)
    var empty = nm.zeros[dtype](nm.Shape(0, 3))
    assert_equal(nm.sum(empty), 0)
    var sums = sum_axis2d(empty, 0)
    assert_equal(sums.size, 3)
    assert_equal(nm.sum(sums), 0)
    var no_columns = nm.zeros[dtype](nm.Shape(3, 0))
    var norms = row_squared_norms(no_columns)
    assert_equal(nm.sum(norms), 0)
    var large = nm.zeros[dtype](nm.Shape(1, 3))
    large[nm.Item(0, 0)] = SIMD[dtype, 1](1e8)
    large[nm.Item(0, 1)] = 1
    large[nm.Item(0, 2)] = SIMD[dtype, 1](-1e8)
    var accurate = sum_axis2d(large, 1)
    assert_equal(accurate.item(0), 1)
    var zero = SIMD[dtype, 1](0)
    large[nm.Item(0, 0)] = zero / zero
    var nans = row_squared_norms(large)
    assert_true(nans.item(0) != nans.item(0))
    large[nm.Item(0, 0)] = SIMD[dtype, 1](1) / zero
    var infs = row_squared_norms(large)
    assert_true(not isfinite(infs.item(0)))
    print("PASS edges")


def main() raises:
    var args = argv()
    if len(args) == 2 and args[1] == "singleton":
        _ = nm.sum(nm.ones[DType.float64](nm.Shape(1, 1)), axis=0)
    else:
        edges[DType.float32]()
        edges[DType.float64]()
