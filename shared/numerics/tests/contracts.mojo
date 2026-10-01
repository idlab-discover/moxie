"""Candidate behavior contracts; these are not full NumPy compatibility."""
import numojo as nm
from numojo.prelude import Shape, Item
from numojo.routines.linalg import dot, solve
from std.testing import assert_equal, assert_true, assert_raises


def escaped_view[dtype: DType]() raises -> nm.NDArray[dtype]:
    var owner = nm.arange[dtype](6)
    return owner.view()


def contracts[dtype: DType]() raises:
    var a = nm.arange[dtype](6).reshape(Shape(2, 3))
    var copied = a.copy()
    copied[Item(0, 0)] = 20
    assert_equal(a.item(0, 0), 0)
    var shared = a.view()
    shared[Item(0, 1)] = 10
    assert_equal(a.item(0, 1), 10)
    a[Item(0, 1)] = 1
    var sliced = a[0:2, 1:3]
    assert_equal(sliced.item(1, 0), 4)
    assert_equal(nm.sum(sliced), 12)
    var columns = nm.sum(a, axis=0)
    assert_equal(columns.item(0), 3)
    assert_equal(columns.item(2), 7)
    var centered = a - nm.mean[returned_dtype=dtype](a, axis=0)
    assert_equal(nm.sum(centered * centered), 13.5)
    var product = a @ nm.eye[dtype](3, 3)
    assert_equal(product.item(1, 2), 5)
    var reshaped = a.reshape(Shape(6))
    reshaped[Item(0)] = 100
    assert_equal(a.item(0, 0), 0)
    var surviving = escaped_view[dtype]()
    assert_equal(surviving.item(5), 5)
    var x = nm.arange[dtype](1, 4)
    var vector_product = x @ x
    assert_equal(vector_product.ndim, 2)
    assert_equal(vector_product.shape[0], 1)
    assert_equal(vector_product.shape[1], 1)
    assert_equal(vector_product.item(0, 0), 14)
    with assert_raises():
        _ = a @ nm.ones[dtype](Shape(2, 2))
    with assert_raises():
        _ = a + nm.ones[dtype](Shape(2, 2))
    var broken_dot = dot(x, x)
    assert_equal(broken_dot.ndim, 1)
    assert_equal(broken_dot.size, 3)
    assert_equal(broken_dot.item(2), 9)
    with assert_raises():
        _ = nm.sum(x, axis=0)
    with assert_raises():
        _ = nm.sum(a, axis=2)
    with assert_raises():
        _ = a.reshape(Shape(5))
    var pivot = nm.zeros[dtype](Shape(2, 2))
    pivot[Item(0, 1)] = 1
    pivot[Item(1, 0)] = 1
    pivot[Item(1, 1)] = 1
    var rhs = nm.ones[dtype](Shape(2, 1))
    rhs[Item(1, 0)] = 2
    var solution = solve(pivot, rhs)
    assert_true(solution.item(0) != solution.item(0))


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("PASS: candidate ownership, lifetime, shape and known-gap contracts")
