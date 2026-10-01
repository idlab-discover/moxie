from moxie._linalg import covariance_factor, covariance_solve, covariance_inverse
from numojo import Shape, Item, NDArray
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from std.math import abs


def check[dtype: DType]() raises:
    var a = nm.eye[dtype](2, 2)
    a[Item(0, 0)] = 2
    a[Item(1, 1)] = 4
    var b = nm.zeros[dtype](Shape(2))
    b[Item(0)] = 6
    b[Item(1)] = 8
    var factor = covariance_factor(a)
    assert_equal(factor.order(), 2)
    var x = factor.solve(b)
    var tolerance = Float64(2e-6) if dtype == DType.float32 else Float64(2e-14)
    assert_true(abs(Float64(x[Item(0)]) - 3) < tolerance)
    assert_true(abs(Float64(x[Item(1)]) - 2) < tolerance)
    assert_equal(x.ndim, 1)
    assert_equal(a[Item(0, 0)], Scalar[dtype](2))
    assert_equal(b[Item(0)], Scalar[dtype](6))
    x[Item(0)] = 99
    assert_equal(b[Item(0)], Scalar[dtype](6))
    var inv = factor.inverse()
    assert_true(abs(Float64(inv[Item(0, 0)]) - 0.5) < tolerance)
    var copied = factor.copy()
    var lower = factor.lower_factor()
    lower[Item(0, 0)] = 100
    a[Item(0, 0)] = 100
    var independent = copied.inverse()
    assert_true(abs(Float64(independent[Item(0, 0)]) - 0.5) < tolerance)
    var again = factor.solve(b)
    assert_true(abs(Float64(again[Item(0)]) - 3) < tolerance)
    # The covariance and RHS may be views of the same backing allocation.
    a[Item(0, 0)] = 2
    var alias_rhs = a.view_with_layout(Shape(2), nm.NDArrayStrides(2), 0)
    var alias_solution = covariance_solve(a, alias_rhs)
    assert_true(abs(Float64(alias_solution[Item(0)]) - 1) < tolerance)
    assert_equal(a[Item(0, 0)], Scalar[dtype](2))
    # Larger backing, negative/positive strides and nonzero offsets.
    var backing = nm.zeros[dtype](Shape(30))
    var view = backing.view_with_layout(Shape(2, 2), nm.NDArrayStrides(-6, 2), 12)
    view[Item(0, 0)] = 2
    view[Item(1, 1)] = 4
    var rhs_backing = nm.zeros[dtype](Shape(12))
    var rhs = rhs_backing.view_with_layout(Shape(2, 1), nm.NDArrayStrides(-3, 1), 7)
    rhs[Item(0, 0)] = 6
    rhs[Item(1, 0)] = 8
    var result = covariance_solve(view, rhs)
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[1], 1)
    assert_true(abs(Float64(result[Item(0, 0)]) - 3) < tolerance)
    assert_equal(rhs[Item(0, 0)], Scalar[dtype](6))
    var empty = nm.zeros[dtype](Shape(0, 0))
    var empty_factor = covariance_factor(empty)
    var empty_inverse = empty_factor.inverse()
    assert_equal(empty_inverse.size, 0)
    var empty_rhs = nm.zeros[dtype](Shape(0))
    var empty_solution = empty_factor.solve(empty_rhs)
    assert_equal(empty_solution.ndim, 1)
    var empty_matrix_rhs = nm.zeros[dtype](Shape(0, 3))
    var empty_matrix_solution = empty_factor.solve(empty_matrix_rhs)
    assert_equal(empty_matrix_solution.shape[1], 3)
    var zero_columns = nm.zeros[dtype](Shape(2, 0))
    var zero_result = factor.solve(zero_columns)
    assert_equal(zero_result.shape[0], 2)
    assert_equal(zero_result.shape[1], 0)
    var bad_shape = nm.zeros[dtype](Shape(2, 3))
    with assert_raises():
        _ = covariance_factor(bad_shape)
    with assert_raises():
        _ = covariance_factor(b)
    var rank3 = nm.zeros[dtype](Shape(2, 1, 1))
    with assert_raises():
        _ = covariance_factor(rank3)
    with assert_raises():
        _ = factor.solve(rank3)
    var wrong_rhs = nm.zeros[dtype](Shape(3))
    with assert_raises():
        _ = factor.solve(wrong_rhs)
    var singular = nm.zeros[dtype](Shape(2, 2))
    for i in range(4):
        singular.unsafe_set(i, 1)
    with assert_raises():
        _ = covariance_factor(singular)
    assert_equal(singular[Item(1, 1)], Scalar[dtype](1))
    singular[Item(0, 1)] = 2
    singular[Item(1, 0)] = 2
    with assert_raises():
        _ = covariance_inverse(singular)
    a[Item(0, 1)] = 0.00001
    with assert_raises():
        _ = covariance_factor(a)
    var infinity = Scalar[dtype](Float64.MAX * 2)
    for bad in [infinity, -infinity, infinity - infinity]:
        a[Item(0, 1)] = bad
        with assert_raises():
            _ = covariance_factor(a)
        b[Item(0)] = bad
        with assert_raises():
            _ = factor.solve(b)
    # A finite input whose solution overflows must fail, preserving RHS.
    a[Item(0, 1)] = 0
    a[Item(0, 0)] = 0.25
    b[Item(0)] = Scalar[dtype].MAX
    with assert_raises():
        _ = covariance_solve(a, b)
    assert_equal(b[Item(0)], Scalar[dtype].MAX)
    print("PASS covariance dtype", dtype)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
