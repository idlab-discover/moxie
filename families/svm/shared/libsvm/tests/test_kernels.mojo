import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.svm._contracts import KernelSpec
from moxie.svm._kernels import kernel_matrix, resolve_gamma, power


def main() raises:
    var x = nm.arange[DType.float64](6).reshape(nm.Shape(3, 2))
    var linear = kernel_matrix(x, x, KernelSpec("linear", 0.5, 3, 0))
    assert_equal(linear.shape[0], 3)
    assert_equal(linear.shape[1], 3)
    assert_equal(linear.item(0, 2), 5.0)
    assert_equal(linear.item(2, 2), 41.0)
    assert_equal(resolve_gamma(x, "auto"), 0.5)
    assert_true(abs(resolve_gamma(x, "scale") - 6.0 / 35.0) < 1e-14)
    var constant = nm.ones[DType.float64](nm.Shape(1, 1))
    assert_equal(resolve_gamma(constant, "scale"), 1.0)
    assert_equal(power(-2, 3), -8.0)
    assert_equal(power(0, 0), 1.0)
    var rbf = kernel_matrix(x, x, KernelSpec("rbf", 0.5, 3, 0))
    for i in range(3):
        assert_equal(rbf.item(i, i), 1.0)
    var poly = kernel_matrix(x, x, KernelSpec("poly", 0, 0, 0))
    assert_equal(poly.item(1, 2), 1.0)
    var gram = kernel_matrix(linear, x, KernelSpec("precomputed", 0.5, 3, 0))
    gram.unsafe_set(0, 99)
    assert_equal(linear.unsafe_get(0), 1.0)
    with assert_raises():
        _ = kernel_matrix(x, linear, KernelSpec("precomputed", 0.5, 3, 0))
    with assert_raises():
        _ = kernel_matrix(x, constant, KernelSpec("linear", 0.5, 3, 0))
    with assert_raises():
        _ = kernel_matrix(x, x, KernelSpec("callable", 0.5, 3, 0))
    with assert_raises():
        _ = kernel_matrix(x, x, KernelSpec("unknown", 0.5, 3, 0))
    with assert_raises():
        _ = resolve_gamma(x, "unknown")
    with assert_raises():
        _ = kernel_matrix(x, x, KernelSpec("rbf", -1, 3, 0))
    var empty = nm.zeros[DType.float64](nm.Shape(0, 2))
    with assert_raises():
        _ = kernel_matrix(empty, x, KernelSpec("linear", 0.5, 3, 0))
    var strided = x.view_with_layout(nm.Shape(3, 2), nm.NDArrayStrides(2, -1), 1)
    var reverse = kernel_matrix(strided, strided, KernelSpec("linear", 0.5, 3, 0))
    for i in range(9):
        assert_equal(reverse.unsafe_get(i), linear.unsafe_get(i))
    assert_true(abs(resolve_gamma(strided, "scale") - 6.0 / 35.0) < 1e-14)
    x.unsafe_set(0, Float64("nan"))
    with assert_raises():
        _ = kernel_matrix(x, x, KernelSpec("linear", 0.5, 3, 0))
    print("PASS: dense kernel contracts")
