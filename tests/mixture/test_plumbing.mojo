from numojo import zeros, arange, Shape, Item, NDArray, NDArrayStrides
from moxie.mixture._contracts import covariance_shape, check_shape, check_input, snapshot, EMControl
from std.testing import assert_equal, assert_true, assert_raises

def contracts[dtype: DType]() raises:
    var x = arange[dtype](12).reshape(Shape(3, 4))
    check_input(x, 3)
    check_input(x, 1, False, 4)
    for kind in [String("full"), String("tied"), String("diag"), String("spherical")]:
        var shape = covariance_shape(kind, 2, 4)
        var a = zeros[dtype](shape)
        check_shape(a, shape)
        assert_equal(a.size, 32 if kind == "full" else (16 if kind == "tied" else (8 if kind == "diag" else 2)))
    var copy = snapshot(x)
    copy[Item(0, 0)] = 77
    assert_equal(x.item(0, 0), 0)
    var view = x.view()
    view[Item(0, 0)] = 11
    assert_equal(x.item(0, 0), 11)
    var strided = x.view_with_layout(Shape(3, 2), NDArrayStrides(4, -2), 3)
    check_input(strided)
    var saved = snapshot(strided)
    strided[Item(0, 0)] = 42
    assert_equal(saved.item(0, 0), 3)
    with assert_raises():
        check_shape(x, Shape(12))
    with assert_raises():
        check_shape(x, Shape(4, 3))
    with assert_raises():
        check_input(x, 4)
    with assert_raises():
        check_input(x, 1, False, 3)
    with assert_raises():
        check_input(zeros[dtype](Shape(1, 4)))
    check_input(zeros[dtype](Shape(1, 4)), 1, False, 4)
    with assert_raises():
        check_input(zeros[dtype](Shape(0, 4)), 1, False)
    with assert_raises():
        check_input(zeros[dtype](Shape(2, 0)))
    with assert_raises():
        check_input(zeros[dtype](Shape(3)))
    var zero = SIMD[dtype, 1](0)
    x[Item(0, 0)] = 1 / zero
    with assert_raises():
        check_input(x)
    x[Item(0, 0)] = zero / zero
    with assert_raises():
        check_input(x)

def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    var c = EMControl()
    c.validate()
    assert_equal(c.max_iter, 100)
    assert_equal(c.n_components, 1)
    assert_equal(c.covariance_type, "full")
    assert_equal(c.tol, 1e-3)
    assert_equal(c.reg_covar, 1e-6)
    c.n_init = 4
    assert_equal(c.restart_count(True), 4)
    c.warm_start = True
    assert_equal(c.restart_count(True), 1)
    assert_equal(c.restart_count(False), 4)
    var cloned = c.copy()
    cloned.n_init = 2
    assert_equal(c.n_init, 4)
    c.max_iter = 0
    c.validate()
    for method in [String("kmeans"), String("k-means++"), String("random"), String("random_from_data")]:
        c.init_params = method.copy()
        c.validate()
    c.init_params = "unknown"
    with assert_raises():
        c.validate()
    c.init_params = "kmeans"
    c.n_init = 0
    with assert_raises():
        c.validate()
    c.n_init = 1
    c.max_iter = -1
    with assert_raises():
        c.validate()
    c.max_iter = 100
    c.tol = -1
    with assert_raises():
        c.validate()
    c.tol = 0
    c.reg_covar = -1
    with assert_raises():
        c.validate()
    c.reg_covar = 0
    c.verbose_interval = 0
    with assert_raises():
        c.validate()
    c.verbose_interval = 10
    c.verbose = -1
    with assert_raises():
        c.validate()
    var zero = Float64(0)
    c.verbose = 0
    c.tol = 1 / zero
    with assert_raises():
        c.validate()
    c.tol = 0
    c.reg_covar = zero / zero
    with assert_raises():
        c.validate()
    with assert_raises():
        _ = covariance_shape("unknown", 2, 3)
    with assert_raises():
        _ = covariance_shape("full", 0, 3)
    with assert_raises():
        _ = covariance_shape("full", 2, 0)
    print("PASS dense mixture plumbing")
