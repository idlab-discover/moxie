"""Raw coordinate engine contracts, including zero-step and nonconverged states."""
from numojo import zeros, Shape
from std.testing import assert_true, assert_equal, assert_almost_equal, assert_raises
from moxie.linear_model._cd_fast import coordinate_descent


def check[dtype: DType]() raises:
    var X = zeros[dtype](Shape(2, 1))
    X.unsafe_set(0, 1)
    X.unsafe_set(1, 2)
    var y = zeros[dtype](Shape(2))
    y.unsafe_set(0, 2)
    y.unsafe_set(1, 4)
    var start = zeros[dtype](Shape(1))
    var result = coordinate_descent(X, y, start, 1.0, 1.0, 100, 1e-6)
    assert_almost_equal(Float64(result.coef.unsafe_get(0)), 1.5, atol=1e-6)
    assert_true(result.converged)
    assert_equal(start.unsafe_get(0), 0)
    var warm = coordinate_descent(X, y, result.coef, 1.0, 1.0, 100, 1e-6)
    assert_equal(warm.iterations, 0)
    with assert_raises():
        _ = coordinate_descent(X, y, start, -1.0, 0.0, 100, 1e-4)
    with assert_raises():
        _ = coordinate_descent(X, y, start, 1.0, 0.0, 0, 1e-4)
    with assert_raises():
        _ = coordinate_descent(X, y, zeros[dtype](Shape(2)), 1.0, 0.0, 100, 1e-4)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("Coordinate engine contracts passed")
