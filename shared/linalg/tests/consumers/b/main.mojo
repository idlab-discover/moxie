from moxie._linalg import covariance_factor
from numojo import Shape, Item, zeros
from std.math import abs
from std.testing import assert_true


def main() raises:
    var covariance = zeros[DType.float32](Shape(2, 2))
    covariance[Item(0, 0)] = 4
    covariance[Item(1, 1)] = 9
    var factor = covariance_factor(covariance)
    var rhs = zeros[DType.float32](Shape(2, 2))
    rhs[Item(0, 0)] = 8
    rhs[Item(1, 1)] = 27
    var solution = factor.solve(rhs)
    assert_true(abs(solution[Item(0, 0)] - 2) < 1e-5)
    assert_true(abs(solution[Item(1, 1)] - 3) < 1e-5)
    var inverse = factor.inverse()
    assert_true(abs(inverse[Item(1, 1)] - 1.0 / 9) < 1e-5)
    print("PASS reusable factor consumer")
