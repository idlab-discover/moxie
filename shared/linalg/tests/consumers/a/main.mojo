from moxie._linalg import covariance_inverse
from numojo import Shape, Item, zeros
from std.math import abs
from std.testing import assert_true


def main() raises:
    # Sample covariance of [[0,0],[1,0],[0,1]], ddof=1.
    var covariance = zeros[DType.float64](Shape(2, 2))
    covariance[Item(0, 0)] = 1.0 / 3
    covariance[Item(1, 1)] = 1.0 / 3
    covariance[Item(0, 1)] = -1.0 / 6
    covariance[Item(1, 0)] = -1.0 / 6
    var precision = covariance_inverse(covariance)
    assert_true(abs(precision[Item(0, 0)] - 4) < 1e-12)
    assert_true(abs(precision[Item(0, 1)] - 2) < 1e-12)
    assert_true(abs(precision[Item(1, 1)] - 4) < 1e-12)
    print("PASS Mahalanobis precision consumer")
