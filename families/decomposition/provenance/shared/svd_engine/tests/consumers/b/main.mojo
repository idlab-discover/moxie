from numojo import zeros, Shape
from std.testing import assert_true
from moxie.svd_consumer_a import leading_value as randomized_value
from moxie.svd_consumer_b import leading_value as exact_value


def main() raises:
    var x = zeros[DType.float64](Shape(3, 2))
    x.unsafe_set(0, 3)
    x.unsafe_set(3, 2)
    assert_true(abs(randomized_value(x) - exact_value(x)) < 1e-12)
    print("PASS composed SVD consumers")
