from numojo import zeros, Shape
from moxie.decomposition import randomized_svd
from moxie.decomposition._svd_engine import dense_svd, symmetric_eigh
from std.testing import assert_true

def main() raises:
    var a = zeros[DType.float64](Shape(3, 2))
    a.unsafe_set(0, 3)
    a.unsafe_set(3, 2)
    var result = dense_svd(a)
    assert_true(result.singular_values.unsafe_get(0) == 3)
    var randomized = randomized_svd(a, 2, random_state=0)
    assert_true(abs(randomized.singular_values.unsafe_get(0) - 3) < 1e-12)
    var square = zeros[DType.float64](Shape(2, 2))
    square.unsafe_set(0, -1)
    square.unsafe_set(3, 2)
    var eigen = symmetric_eigh(square)
    assert_true(eigen.eigenvalues.unsafe_get(0) == 2)
    print("PASS engine smoke")
