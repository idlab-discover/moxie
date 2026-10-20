from std.testing import assert_equal, assert_almost_equal, assert_raises
from numojo import zeros, Shape, NDArrayStrides
from moxie.manifold import trustworthiness


def check[dtype: DType]() raises:
    var x = zeros[dtype](Shape(7, 2))
    var y = zeros[dtype](Shape(7, 1))
    for i in range(7):
        x.unsafe_set(i * 2, SIMD[dtype, 1](i * i))
        x.unsafe_set(i * 2 + 1, SIMD[dtype, 1](i))
        y.unsafe_set(i, SIMD[dtype, 1](i * i))
    assert_equal(trustworthiness(x, x, n_neighbors=2), Float64(1))
    assert_equal(trustworthiness(x, y, n_neighbors=2), Float64(1))
    with assert_raises():
        _ = trustworthiness(x, y, n_neighbors=4)
    with assert_raises():
        _ = trustworthiness(x, y, n_neighbors=0)
    with assert_raises():
        _ = trustworthiness(x, y, n_neighbors=1, metric="invalid")
    with assert_raises():
        _ = trustworthiness(x, y, n_neighbors=1, metric="precomputed")
    var wrong = zeros[dtype](Shape(6, 1))
    with assert_raises():
        _ = trustworthiness(x, wrong, n_neighbors=1)
    var invalid = x.copy()
    invalid.unsafe_set(0, Float64("nan").cast[dtype]())
    with assert_raises():
        _ = trustworthiness(invalid, y, n_neighbors=1)
    var distances = zeros[dtype](Shape(7, 7))
    distances.unsafe_set(1, -1)
    with assert_raises():
        _ = trustworthiness(distances, y, n_neighbors=1, metric="precomputed")
    var view = x.view_with_layout(Shape(7, 2), NDArrayStrides(-2, 1), 12)
    assert_equal(trustworthiness(view, view, n_neighbors=2), Float64(1))
    assert_equal(x.unsafe_get(0), SIMD[dtype, 1](0))


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: trustworthiness contracts")
