from numojo import NDArray, NDArrayStrides, Shape, zeros
from std.testing import assert_equal, assert_true, assert_raises
from std.math import abs
from moxie._random import RandomState
from moxie.decomposition import randomized_svd
from moxie.decomposition._svd_engine import dense_svd, symmetric_eigh, qr_basis, svd_flip, randomized_range_finder, truncated_svd, center_matrix, golub_kahan_svd


def exercise[dtype: DType]() raises:
    var tol: Float64 = 3e-5 if dtype == DType.float32 else 1e-11
    var owner = zeros[dtype](Shape(5, 8))
    for i in range(40):
        owner.unsafe_set(i, SIMD[dtype, 1](Float64((i * 13) % 23 - 11)))
    var view = owner.view_with_layout(Shape(5, 3), NDArrayStrides(8, -2), 6)
    var snapshot = owner.copy()
    var centered_result = center_matrix(view)
    var centered = centered_result[0].copy()
    for j in range(3):
        var total: Float64 = 0
        for i in range(5):
            total += Float64(centered.unsafe_get(i*3+j))
        assert_true(abs(total) < tol * 10)
    var truncated = truncated_svd(view, 1)
    assert_equal(truncated.U.shape[1], 1)
    with assert_raises():
        var bad = truncated_svd(view, 4)
    with assert_raises():
        var bad = randomized_svd(view, 1, random_state=0, svd_lapack_driver="gesvx")
    # Both drivers reproduce the same deterministic spectrum on a strided view.
    var jacobi = randomized_svd(view, 2, random_state=0)
    var bidiagonal = randomized_svd(view, 2, random_state=0, svd_lapack_driver="gesvd")
    for i in range(2):
        assert_true(abs(Float64(jacobi.singular_values.unsafe_get(i)) - Float64(bidiagonal.singular_values.unsafe_get(i))) < tol * 20)
    var direct = golub_kahan_svd(view)
    assert_equal(direct.U.shape[0], 5)
    assert_equal(direct.Vt.shape[1], 3)
    with assert_raises():
        var bad = golub_kahan_svd(view, max_iterations=0)
    var result = dense_svd(view)
    var copy = result.copy()
    result.U.unsafe_set(0, 100)
    assert_true(copy.U.unsafe_get(0) != 100)
    for i in range(40):
        assert_equal(owner.unsafe_get(i), snapshot.unsafe_get(i))
    var residual: Float64 = 0
    for i in range(5):
        for j in range(3):
            var value: Float64 = 0
            for k in range(3):
                value += Float64(copy.U.unsafe_get(i*3+k)) * Float64(copy.singular_values.unsafe_get(k)) * Float64(copy.Vt.unsafe_get(k*3+j))
            residual = max(residual, abs(value - Float64(view.unsafe_get(i*8-j*2))))
    assert_true(residual < tol * 20)
    var seeded = randomized_svd(view, 2, random_state=42)
    var repeated = randomized_svd(view, 2, random_state=42)
    for i in range(2):
        assert_equal(seeded.singular_values.unsafe_get(i), repeated.singular_values.unsafe_get(i))
    var rng = RandomState(42)
    var fresh = rng.copy()
    var advanced = randomized_svd(view, 2, rng)
    var second = randomized_svd(view, 2, rng)
    assert_true(rng.next_uint32() != fresh.next_uint32())
    var zero = zeros[dtype](Shape(4, 3))
    var z = dense_svd(zero)
    for i in range(3):
        assert_equal(z.singular_values.unsafe_get(i), 0)
        var norm: Float64 = 0
        for j in range(4):
            norm += Float64(z.U.unsafe_get(j*3+i)) ** 2
        assert_true(abs(norm - 1) < tol)
    var u = zeros[dtype](Shape(2, 1))
    var vt = zeros[dtype](Shape(1, 2))
    u.unsafe_set(0, -1)
    u.unsafe_set(1, 1)
    vt.unsafe_set(0, 1)
    vt.unsafe_set(1, -1)
    svd_flip(u, vt)
    assert_equal(u.unsafe_get(0), 1)
    assert_equal(vt.unsafe_get(0), -1)
    svd_flip(u, vt, u_based_decision=False)
    assert_equal(vt.unsafe_get(0), 1)
    with assert_raises():
        var bad = randomized_svd(view, 0, random_state=0)
    with assert_raises():
        var bad = randomized_svd(view, 1, random_state=-1)
    with assert_raises():
        var bad = randomized_svd(view, 1, random_state=0, n_iter=-2)
    with assert_raises():
        var bad = randomized_svd(view, 1, random_state=0, n_oversamples=-1)
    with assert_raises():
        var bad = randomized_svd(view, 1, random_state=0, transpose="invalid")
    with assert_raises():
        var bad = randomized_svd(view, 1, random_state=0, power_iteration_normalizer="invalid")
    with assert_raises():
        var bad = randomized_svd(view, 1, random_state=0, svd_lapack_driver="invalid")
    with assert_raises():
        var bad = dense_svd(view, max_sweeps=1)
    with assert_raises():
        var bad = symmetric_eigh(view)
    var nonsymmetric = zeros[dtype](Shape(2, 2))
    nonsymmetric.unsafe_set(1, 1)
    with assert_raises():
        var bad = symmetric_eigh(nonsymmetric)
    var empty = zeros[dtype](Shape(0, 2))
    with assert_raises():
        var bad = dense_svd(empty)
    var vector = zeros[dtype](Shape(2))
    with assert_raises():
        var bad = dense_svd(vector)
    var invalid = zero.copy()
    invalid.unsafe_set(0, SIMD[dtype, 1](Float64("nan")))
    with assert_raises():
        var bad = dense_svd(invalid)
    invalid.unsafe_set(0, SIMD[dtype, 1](Float64("inf")))
    with assert_raises():
        var bad = dense_svd(invalid)
    print("PASS dense contracts", dtype)


def main() raises:
    exercise[DType.float32]()
    exercise[DType.float64]()
