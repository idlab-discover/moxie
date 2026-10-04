"""Native ownership, layout, failure and callable contracts, both dtypes."""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie._kernel_ridge import (
    pairwise,
    pairwise_callable,
    solve_kernel,
    least_squares,
    cholesky,
    triangular_solve,
)
from moxie._kernel_ridge._solve import solve_kernel_scalar
from moxie._kernel_ridge._kernels import RowKernel


@fieldwise_init
struct Metric(RowKernel):
    var offset: Float64

    def evaluate[
        dtype: DType
    ](
        self, x: nm.NDArray[dtype], i: Int, z: nm.NDArray[dtype], j: Int
    ) raises -> SIMD[dtype, 1]:
        return (
            x.unsafe_get(i * Int(x.strides[0]))
            + 2 * z.unsafe_get(j * Int(z.strides[0]))
            + SIMD[dtype, 1](self.offset)
        )


def contracts[dtype: DType]() raises:
    var A = nm.zeros[dtype](nm.Shape(2, 2))
    A.unsafe_set(0, 2)
    A.unsafe_set(1, 0)
    A.unsafe_set(2, 999)
    A.unsafe_set(3, 4)
    var y = nm.ones[dtype](nm.Shape(2, 1))
    var r = solve_kernel(A, y, [0.0], [1.0])
    assert_true(abs(r.unsafe_get(0) - 0.5) < 1e-6)
    assert_true(abs(r.unsafe_get(1) - 0.25) < 1e-6)
    assert_equal(A.unsafe_get(2), 999)
    assert_equal(y.unsafe_get(0), 1)
    r.unsafe_set(0, 10)
    var again = solve_kernel(A, y, [0.0], [1.0])
    assert_true(abs(again.unsafe_get(0) - 0.5) < 1e-6)
    var vec = nm.ones[dtype](nm.Shape(2))
    var vr = solve_kernel(A, vec, [0.0], [1.0])
    assert_equal(vr.ndim, 1)
    assert_equal(r.ndim, 2)
    var scalar = solve_kernel_scalar(A, y, 0, 2)
    assert_true(abs(scalar.unsafe_get(0) - 0.5) < 1e-6)
    var zeros = solve_kernel(A, y, [0.0], [0.0])
    assert_equal(zeros.unsafe_get(0), 0)
    assert_equal(zeros.unsafe_get(1), 0)
    var weighted = solve_kernel(A, y, [1.0], [0.0, 1.0])
    assert_equal(weighted.unsafe_get(0), 0)
    var multi = nm.ones[dtype](nm.Shape(2, 2))
    var singular = nm.ones[dtype](nm.Shape(2, 2))
    with assert_raises():
        _ = solve_kernel(singular, multi, [0.0, 1.0], [1.0])
    var equal = solve_kernel(singular, multi, [0.0, 0.0], [1.0])
    assert_true(abs(equal.unsafe_get(0) - 0.5) < 1e-5)
    with assert_raises():
        _ = least_squares(singular, y, max_sweeps=0)
    with assert_raises():
        _ = solve_kernel(A, y, [-1.0], [1.0])
    with assert_raises():
        _ = solve_kernel(A, y, [1.0], [-1.0])
    with assert_raises():
        _ = solve_kernel(A, y, [1.0, 1.0], [1.0])
    with assert_raises():
        _ = solve_kernel(A, y, [1.0], [1.0, 1.0, 1.0])
    with assert_raises():
        _ = solve_kernel(A, y, List[SIMD[dtype, 1]](), [1.0])
    with assert_raises():
        _ = solve_kernel(A, y, [1.0], [SIMD[dtype, 1](Float64("inf"))])
    with assert_raises():
        _ = solve_kernel(A, y, [SIMD[dtype, 1](Float64("nan"))], [1.0])
    var bad = nm.zeros[dtype](nm.Shape(0, 2))
    with assert_raises():
        _ = pairwise(bad, A)
    with assert_raises():
        _ = cholesky(y)
    with assert_raises():
        _ = solve_kernel(y, y, [1.0], [1.0])
    with assert_raises():
        _ = pairwise(A, y)
    with assert_raises():
        _ = pairwise(A, A, "unknown")
    with assert_raises():
        _ = pairwise(A, A, "chi2")
    with assert_raises():
        _ = pairwise(A, A, "rbf", -1.0)
    with assert_raises():
        _ = pairwise(A, A, "poly", degree=-1)
    var negative = nm.ones[dtype](nm.Shape(2, 2))
    negative.unsafe_set(0, -1)
    with assert_raises():
        _ = pairwise(negative, A, "chi2", 1.0)
    var nonfinite = A.copy()
    nonfinite.unsafe_set(0, SIMD[dtype, 1](Float64("nan")))
    with assert_raises():
        _ = pairwise(nonfinite, A)
    with assert_raises():
        _ = least_squares(nonfinite, y)
    with assert_raises():
        _ = triangular_solve(nm.zeros[dtype](nm.Shape(2, 2)), y)
    var pre = pairwise(A, A, "precomputed")
    pre.unsafe_set(0, 99)
    assert_equal(A.unsafe_get(0), 2)
    # Negative strides and nonzero offsets: use only NuMojo layout facilities.
    var backing = nm.zeros[dtype](nm.Shape(4, 4))
    backing.unsafe_set(15, 2)
    backing.unsafe_set(13, 0)
    backing.unsafe_set(7, 999)
    backing.unsafe_set(5, 4)
    var view = backing.view_with_layout(
        nm.Shape(2, 2), nm.NDArrayStrides(-8, -2), 15
    )
    var sv = solve_kernel(view, y, [0.0], [1.0])
    assert_true(abs(sv.unsafe_get(0) - 0.5) < 1e-6)
    assert_true(abs(sv.unsafe_get(1) - 0.25) < 1e-6)
    var gram = pairwise(view, A)
    var expected = pairwise(A, A)
    for i in range(4):
        assert_equal(gram.unsafe_get(i), expected.unsafe_get(i))
    var large_backing = nm.ones[dtype](nm.Shape(32, 4))
    var large_view = large_backing.view_with_layout(
        nm.Shape(16, 2), nm.NDArrayStrides(-8, 2), 124
    )
    var large_gram = pairwise(large_view, large_view, same_data=True)
    for i in range(256):
        assert_true(abs(large_gram.unsafe_get(i) - 2) < 1e-6)
    with assert_raises():
        _ = pairwise(nm.ones[dtype](nm.Shape(1, 1, 1)), A)
    var metric = Metric(2)
    var custom = pairwise_callable(A, A, metric, True)
    assert_equal(custom.unsafe_get(1), custom.unsafe_get(2))
    assert_equal(custom.unsafe_get(0), 8)
    assert_equal(custom.unsafe_get(1), 2002)
    var rect = pairwise_callable(A, A, metric)
    assert_equal(rect.unsafe_get(2), 1005)
    var invalid = Metric(Float64("inf"))
    with assert_raises():
        _ = pairwise_callable(A, A, invalid)


def main() raises:
    var large = nm.zeros[DType.float64](nm.Shape(2, 1))
    large.unsafe_set(0, 95306579.74822405)
    large.unsafe_set(1, 95306575.48487124)
    var gram = pairwise(large, large, "rbf", 0.1, same_data=True)
    assert_true(abs(gram.unsafe_get(1) - 0.16529888822158653) < 2e-12)
    assert_equal(gram.unsafe_get(0), 1.0)

    contracts[DType.float32]()
    contracts[DType.float64]()
    print("PASS native contracts Float32/Float64")
