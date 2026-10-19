"""Unit tests of the shared PLSEngine (TwoBlockEngine) kernels."""
import numojo as nm
from std.testing import assert_almost_equal, assert_equal, assert_raises, assert_true, TestSuite
from moxie.cross_decomposition import PLSEngine
from moxie.cross_decomposition._contracts import KernelOptions, PairedScores, TwoBlockEngine
from moxie.cross_decomposition._pls import pinv, svd_flip_1d, EPS64

comptime F64 = DType.float64


def block(rows: Int, cols: Int, values: List[Float64]) raises -> nm.NDArray[F64]:
    var a = nm.zeros[F64](nm.Shape(rows, cols))
    for i in range(rows * cols):
        a.unsafe_set(i, values[i])
    return a^


def vector(values: List[Float64]) raises -> nm.NDArray[F64]:
    var a = nm.zeros[F64](nm.Shape(len(values)))
    for i in range(len(values)):
        a.unsafe_set(i, values[i])
    return a^


def pseudo_random(rows: Int, cols: Int, seed: Int) raises -> nm.NDArray[F64]:
    var a = nm.zeros[F64](nm.Shape(rows, cols))
    var state = UInt64(seed * 2654435761 + 12345)
    for i in range(rows * cols):
        state = state * 6364136223846793005 + 1442695040888963407
        a.unsafe_set(i, Float64(state >> 11) / Float64(UInt64(1) << 53) - 0.5)
    return a^


def generic_engine[E: TwoBlockEngine](engine: E, mut x: nm.NDArray[F64], mut y: nm.NDArray[F64]) raises -> Float64:
    var scaling = engine.center_scale(x, y, True)
    return scaling.x_std.item(0)


def test_svd_flip_1d() raises:
    """Mirror sklearn test_svd_flip_1d: [1,-4,2] -> [-1,4,-2], v flips too."""
    var u = vector([1.0, -4, 2])
    var v = vector([1.0, 2, 3])
    svd_flip_1d(u, v)
    assert_equal(u.item(1), 4.0)
    assert_equal(u.item(0), -1.0)
    assert_equal(v.item(2), -3.0)
    # First maximum wins on exact ties.
    var tie = vector([-2.0, 2.0])
    var other = vector([1.0])
    svd_flip_1d(tie, other)
    assert_equal(tie.item(0), 2.0)
    assert_equal(other.item(0), -1.0)


def test_center_scale() raises:
    var x = block(4, 2, [1.0, 5, 2, 5, 3, 5, 6, 5])
    var y = block(4, 1, [2.0, 4, 6, 8])
    var engine = PLSEngine()
    var s = engine.center_scale(x, y, True)
    assert_equal(s.x_mean.item(0), 3.0)
    assert_equal(s.x_std.item(1), 1.0)  # constant column: std 0 -> 1
    assert_almost_equal(s.x_std.item(0), (14.0 / 3.0) ** 0.5, atol=1e-15)
    assert_equal(x.item(0, 1), 0.0)
    assert_almost_equal(y.item(3, 0), 3.0 / (20.0 / 3.0) ** 0.5, atol=1e-15)
    var x2 = block(2, 1, [1.0, 3])
    var y2 = block(2, 1, [0.0, 0])
    var unit = engine.center_scale(x2, y2, False)
    assert_equal(unit.x_std.item(0), 1.0)
    assert_equal(x2.item(1, 0), 1.0)
    var x3 = block(3, 1, [1.0, 2, 4])
    var y3 = block(3, 1, [1.0, 2, 4])
    assert_true(generic_engine(engine, x3, y3) > 0)


def test_power_method_matches_svd() raises:
    """Mirror sklearn test_singular_value_helpers (mode A, normalized y)."""
    var x = pseudo_random(60, 8, 3)
    var y = x @ pseudo_random(8, 4, 4)
    var engine = PLSEngine()
    var nipals = engine.leading_weights(x, y, KernelOptions(deflation_mode="canonical"))
    var svd = engine.leading_weights(x, y, KernelOptions(deflation_mode="canonical", algorithm="svd"))
    assert_equal(svd.n_iter, 0)
    assert_true(nipals.n_iter >= 1 and nipals.converged)
    for i in range(8):
        assert_almost_equal(nipals.x_weights.item(i), svd.x_weights.item(i), atol=1e-3)
    for j in range(4):
        assert_almost_equal(nipals.y_weights.item(j), svd.y_weights.item(j), atol=1e-3)


def test_constant_target_and_iteration_limit() raises:
    var engine = PLSEngine()
    var x = pseudo_random(10, 3, 1)
    var y = nm.zeros[F64](nm.Shape(10, 2))
    var step = engine.leading_weights(x, y, KernelOptions())
    assert_true(step.constant_target)
    var y2 = pseudo_random(10, 2, 2)
    var capped = engine.leading_weights(x, y2, KernelOptions(max_iter=1))
    assert_equal(capped.n_iter, 1)
    assert_true(not capped.converged)
    with assert_raises(contains="mode must be A or B"):
        _ = engine.leading_weights(x, y2, KernelOptions(mode="C"))


def test_deflate_and_rotations() raises:
    var engine = PLSEngine()
    var x = pseudo_random(12, 3, 5)
    var y = pseudo_random(12, 2, 6)
    var t = x @ block(3, 1, [1.0, 0, 0])
    var u = y @ block(2, 1, [0.0, 1])
    var loadings = engine.deflate(x, y, PairedScores(t.copy(), u.copy()), True)
    # After deflation the residual is orthogonal to the score.
    for j in range(3):
        var dot: Float64 = 0
        for i in range(12):
            dot += x.item(i, j) * t.item(i, 0)
        assert_almost_equal(dot, 0.0, atol=1e-13)
    assert_equal(loadings.y_loadings.shape[0], 2)
    var w = block(3, 2, [1.0, 0, 0, 1, 0, 0])
    var r = engine.rotations(w, w)
    for i in range(6):
        assert_almost_equal(r.unsafe_get(i), w.unsafe_get(i), atol=1e-15)


def test_projection_weights_and_pinv() raises:
    var engine = PLSEngine()
    var x = pseudo_random(20, 4, 7)
    var y = pseudo_random(20, 3, 8)
    var weights = engine.projection_weights(x, y, 2)
    assert_equal(weights.x_weights.shape[1], 2)
    for c in range(2):
        var best: Float64 = 0
        var signed: Float64 = 0
        for i in range(4):
            if abs(weights.x_weights.item(i, c)) > best:
                best = abs(weights.x_weights.item(i, c))
                signed = weights.x_weights.item(i, c)
        assert_true(signed > 0)
    with assert_raises(contains="outside [1, 3]"):
        _ = engine.projection_weights(x, y, 4)
    # Rank-deficient pseudoinverse drops the zero singular value.
    var a = block(2, 2, [1.0, 2, 2, 4])
    var p = pinv(a, 2 * EPS64, "test")
    assert_almost_equal(p.item(0, 0), 0.04, atol=1e-14)
    assert_almost_equal(p.item(1, 1), 0.16, atol=1e-14)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
