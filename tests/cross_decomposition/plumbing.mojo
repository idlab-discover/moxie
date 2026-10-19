from std.testing import assert_equal, assert_raises
import numojo as nm
from moxie.cross_decomposition._contracts import (
    Matrix,
    KernelOptions,
    validate_blocks,
    validate_matrix,
    owned_blocks,
    PairedScores,
    Scaling,
    WeightStep,
    Loadings,
    TwoBlockEngine,
    owned_target,
    ProjectionWeights,
)
from moxie._dense import sum_axis2d


struct UnimplementedEngine(TwoBlockEngine):
    def __init__(out self):
        pass

    def center_scale(
        self, mut x: Matrix, mut y: Matrix, scale: Bool
    ) raises -> Scaling:
        raise Error(
            "NotImplementedError: test engine intentionally raises"
        )

    def leading_weights(
        self, x: Matrix, y: Matrix, options: KernelOptions
    ) raises -> WeightStep:
        raise Error(
            "NotImplementedError: test engine intentionally raises"
        )

    def deflate(
        self,
        mut x: Matrix,
        mut y: Matrix,
        scores: PairedScores,
        canonical: Bool,
    ) raises -> Loadings:
        raise Error(
            "NotImplementedError: test engine intentionally raises"
        )

    def projection_weights(
        self, x: Matrix, y: Matrix, n_components: Int
    ) raises -> ProjectionWeights:
        raise Error("NotImplementedError: SVD adapter belongs to pls session")

    def rotations(self, weights: Matrix, loadings: Matrix) raises -> Matrix:
        raise Error(
            "NotImplementedError: test engine intentionally raises"
        )


def main() raises:
    var x = nm.ones[DType.float64](nm.Shape(3, 2))
    var y = nm.ones[DType.float64](nm.Shape(3, 1))
    validate_blocks(x, y, 2, True)
    validate_blocks(x, y, 1)
    with assert_raises():
        validate_blocks(x, y, 2)
    with assert_raises():
        validate_blocks(x, y, 0, True)
    with assert_raises():
        validate_blocks(x, y, 4, True)
    var mismatch = nm.zeros[DType.float64](nm.Shape(2, 1))
    with assert_raises():
        validate_blocks(x, mismatch, 1)
    var single = nm.zeros[DType.float64](nm.Shape(1, 2))
    with assert_raises():
        validate_blocks(single, single, 1)
    var empty = nm.zeros[DType.float64](nm.Shape(3, 0))
    with assert_raises():
        validate_matrix(empty)
    var vector = nm.zeros[DType.float64](nm.Shape(3))
    with assert_raises():
        validate_matrix(vector)
    var owned = owned_blocks(x, y, 1)
    owned.x.unsafe_set(0, 7.0)
    assert_equal(x.unsafe_get(0), 1.0)
    var view = x.view()
    view.unsafe_set(0, 2.0)
    assert_equal(x.unsafe_get(0), 2.0)
    var totals = sum_axis2d(x, 0)
    assert_equal(totals.unsafe_get(0), 4.0)
    var target = owned_target(vector)
    assert_equal(target.was_1d, True)
    assert_equal(target.values.ndim, 2)
    assert_equal(target.values.shape[1], 1)
    target.values.unsafe_set(0, 5.0)
    assert_equal(vector.unsafe_get(0), 0.0)
    var matrix_target = owned_target(y)
    assert_equal(matrix_target.was_1d, False)
    var nonfinite = x.copy()
    var zero = Float64(0)
    nonfinite.unsafe_set(0, zero / zero)
    with assert_raises():
        validate_matrix(nonfinite)
    nonfinite.unsafe_set(0, Float64(1) / zero)
    with assert_raises():
        validate_matrix(nonfinite)
    var backing = nm.arange[DType.float64](12).reshape(nm.Shape(3, 4))
    var strided = backing.view_with_layout(
        nm.Shape(3, 2), nm.NDArrayStrides(4, 2), 1
    )
    validate_blocks(strided, y, 1)
    var strided_copy = owned_blocks(strided, y, 1)
    strided_copy.x[nm.Item(1, 1)] = 99
    assert_equal(backing.item(1, 3), 7)
    var reverse = backing.view_with_layout(
        nm.Shape(3, 2), nm.NDArrayStrides(-4, 1), 8
    )
    validate_blocks(reverse, y, 1)
    var reverse_vector = backing.view_with_layout(
        nm.Shape(3), nm.NDArrayStrides(-4), 8
    )
    var reverse_target = owned_target(reverse_vector)
    assert_equal(reverse_target.values.item(0, 0), 8.0)
    assert_equal(reverse_target.values.item(2, 0), 0.0)
    var multi = nm.ones[DType.float64](nm.Shape(3, 2))
    validate_blocks(x, multi, 2)
    var empty_rows = nm.zeros[DType.float64](nm.Shape(0, 2))
    with assert_raises():
        validate_matrix(empty_rows)
    with assert_raises():
        _ = owned_target(empty)
    with assert_raises():
        _ = owned_target(nonfinite)
    var independent = owned.copy()
    independent.y.unsafe_set(0, 12.0)
    assert_equal(owned.y.unsafe_get(0), 1.0)
    assert_equal(y.unsafe_get(0), 1.0)
    var defaults = KernelOptions()
    defaults.validate()
    assert_equal(defaults.max_iter, 500)
    assert_equal(defaults.tol, 1e-6)
    for algorithm in [String("nipals"), String("svd")]:
        var canonical = KernelOptions("A", "canonical", algorithm)
        canonical.validate()
    var cca = KernelOptions("B", "canonical")
    cca.validate()
    var invalid = KernelOptions("C")
    with assert_raises():
        invalid.validate()
    invalid = KernelOptions(deflation_mode="bad")
    with assert_raises():
        invalid.validate()
    invalid = KernelOptions(algorithm="bad")
    with assert_raises():
        invalid.validate()
    invalid = KernelOptions(max_iter=0)
    with assert_raises():
        invalid.validate()
    invalid = KernelOptions(tol=-1.0)
    with assert_raises():
        invalid.validate()
    var engine = UnimplementedEngine()
    with assert_raises():
        _ = engine.center_scale(x, y, True)
    with assert_raises():
        _ = engine.leading_weights(x, y, defaults)
    with assert_raises():
        _ = engine.deflate(x, y, PairedScores(x.copy(), y.copy()), False)
    with assert_raises():
        _ = engine.projection_weights(x, y, 1)
    with assert_raises():
        _ = engine.rotations(x, x)
    print("PASS: dense block validation, ownership and engine contracts")
