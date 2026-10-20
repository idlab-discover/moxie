from numojo import NDArray, Shape, Item, NDArrayStrides, arange, zeros
from moxie.decomposition._contracts import (
    validate_matrix, validate_mean, validate_projection, owned_matrix,
    SVDResult, EigenResult, DecompositionEngine,
)
from moxie._random import RandomState
from std.testing import assert_equal, assert_raises

@fieldwise_init
struct RaisingEngine(DecompositionEngine):
    comptime Input = NDArray[DType.float64]
    comptime Params = Int
    comptime State = RandomState
    comptime Output = SVDResult[DType.float64]

    def run(mut self, x: Self.Input, params: Self.Params, mut state: Self.State) raises -> Self.Output:
        raise Error("Contract probe: no numerical engine implemented")

def invoke[E: DecompositionEngine](mut engine: E, x: E.Input, params: E.Params, mut state: E.State) raises -> E.Output:
    return engine.run(x, params, state)

def contracts[dtype: DType]() raises:
    var x = arange[dtype](12).reshape(Shape(3, 4))
    validate_matrix(x)
    var view = x.view_with_layout(Shape(3, 2), NDArrayStrides(4, -2), 3)
    validate_matrix(view, nonnegative=True)
    var snapshot = owned_matrix(view)
    view[Item(1, 1)] = -4
    assert_equal(snapshot.item(1, 1), 5)
    with assert_raises():
        validate_matrix(view, nonnegative=True)
    validate_matrix(view)
    var components = zeros[dtype](Shape(0, 4))
    validate_projection(x, components)
    var wrong = zeros[dtype](Shape(2, 3))
    with assert_raises():
        validate_projection(x, wrong)
    var empty = zeros[dtype](Shape(0, 4))
    with assert_raises():
        validate_matrix(empty)
    validate_matrix(empty, min_samples=0)
    with assert_raises():
        validate_matrix(x, min_samples=4)
    with assert_raises():
        validate_matrix(x, min_samples=-1)
    var no_features = zeros[dtype](Shape(3, 0))
    with assert_raises():
        validate_matrix(no_features)
    var mean = zeros[dtype](Shape(4))
    validate_mean(mean, 4)
    with assert_raises():
        validate_matrix(mean)
    with assert_raises():
        validate_mean(mean, 3)
    with assert_raises():
        validate_mean(x, 4)
    var zero = SIMD[dtype, 1](0)
    x[Item(0, 0)] = SIMD[dtype, 1](1) / zero
    with assert_raises():
        validate_matrix(x)
    x[Item(0, 0)] = zero / zero
    with assert_raises():
        _ = owned_matrix(x)
    mean[Item(0)] = zero / zero
    with assert_raises():
        validate_mean(mean, 4)
    # Records establish shapes and movement only; these are fixture factors.
    var u = zeros[dtype](Shape(3, 2))
    var s = zeros[dtype](Shape(2))
    var vt = zeros[dtype](Shape(2, 4))
    var factors = SVDResult[dtype](u^, s^, vt^)
    var copied = factors.copy()
    factors.Vt[Item(0, 0)] = 7
    assert_equal(copied.Vt.item(0, 0), 0)
    var values = zeros[dtype](Shape(2))
    var vectors = zeros[dtype](Shape(4, 2))
    var eigen = EigenResult[dtype](values^, vectors^)
    assert_equal(eigen.eigenvectors.shape[1], 2)

def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    var engine = RaisingEngine()
    var rng = RandomState(42)
    var x = zeros[DType.float64](Shape(2, 2))
    with assert_raises():
        _ = invoke(engine, x, 1, rng)
    print("PASS: decomposition layouts, ownership, validation and engine extension")
