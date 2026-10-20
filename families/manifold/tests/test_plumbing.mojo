from std.testing import assert_equal, assert_raises
import numojo as nm
from moxie.manifold._plumbing import (
    validate_matrix, validate_pairwise, DenseAffinity, DenseNeighborhood,
    DenseSpectralProblem, DenseSpectralSolver, DenseNeighborQuery, EmbeddingResult,
)

struct SolverProbe(DenseSpectralSolver):
    comptime dtype = DType.float64
    def __init__(out self):
        pass
    def solve(self, problem: DenseSpectralProblem[Self.dtype]) raises -> EmbeddingResult[Self.dtype]:
        raise Error("Contract probe: eigensolver is not implemented")

struct QueryProbe(DenseNeighborQuery):
    comptime dtype = DType.float64
    def __init__(out self):
        pass
    def query(self, x: nm.NDArray[Self.dtype], n_neighbors: Int) raises -> DenseNeighborhood[Self.dtype]:
        raise Error("Contract probe: neighbor search is not implemented")

def exercise[dtype: DType]() raises:
    var x = nm.ones[dtype](nm.Shape(3, 3))
    var affinity = DenseAffinity(x)
    x.unsafe_set(0, 9)
    assert_equal(affinity.matrix.unsafe_get(0), 1)
    var result = EmbeddingResult(x, 3, 3)
    x.unsafe_set(0, 8)
    assert_equal(result.embedding.unsafe_get(0), 9)
    # Signed symmetric operators are valid, unlike affinities.
    x.unsafe_set(0, -1)
    var problem = DenseSpectralProblem(x, 2, 1)
    x.unsafe_set(0, 7)
    assert_equal(problem.operator.unsafe_get(0), -1)
    with assert_raises():
        _ = DenseSpectralProblem(x, 3, 1)
    with assert_raises():
        _ = DenseSpectralProblem(x, 1, -1)
    with assert_raises():
        _ = EmbeddingResult(x, 2, 3)
    var indices = nm.zeros[DType.int64](nm.Shape(3, 2))
    var distances = nm.ones[dtype](nm.Shape(3, 2))
    var neighbors = DenseNeighborhood(indices, distances, 3)
    indices.unsafe_set(0, 2)
    distances.unsafe_set(0, 4)
    assert_equal(neighbors.indices.unsafe_get(0), 0)
    assert_equal(neighbors.distances.unsafe_get(0), 1)
    indices.unsafe_set(0, 3)
    with assert_raises():
        _ = DenseNeighborhood(indices, distances, 3)
    indices.unsafe_set(0, -1)
    with assert_raises():
        _ = DenseNeighborhood(indices, distances, 3)
    indices.unsafe_set(0, 0)
    distances.unsafe_set(0, -1)
    with assert_raises():
        _ = DenseNeighborhood(indices, distances, 3)
    var wrong = nm.zeros[DType.int64](nm.Shape(1, 1))
    with assert_raises():
        _ = DenseNeighborhood(wrong, distances, 3)
    var empty = nm.zeros[dtype](nm.Shape(0, 2))
    with assert_raises():
        validate_matrix(empty)
    var vector = nm.ones[dtype](nm.Shape(3))
    with assert_raises():
        validate_matrix(vector)
    with assert_raises():
        validate_pairwise(distances)
    x.unsafe_set(1, 2)
    with assert_raises():
        validate_pairwise(x)
    validate_pairwise(x, symmetric=False)
    validate_pairwise(x, tolerance=1)
    with assert_raises():
        validate_pairwise(x, tolerance=-1)
    x.unsafe_set(1, 1)
    with assert_raises():
        validate_pairwise(x, zero_diagonal=True)
    x.unsafe_set(0, -1)
    with assert_raises():
        _ = DenseAffinity(x)
    x.unsafe_set(0, Float64("nan").cast[dtype]())
    with assert_raises():
        validate_matrix(x)
    x.unsafe_set(0, Float64("inf").cast[dtype]())
    with assert_raises():
        validate_matrix(x)
    # Offset and negative stride semantics belong to NuMojo, not this family.
    var backing = nm.ones[dtype](nm.Shape(5, 5))
    var strides = nm.NDArrayStrides(5, -1)
    var view = backing.view_with_layout(nm.Shape(3, 3), strides, 3)
    var snapshot = DenseAffinity(view)
    view.unsafe_set(0, 5)
    assert_equal(snapshot.matrix.unsafe_get(0), 1)
    assert_equal(backing.unsafe_get(3), 5)
    validate_matrix(view)

def main() raises:
    exercise[DType.float32]()
    exercise[DType.float64]()
    var x = nm.ones[DType.float64](nm.Shape(2, 2))
    var problem = DenseSpectralProblem(x, 1)
    var solver = SolverProbe()
    var query = QueryProbe()
    with assert_raises():
        _ = solver.solve(problem)
    with assert_raises():
        _ = query.query(x, 1)
    print("PASS: manifold dense validation, ownership and extension contracts")
