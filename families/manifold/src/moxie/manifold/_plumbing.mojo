"""Dense manifold exchange contracts; no embedding or eigensolver engine."""
from std.math import isfinite, abs
import numojo as nm


def validate_matrix[dtype: DType](x: nm.NDArray[dtype]) raises:
    """Borrow and validate a nonempty finite matrix, including strided inputs."""
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim != 2:
        raise Error("ValueError: expected rank two")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise Error("ValueError: expected nonempty matrix")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not isfinite(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))):
                raise Error("ValueError: expected finite entries")


def validate_pairwise[dtype: DType](
    x: nm.NDArray[dtype], *, nonnegative: Bool = False,
    zero_diagonal: Bool = False, symmetric: Bool = True,
    tolerance: Float64 = 0,
) raises:
    """Validate without repairing, symmetrizing, squaring, or normalizing."""
    validate_matrix(x)
    if x.shape[0] != x.shape[1]:
        raise Error("ValueError: expected square pairwise matrix")
    if not isfinite(tolerance) or tolerance < 0:
        raise Error("ValueError: tolerance must be finite and nonnegative")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            var a = Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
            if nonnegative and a < 0:
                raise Error("ValueError: expected nonnegative entries")
            if zero_diagonal and i == j and abs(a) > tolerance:
                raise Error("ValueError: expected zero diagonal")
            var b = Float64(x.unsafe_get(j * Int(x.strides[0]) + i * Int(x.strides[1])))
            if symmetric and abs(a - b) > tolerance:
                raise Error("ValueError: expected symmetric matrix")


struct DenseAffinity[dtype: DType](Movable):
    """Independent dense adjacency snapshot. Diagonal is preserved."""
    var matrix: nm.NDArray[Self.dtype]

    def __init__(out self, x: nm.NDArray[Self.dtype]) raises:
        validate_pairwise(x, nonnegative=True)
        self.matrix = x.copy()


struct DenseNeighborhood[dtype: DType](Movable):
    """Fixed-k query rows; indices address the reference sample axis."""
    var indices: nm.NDArray[DType.int64]
    var distances: nm.NDArray[Self.dtype]
    var n_reference: Int

    def __init__(out self, indices: nm.NDArray[DType.int64],
                 distances: nm.NDArray[Self.dtype], n_reference: Int) raises:
        validate_matrix(distances)
        if indices.ndim != 2 or n_reference < 1:
            raise Error("ValueError: invalid neighborhood metadata")
        if indices.shape[0] != distances.shape[0] or indices.shape[1] != distances.shape[1]:
            raise Error("ValueError: neighbor arrays must have matching shapes")
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                var index = Int(indices.unsafe_get(i * Int(indices.strides[0]) + j * Int(indices.strides[1])))
                var distance = distances.unsafe_get(i * Int(distances.strides[0]) + j * Int(distances.strides[1]))
                if index < 0 or index >= n_reference or distance < 0:
                    raise Error("ValueError: invalid neighbor index or distance")
        self.indices = indices.copy()
        self.distances = distances.copy()
        self.n_reference = n_reference


struct EmbeddingResult[dtype: DType](Movable):
    """Owned coordinates in input row order, without algorithm-specific scores."""
    var embedding: nm.NDArray[Self.dtype]

    def __init__(out self, coordinates: nm.NDArray[Self.dtype], n_samples: Int,
                 n_components: Int) raises:
        validate_matrix(coordinates)
        if coordinates.shape[0] != n_samples or coordinates.shape[1] != n_components:
            raise Error("ValueError: embedding shape mismatch")
        self.embedding = coordinates.copy()


struct DenseSpectralProblem[dtype: DType](Movable):
    """Symmetric operator; request smallest algebraic eigenpairs then skip."""
    var operator: nm.NDArray[Self.dtype]
    var n_components: Int
    var skip: Int

    def __init__(out self, operator: nm.NDArray[Self.dtype], n_components: Int,
                 skip: Int = 0) raises:
        validate_pairwise(operator)
        if n_components < 1 or skip < 0 or skip >= operator.shape[0]:
            raise Error("ValueError: invalid eigenpair request")
        if n_components > operator.shape[0] - skip:
            raise Error("ValueError: too many eigenpairs")
        self.operator = operator.copy()
        self.n_components = n_components
        self.skip = skip


trait DenseSpectralSolver(Movable):
    """Extension for decomposition-owned solvers, not an implementation."""
    comptime dtype: DType

    def solve(self, problem: DenseSpectralProblem[Self.dtype]) raises -> EmbeddingResult[Self.dtype]:
        ...


trait DenseNeighborQuery(Movable):
    """Adapter implemented by neighbors.nearest, preserving its query order."""
    comptime dtype: DType

    def query(self, x: nm.NDArray[Self.dtype], n_neighbors: Int) raises -> DenseNeighborhood[Self.dtype]:
        ...
