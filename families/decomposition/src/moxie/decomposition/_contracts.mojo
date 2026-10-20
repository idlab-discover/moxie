"""Dense decomposition validation and owned result records; no numerical engine."""
from numojo import NDArray
from std.math import isfinite


def validate_matrix[dtype: DType](
    x: NDArray[dtype], *, min_samples: Int = 1, nonnegative: Bool = False
) raises:
    """Borrow a finite matrix; allow valid strided layouts without copying."""
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if min_samples < 0:
        raise Error("ValueError: min_samples must be nonnegative")
    if x.ndim != 2:
        raise Error("ValueError: expected a dense rank-2 matrix")
    if x.shape[0] < min_samples or x.shape[1] == 0:
        raise Error("ValueError: insufficient samples or features")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            var value = x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))
            if not isfinite(value):
                raise Error("ValueError: input must be finite")
            if nonnegative and value < 0:
                raise Error("ValueError: input must be nonnegative")


def owned_matrix[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Validated independent snapshot; copy preserves NuMojo's backing layout."""
    validate_matrix(x, min_samples=0)
    return x.copy()


def validate_projection[dtype: DType](
    x: NDArray[dtype], components: NDArray[dtype]
) raises:
    """Check X(n,p), components(k,p), including k=0; perform no projection."""
    validate_matrix(x)
    validate_matrix(components, min_samples=0)
    if x.shape[1] != components.shape[1]:
        raise Error("ValueError: component feature count differs from input")


def validate_mean[dtype: DType](mean: NDArray[dtype], n_features: Int) raises:
    """Check the finite, rank-1 centering vector shared by centered models."""
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if n_features <= 0 or mean.ndim != 1:
        raise Error("ValueError: expected a feature mean vector")
    if mean.shape[0] != n_features:
        raise Error("ValueError: mean feature count differs from input")
    for j in range(n_features):
        if not isfinite(mean.unsafe_get(j * Int(mean.strides[0]))):
            raise Error("ValueError: mean must be finite")


@fieldwise_init
struct SVDResult[dtype: DType](Copyable, Movable):
    """Owned U(n,k), singular_values(k), Vt(k,p); engine certifies invariants."""
    var U: NDArray[Self.dtype]
    var singular_values: NDArray[Self.dtype]
    var Vt: NDArray[Self.dtype]


@fieldwise_init
struct EigenResult[dtype: DType](Copyable, Movable):
    """Owned eigenvalues(k), eigenvectors(p,k), descending eigenvalues."""
    var eigenvalues: NDArray[Self.dtype]
    var eigenvectors: NDArray[Self.dtype]


trait DecompositionEngine(Movable):
    """Typed extension point; Params and State belong to the dedicated engine.

    SVD/eigen, coding, NMF and topic engines use separate implementations.
    Input/Output/State are constrained by docs/plumbing.md, not interchangeable
    numerical algorithms. This trait deliberately imposes no estimator API.
    """
    comptime Input: Movable
    comptime Params: Copyable
    comptime State: Movable
    comptime Output: Movable

    def run(mut self, x: Self.Input, params: Self.Params, mut state: Self.State) raises -> Self.Output:
        ...
