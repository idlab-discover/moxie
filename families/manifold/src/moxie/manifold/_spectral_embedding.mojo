# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense Laplacian eigenmaps for manifold learning and spectral clustering."""
from std.math import abs, isfinite, sqrt
from numojo import NDArray, Shape, ones, zeros
from mojo_linalg import symmetric_eigh
from moxie._dense import sum_axis2d
from moxie._feature_metadata import InputFeatureCount
from moxie._kernel_ridge._kernels import pairwise
from moxie._multiclass_contracts import PairwiseInput
from moxie.manifold._base import EmbeddingEstimator
from moxie.manifold._plumbing import (
    DenseSpectralProblem, DenseSpectralSolver, EmbeddingResult, validate_matrix,
    validate_pairwise,
)
from moxie.neighbors._distance import at
from moxie.warnings import FitWarning


struct _JacobiSolver(DenseSpectralSolver):
    comptime dtype = DType.float64

    def __init__(out self):
        pass

    def solve(
        self, problem: DenseSpectralProblem[Self.dtype],
    ) raises -> EmbeddingResult[Self.dtype]:
        var factor = symmetric_eigh(problem.operator)
        var vectors = factor.eigenvectors()
        var n = problem.operator.shape[0]
        var dimensions = problem.n_components
        var coordinates = zeros[Self.dtype](Shape(n, dimensions))
        for row in range(n):
            for col in range(dimensions):
                coordinates.unsafe_set(row * dimensions + col, at(vectors, row, col + problem.skip))
        return EmbeddingResult(coordinates, n, dimensions)


def _validate_solver(
    eigen_solver: Optional[String], eigen_tol: Optional[Float64], random_state: Optional[Int],
) raises:
    var tolerance = eigen_tol.or_else(0)
    if not isfinite(tolerance) or tolerance < 0:
        raise Error("ValueError: eigen_tol must be finite and nonnegative")
    var seed = random_state.or_else(0)
    if seed < 0 or seed > 4294967295:
        raise Error("ValueError: random_state must be between 0 and 2**32 - 1")
    if not eigen_solver:
        raise Error(
            "NotImplementedError: eigen_solver=None selects ARPACK; use eigen_solver='dense'"
        )
    ref solver = eigen_solver.value()
    if solver == "dense":
        return
    if solver == "arpack" or solver == "lobpcg" or solver == "amg":
        raise Error("NotImplementedError: use eigen_solver='dense' for dense spectral embedding")
    raise Error("ValueError: eigen_solver must be dense, arpack, lobpcg, amg, or None")


def _graph_is_connected(off_diagonal: NDArray[DType.float64]) raises -> Bool:
    var n = off_diagonal.shape[0]
    var visited = List[Bool](capacity=n)
    for _ in range(n):
        visited.append(False)
    var queue = List[Int](capacity=n)
    visited[0] = True
    queue.append(0)
    var head = 0
    while head < len(queue):
        var row = queue[head]
        head += 1
        for col in range(n):
            if not visited[col] and off_diagonal.unsafe_get(row * n + col) != 0:
                visited[col] = True
                queue.append(col)
        if len(queue) == n:
            return True
    return False


def _spectral_coordinates[dtype: DType](
    adjacency: NDArray[dtype], requested_components: Int, norm_laplacian: Bool,
    drop_first: Bool, mut warnings: List[FitWarning],
) raises -> NDArray[dtype]:
    validate_pairwise(adjacency, nonnegative=True, symmetric=False)
    var n = adjacency.shape[0]
    var skip = Int(drop_first)
    if requested_components < 1:
        raise Error("ValueError: n_components must be positive")
    var n_components = min(requested_components, n - skip)
    if n_components == 0:
        return zeros[dtype](Shape(n, 0))

    var laplacian = zeros[DType.float64](Shape(n, n))
    var needs_warning = False
    for row in range(n):
        for col in range(row):
            var a = at(adjacency, row, col)
            var b = at(adjacency, col, row)
            var difference = abs(a - b)
            if difference > 1e-10 + 1e-5 * min(a, b):
                needs_warning = True
            # Use the lower weight plus half the gap to avoid overflow.
            var weight = min(a, b) + 0.5 * difference
            laplacian.unsafe_set(row * n + col, -weight)
            laplacian.unsafe_set(col * n + row, -weight)
    if needs_warning:
        warnings.append(FitWarning(
            "UserWarning",
            "Array is not symmetric, and will be converted to symmetric by average with its transpose.",
        ))
    if not _graph_is_connected(laplacian):
        warnings.append(FitWarning(
            "UserWarning",
            "Graph is not fully connected, spectral embedding may not work as expected.",
        ))

    var negative_degrees = sum_axis2d(laplacian, 1)
    var scales = List[Float64](capacity=n)
    for row in range(n):
        var degree = -negative_degrees.unsafe_get(row)
        if not isfinite(degree):
            raise Error("ValueError: spectral graph degree must be finite")
        var diagonal = degree
        var scale: Float64 = 1
        if norm_laplacian and degree > 0:
            diagonal = 1
            scale = sqrt(degree)
        laplacian.unsafe_set(row * n + row, diagonal)
        scales.append(scale)
    if norm_laplacian:
        for row in range(n):
            for col in range(row):
                var value = laplacian.unsafe_get(row * n + col) / scales[row] / scales[col]
                laplacian.unsafe_set(row * n + col, value)
                laplacian.unsafe_set(col * n + row, value)

    var problem = DenseSpectralProblem(laplacian, n_components, skip)
    var solved = _JacobiSolver().solve(problem)
    var output = zeros[dtype](Shape(n, n_components))
    for col in range(n_components):
        var pivot = SIMD[dtype, 1](0)
        for row in range(n):
            var index = row * n_components + col
            var value = SIMD[dtype, 1](solved.embedding.unsafe_get(index) / scales[row])
            if not isfinite(value):
                raise Error("ValueError: spectral coordinate exceeds output dtype")
            output.unsafe_set(index, value)
            if abs(value) > abs(pivot):
                pivot = value
        # Apply sklearn's sign rule after undoing D**1/2 and casting.
        if pivot < 0:
            for row in range(n):
                var index = row * n_components + col
                output.unsafe_set(index, -output.unsafe_get(index))
    return output^


def spectral_embedding[dtype: DType](
    adjacency: NDArray[dtype], *, n_components: Int = 8,
    eigen_solver: Optional[String] = None, random_state: Optional[Int] = None,
    eigen_tol: Optional[Float64] = None, norm_laplacian: Bool = True,
    drop_first: Bool = True,
) raises -> NDArray[dtype]:
    """Return the smallest dense Laplacian eigenvectors in sample-row order."""
    _validate_solver(eigen_solver, eigen_tol, random_state)
    var warnings = List[FitWarning]()
    return _spectral_coordinates(adjacency, n_components, norm_laplacian, drop_first, warnings)


comptime _AffinityFunction[dtype: DType] = def(
    NDArray[dtype],
) thin raises -> NDArray[dtype]


struct SpectralEmbeddingParams[dtype: DType = DType.float64](Copyable, Movable):
    var n_components: Int
    var affinity: String
    var gamma: Optional[Float64]
    var random_state: Optional[Int]
    var eigen_solver: Optional[String]
    var eigen_tol: Optional[Float64]
    var n_neighbors: Optional[Int]
    var n_jobs: Optional[Int]
    var affinity_function: Optional[_AffinityFunction[Self.dtype]]

    def __init__(
        out self, n_components: Int, affinity: String, gamma: Optional[Float64],
        random_state: Optional[Int], eigen_solver: Optional[String],
        eigen_tol: Optional[Float64], n_neighbors: Optional[Int], n_jobs: Optional[Int],
        affinity_function: Optional[_AffinityFunction[Self.dtype]] = None,
    ):
        self.n_components = n_components
        self.affinity = affinity.copy()
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver.copy()
        self.eigen_tol = eigen_tol
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.affinity_function = affinity_function


struct SpectralEmbedding[dtype: DType = DType.float64](
    Copyable, EmbeddingEstimator, InputFeatureCount, PairwiseInput,
):
    """Dense Laplacian eigenmaps with RBF, precomputed, or callable affinity."""
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Transformed = Self.Input
    comptime Params = SpectralEmbeddingParams[Self.dtype]

    var _params: Self.Params
    var embedding_: Optional[Self.Transformed]
    var affinity_matrix_: Optional[Self.Input]
    var n_features_in_: Int
    var gamma_: Optional[Float64]
    var fit_warnings: List[FitWarning]

    def __init__(
        out self, n_components: Int = 2, *, affinity: String = "nearest_neighbors",
        gamma: Optional[Float64] = None, random_state: Optional[Int] = None,
        eigen_solver: Optional[String] = None, eigen_tol: Optional[Float64] = None,
        n_neighbors: Optional[Int] = None, n_jobs: Optional[Int] = None,
        affinity_function: Optional[_AffinityFunction[Self.dtype]] = None,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self._params = Self.Params(
            n_components, affinity, gamma, random_state, eigen_solver,
            eigen_tol, n_neighbors, n_jobs, affinity_function,
        )
        self.embedding_ = None
        self.affinity_matrix_ = None
        self.n_features_in_ = 0
        self.gamma_ = None
        self.fit_warnings = List[FitWarning]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self._params = params.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_components: Optional[Int] = None,
        affinity: Optional[String] = None, gamma: Optional[Float64] = None,
        random_state: Optional[Int] = None, eigen_solver: Optional[String] = None,
        eigen_tol: Optional[Float64] = None, n_neighbors: Optional[Int] = None,
        n_jobs: Optional[Int] = None,
        affinity_function: Optional[_AffinityFunction[Self.dtype]] = None,
    ) raises -> ref[self] Self:
        if n_components:
            self._params.n_components = n_components.value()
        if affinity:
            self._params.affinity = affinity.value().copy()
        if gamma:
            self._params.gamma = gamma
        if random_state:
            self._params.random_state = random_state
        if eigen_solver:
            self._params.eigen_solver = eigen_solver.copy()
        if eigen_tol:
            self._params.eigen_tol = eigen_tol
        if n_neighbors:
            self._params.n_neighbors = n_neighbors
        if n_jobs:
            self._params.n_jobs = n_jobs
        if affinity_function:
            self._params.affinity_function = affinity_function
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result._params = self._params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.embedding_)

    def pairwise_input(self) -> Bool:
        return (
            self._params.affinity == "precomputed"
            or self._params.affinity == "precomputed_nearest_neighbors"
        )

    def get_n_features_in(self) raises -> Int:
        if not self.embedding_:
            raise Error("AttributeError: SpectralEmbedding has no n_features_in_ before fit")
        return self.n_features_in_

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        ref params = self._params
        if (
            params.affinity == "nearest_neighbors"
            or params.affinity == "precomputed_nearest_neighbors"
        ):
            raise Error(
                "NotImplementedError: spectral nearest-neighbor affinity requires sparse graphs"
            )
        var callable_affinity = params.affinity == "callable"
        if params.affinity != "rbf" and params.affinity != "precomputed" and not callable_affinity:
            raise Error("ValueError: unsupported SpectralEmbedding affinity")
        if callable_affinity and not params.affinity_function:
            raise Error("ValueError: affinity='callable' requires affinity_function")
        if not callable_affinity and params.affinity_function:
            raise Error("ValueError: affinity_function requires affinity='callable'")
        _validate_solver(params.eigen_solver, params.eigen_tol, params.random_state)
        var configured_gamma = params.gamma.or_else(0)
        if not isfinite(configured_gamma) or configured_gamma < 0:
            raise Error("ValueError: gamma must be finite and nonnegative")
        if params.n_neighbors.or_else(1) < 1:
            raise Error("ValueError: n_neighbors must be positive")
        validate_matrix(X)
        if X.shape[0] < 2:
            raise Error("ValueError: SpectralEmbedding requires at least two samples")
        if params.n_components < 1:
            raise Error("ValueError: n_components must be positive")

        var affinity_matrix: Self.Input
        var gamma: Optional[Float64] = None
        if params.affinity == "rbf":
            var effective_gamma = params.gamma.or_else(1 / Float64(X.shape[1]))
            gamma = effective_gamma
            if effective_gamma == 0:
                affinity_matrix = ones[Self.dtype](Shape(X.shape[0], X.shape[0]))
            else:
                affinity_matrix = pairwise(X, X, "rbf", gamma, same_data=True)
        elif callable_affinity:
            var callback = params.affinity_function.value()
            var graph = callback(X)
            var n = X.shape[0]
            if graph.ndim != 2 or graph.shape[0] != n or graph.shape[1] != n:
                raise Error("ValueError: affinity_function must return a square matrix with one row per sample")
            affinity_matrix = graph.contiguous()
        else:
            affinity_matrix = X.contiguous()
        var warnings = List[FitWarning]()
        var embedding = _spectral_coordinates(
            affinity_matrix, params.n_components, True, True, warnings,
        )

        self.embedding_ = embedding^
        self.affinity_matrix_ = affinity_matrix^
        self.n_features_in_ = X.shape[1]
        self.gamma_ = gamma
        self.fit_warnings = warnings^
        return self

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.embedding_.value().copy()
