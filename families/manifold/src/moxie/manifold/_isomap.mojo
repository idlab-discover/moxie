# Adapted from scikit-learn sklearn/manifold/_isomap.py.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense Isomap with neighborhood geodesics and held-out projection."""
from numojo import NDArray, Shape, full, zeros
from std.math import isfinite, sqrt
from std.utils.numerics import inf
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie.warnings import FitWarning
from moxie.decomposition._kernel_pca import KernelPCA
from moxie.neighbors._distance import Metric, at
from moxie.neighbors._unsupervised import NearestNeighbors
from moxie.preprocessing._simple import KernelCenterer
from ._base import OutOfSampleEmbedding


def _connect(
    mut graph: NDArray[DType.float64], n: Int, row: Int, col: Int, distance: Float64,
) raises:
    if not isfinite(distance) or distance < 0:
        raise Error("ValueError: Isomap requires finite nonnegative graph distances")
    var weight = min(distance, graph.unsafe_get(row * n + col))
    graph.unsafe_set(row * n + col, weight)
    graph.unsafe_set(col * n + row, weight)


def _repair_components[dtype: DType](
    mut graph: NDArray[DType.float64], X: NDArray[dtype], metric: Metric[dtype],
) raises -> Int:
    var n = X.shape[0]
    var labels = List[Int](length=n, fill=-1)
    var groups = List[List[Int]]()
    for start in range(n):
        if labels[start] >= 0:
            continue
        var component = len(groups)
        var pending: List[Int] = [start]
        labels[start] = component
        while len(pending) > 0:
            var row = pending.pop()
            for candidate in range(n):
                if labels[candidate] < 0 and isfinite(graph.unsafe_get(row * n + candidate)):
                    labels[candidate] = component
                    pending.append(candidate)
        groups.append(List[Int]())
    if len(groups) == 1:
        return 1

    # Form groups in row order to match sklearn's first minimum on a tie.
    for row in range(n):
        groups[labels[row]].append(row)
    var prepared = metric.self_query_metric(X)
    for i in range(1, len(groups)):
        for j in range(i):
            var best = inf[DType.float64]()
            var closest_row = -1
            var closest_col = -1
            for row in groups[i]:
                for col in groups[j]:
                    var distance = (
                        at(X, row, col) if metric.name == "precomputed"
                        else prepared.distance(X, row, X, col)
                    )
                    if distance < best:
                        best = distance
                        closest_row = row
                        closest_col = col
            if closest_row < 0:
                raise Error("ValueError: Isomap could not connect the neighbors graph with finite distances")
            _connect(graph, n, closest_row, closest_col, best)
    return len(groups)


def _floyd_warshall(mut graph: NDArray[DType.float64]) raises:
    var n = graph.shape[0]
    for via in range(n):
        for source in range(n):
            var prefix = graph.unsafe_get(source * n + via)
            if not isfinite(prefix):
                continue
            for target in range(n):
                var offset = source * n + target
                var distance = prefix + graph.unsafe_get(via * n + target)
                graph.unsafe_set(offset, min(distance, graph.unsafe_get(offset)))


def _dijkstra(graph: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    var n = graph.shape[0]
    var paths = full[DType.float64](Shape(n, n), inf[DType.float64]())
    for source in range(n):
        var settled = List[Bool](length=n, fill=False)
        var row = source * n
        paths.unsafe_set(row + source, 0)
        for _ in range(n):
            var closest = -1
            var distance = inf[DType.float64]()
            for target in range(n):
                var candidate = paths.unsafe_get(row + target)
                if not settled[target] and candidate < distance:
                    closest = target
                    distance = candidate
            if closest < 0:
                raise Error("ValueError: Isomap shortest path is not finite")
            settled[closest] = True
            for target in range(n):
                var offset = row + target
                var candidate = distance + graph.unsafe_get(closest * n + target)
                paths.unsafe_set(offset, min(candidate, paths.unsafe_get(offset)))
    return paths^


def _isomap_kernel[dtype: DType](mut values: NDArray[dtype]):
    # Callers provide contiguous buffers and choose the arithmetic dtype.
    for i in range(values.size):
        var value = values.unsafe_get(i)
        values.unsafe_set(i, (-0.5) * (value * value))


def _extend_row[dtype: DType](
    mut output: NDArray[dtype], row: Int, paths: NDArray[dtype],
    indices: List[Int], distances: NDArray[dtype], start: Int, end: Int,
) raises:
    if start == end:
        raise Error("ValueError: Isomap transform found no neighbors within radius")
    var n = paths.shape[0]
    for target in range(n):
        var distance = inf[DType.float64]()
        for edge in range(start, end):
            var candidate = Float64(distances.unsafe_get(edge)) + Float64(
                paths.unsafe_get(indices[edge] * n + target)
            )
            distance = min(distance, candidate)
        output.unsafe_set(row * n + target, distance.cast[dtype]())


@fieldwise_init
struct IsomapParams[dtype: DType](Copyable, Movable):
    var n_neighbors: Optional[Int]
    var radius: Optional[Float64]
    var n_components: Int
    var eigen_solver: String
    var tol: Float64
    var max_iter: Optional[Int]
    var path_method: String
    var neighbors_algorithm: String
    var n_jobs: Optional[Int]
    var metric: String
    var p: Float64
    var metric_params: Optional[Metric[Self.dtype]]


struct Isomap[dtype: DType = DType.float64](OutOfSampleEmbedding, FeatureNamesOut, InputFeatureCount):
    """Embed dense observations using shortest paths in their neighborhood graph."""
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    comptime Params = IsomapParams[Self.dtype]

    var _params: Self.Params
    var embedding_: Optional[Self.Transformed]
    var kernel_pca_: Optional[KernelPCA[Self.dtype]]
    var nbrs_: Optional[NearestNeighbors[Self.dtype]]
    var dist_matrix_: Optional[Self.Input]
    var n_features_in_: Int
    var fit_warnings: List[FitWarning]
    var _uses_radius: Bool

    def __init__(out self, *, n_neighbors: Optional[Int] = 5,
                 radius: Optional[Float64] = None, n_components: Int = 2,
                 eigen_solver: String = "auto", tol: Float64 = 0,
                 max_iter: Optional[Int] = None, path_method: String = "auto",
                 neighbors_algorithm: String = "auto", n_jobs: Optional[Int] = None,
                 metric: String = "minkowski", p: Float64 = 2,
                 metric_params: Optional[Metric[Self.dtype]] = None):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self._params = Self.Params(
            n_neighbors, radius, n_components, eigen_solver.copy(), tol, max_iter,
            path_method.copy(), neighbors_algorithm.copy(), n_jobs, metric.copy(),
            p, metric_params.copy(),
        )
        self.embedding_ = None
        self.kernel_pca_ = None
        self.nbrs_ = None
        self.dist_matrix_ = None
        self.n_features_in_ = 0
        self.fit_warnings = []
        self._uses_radius = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self._params = params.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_neighbors: Optional[Optional[Int]] = None,
        radius: Optional[Optional[Float64]] = None, n_components: Optional[Int] = None,
        eigen_solver: Optional[String] = None, tol: Optional[Float64] = None,
        max_iter: Optional[Optional[Int]] = None, path_method: Optional[String] = None,
        neighbors_algorithm: Optional[String] = None, n_jobs: Optional[Optional[Int]] = None,
        metric: Optional[String] = None, p: Optional[Float64] = None,
        metric_params: Optional[Optional[Metric[Self.dtype]]] = None,
    ) raises -> ref[self] Self:
        if n_neighbors:
            self._params.n_neighbors = n_neighbors.value()
        if radius:
            self._params.radius = radius.value()
        if n_components:
            self._params.n_components = n_components.value()
        if eigen_solver:
            self._params.eigen_solver = eigen_solver.value().copy()
        if tol:
            self._params.tol = tol.value()
        if max_iter:
            self._params.max_iter = max_iter.value()
        if path_method:
            self._params.path_method = path_method.value().copy()
        if neighbors_algorithm:
            self._params.neighbors_algorithm = neighbors_algorithm.value().copy()
        if n_jobs:
            self._params.n_jobs = n_jobs.value()
        if metric:
            self._params.metric = metric.value().copy()
        if p:
            self._params.p = p.value()
        if metric_params:
            self._params.metric_params = metric_params.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result._params = self._params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.embedding_)

    def get_n_features_in(self) raises -> Int:
        if not self.__sklearn_is_fitted__():
            raise Error("AttributeError: Isomap has no n_features_in_ before fit")
        return self.n_features_in_

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.embedding_:
            raise Error("NotFittedError: Isomap is not fitted")
        if input_features and len(input_features.value()) != self.n_features_in_:
            raise Error("ValueError: input_features length differs from n_features_in_")
        var names = List[String]()
        for component in range(self.embedding_.value().shape[1]):
            names.append("isomap" + String(component))
        return names^

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: Isomap supports native array output only")

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        ref params = self._params
        if params.n_neighbors and params.radius:
            raise Error("ValueError: Both n_neighbors and radius are provided. Set n_neighbors=None to use radius-based neighbors")
        if not params.n_neighbors and not params.radius:
            raise Error("ValueError: Isomap requires n_neighbors or radius")
        if params.n_components < 1:
            raise Error("ValueError: n_components must be positive")
        if params.path_method != "auto" and params.path_method != "FW" and params.path_method != "D":
            raise Error("ValueError: path_method must be auto, FW, or D")
        if params.eigen_solver != "auto" and params.eigen_solver != "dense" and params.eigen_solver != "arpack":
            raise Error("ValueError: eigen_solver must be auto, dense, or arpack")
        if not (params.p >= 1):
            raise Error("ValueError: Isomap p must be at least one")

        var nbrs = NearestNeighbors[Self.dtype](
            params.n_neighbors.or_else(5), params.radius.or_else(1.0),
            algorithm=params.neighbors_algorithm, metric=params.metric, p=params.p,
            metric_params=params.metric_params, n_jobs=params.n_jobs,
        )
        _ = nbrs.fit(X)
        var n = nbrs.n_samples_fit_
        var graph = full[DType.float64](Shape(n, n), inf[DType.float64]())
        for i in range(n):
            graph.unsafe_set(i * n + i, 0)
        var edges: Int
        if params.n_neighbors:
            var neighbors = nbrs.kneighbors()
            ref distances = neighbors.distances.value()
            for i in range(n):
                for j in range(neighbors.k):
                    var edge = i * neighbors.k + j
                    _connect(graph, n, i, neighbors.indices[edge], Float64(distances.unsafe_get(edge)))
            edges = len(neighbors.indices)
        else:
            var neighbors = nbrs.radius_neighbors()
            ref distances = neighbors.distances.value()
            for i in range(n):
                for edge in range(neighbors.offsets[i], neighbors.offsets[i + 1]):
                    _connect(graph, n, i, neighbors.indices[edge], Float64(distances.unsafe_get(edge)))
            edges = len(neighbors.indices)

        var components = _repair_components(graph, X, nbrs.effective_metric_params_)
        var warnings = List[FitWarning]()
        if components > 1:
            var message = (
                "The number of connected components of the neighbors graph is "
                + String(components)
                + " > 1. Completing the graph to fit Isomap might be slow. "
                + "Increase the number of neighbors to avoid this issue."
            )
            warnings.append(FitWarning(category="UserWarning", message=message))
        if params.path_method == "FW" or (params.path_method == "auto" and edges >= n * n // 4):
            _floyd_warshall(graph)
        else:
            graph = _dijkstra(graph)

        var paths = graph.astype[Self.dtype]()
        var kernel = paths.copy()
        _isomap_kernel(kernel)
        var kpca = KernelPCA[Self.dtype](
            params.n_components, kernel="precomputed", eigen_solver=params.eigen_solver,
            tol=params.tol, max_iter=params.max_iter, n_jobs=params.n_jobs,
        )
        var embedding = kpca.fit_transform(kernel)
        self.dist_matrix_ = paths^
        self.embedding_ = embedding^
        self.kernel_pca_ = kpca^
        self.n_features_in_ = nbrs.n_features_in_
        self.nbrs_ = nbrs^
        self.fit_warnings = warnings^
        self._uses_radius = Bool(params.radius)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.embedding_.value().copy()

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if not self.nbrs_:
            raise Error("NotFittedError: Isomap is not fitted")
        ref nbrs = self.nbrs_.value()
        ref paths = self.dist_matrix_.value()
        # Query validates rank and fitted feature count before allocating the kernel.
        var kernel: Self.Input
        if self._uses_radius:
            var neighbors = nbrs.radius_neighbors(X.copy())
            ref distances = neighbors.distances.value()
            kernel = zeros[Self.dtype](Shape(X.shape[0], nbrs.n_samples_fit_))
            for row in range(X.shape[0]):
                _extend_row(kernel, row, paths, neighbors.indices, distances,
                            neighbors.offsets[row], neighbors.offsets[row + 1])
        else:
            var neighbors = nbrs.kneighbors(X)
            ref distances = neighbors.distances.value()
            kernel = zeros[Self.dtype](Shape(X.shape[0], nbrs.n_samples_fit_))
            for row in range(X.shape[0]):
                _extend_row(kernel, row, paths, neighbors.indices, distances,
                            row * neighbors.k, (row + 1) * neighbors.k)
        _isomap_kernel(kernel)
        return self.kernel_pca_.value().transform(kernel)

    def reconstruction_error(self) raises -> Scalar[Self.dtype]:
        if not self.dist_matrix_:
            raise Error("NotFittedError: Isomap is not fitted")
        var centered = self.dist_matrix_.value().astype[DType.float64]()
        _isomap_kernel(centered)
        var centerer = KernelCenterer[DType.float64]()
        _ = centerer.fit(centered)
        centerer.transform_inplace(centered)

        # The direct residual avoids subtracting nearly equal squared norms.
        # Accumulate the reconstructed Gram entry without allocating a second matrix.
        ref embedding = self.embedding_.value()
        var n = centered.shape[0]
        var components = embedding.shape[1]
        var squared_error: Float64 = 0
        for i in range(n):
            for j in range(n):
                var reconstructed: Float64 = 0
                for k in range(components):
                    reconstructed += Float64(embedding.unsafe_get(i * components + k)) * Float64(
                        embedding.unsafe_get(j * components + k)
                    )
                var residual = centered.unsafe_get(i * n + j) - reconstructed
                squared_error += residual * residual
        return (sqrt(squared_error) / Float64(n)).cast[Self.dtype]()
