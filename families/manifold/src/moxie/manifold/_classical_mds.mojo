# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Classical scaling through the decomposition owner's dense eigensolver."""
from numojo import NDArray, zeros, Shape
from std.math import sqrt, abs, pow, isfinite
from moxie.manifold._base import EmbeddingEstimator
from moxie.manifold._plumbing import validate_matrix, validate_pairwise
from moxie.manifold._t_sne import _distance
from moxie.decomposition._svd_engine import symmetric_eigh
from moxie._dense import sum_axis2d


@fieldwise_init
struct ClassicalMDSParams(Copyable, Movable):
    var n_components: Int
    var metric: String
    var metric_params: Optional[Dict[String, Float64]]


struct ClassicalMDS[dtype: DType = DType.float64](EmbeddingEstimator):
    comptime Params = ClassicalMDSParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[DType.float64]
    var n_components: Int
    var metric: String
    var metric_params: Optional[Dict[String, Float64]]
    var embedding_: Optional[NDArray[DType.float64]]
    var eigenvalues_: Optional[NDArray[DType.float64]]
    var dissimilarity_matrix_: Optional[NDArray[Self.dtype]]
    var n_features_in_: Int

    def __init__(out self, n_components: Int = 2, *, metric: String = "euclidean", var metric_params: Optional[Dict[String, Float64]] = None):
        self.n_components = n_components
        self.metric = metric
        self.metric_params = metric_params^
        self.embedding_ = None
        self.eigenvalues_ = None
        self.dissimilarity_matrix_ = None
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return ClassicalMDSParams(self.n_components, self.metric, self.metric_params.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.metric = params.metric
        self.metric_params = params.metric_params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_components: Optional[Int] = None, metric: Optional[String] = None) raises -> ref[self] Self:
        if n_components:
            self.n_components = n_components.value()
        if metric:
            self.metric = metric.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.n_components, metric=self.metric, metric_params=self.metric_params.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.embedding_)

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _ = self.fit_transform(X)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        validate_matrix(X)
        if self.n_components < 1:
            raise Error("ValueError: n_components must be positive")
        if self.metric != "precomputed" and self.metric != "euclidean" and self.metric != "sqeuclidean" and self.metric != "manhattan" and self.metric != "cityblock" and self.metric != "cosine" and self.metric != "minkowski":
            raise Error("ValueError: unsupported ClassicalMDS metric")
        var power: Float64 = 2
        if self.metric_params:
            for entry in self.metric_params.value().items():
                if self.metric != "minkowski" or entry.key != "p":
                    raise Error("ValueError: unsupported metric_params entry")
                power = entry.value
        if not isfinite(power) or power <= 0:
            raise Error("ValueError: Minkowski p must be finite and positive")
        var n = X.shape[0]
        var distances = zeros[Self.dtype](Shape(n, n))
        if self.metric == "precomputed":
            validate_pairwise(X, symmetric=False)
            for i in range(n):
                for j in range(n):
                    var a = Float64(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])))
                    var b = Float64(X.unsafe_get(j * Int(X.strides[0]) + i * Int(X.strides[1])))
                    if abs(a - b) > 1e-10 + 1e-5 * abs(b):
                        raise Error("ValueError: array must be symmetric")
            distances = X.copy()
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    var d = _distance(X, i, j, self.metric)
                    if self.metric == "minkowski":
                        d = 0
                        for c in range(X.shape[1]):
                            var a = Float64(X.unsafe_get(i * Int(X.strides[0]) + c * Int(X.strides[1])))
                            var b = Float64(X.unsafe_get(j * Int(X.strides[0]) + c * Int(X.strides[1])))
                            d += pow(abs(a - b), power)
                        d = pow(d, 1 / power)
                    if self.metric == "euclidean":
                        d = sqrt(d)
                    distances.unsafe_set(i * n + j, d.cast[Self.dtype]())
                    distances.unsafe_set(j * n + i, d.cast[Self.dtype]())
        var gram = zeros[DType.float64](Shape(n, n))
        for i in range(n):
            for j in range(n):
                var d = distances.unsafe_get(i * Int(distances.strides[0]) + j * Int(distances.strides[1]))
                gram.unsafe_set(i * n + j, Float64(d * d))
        var sums = sum_axis2d(gram, 0)
        for i in range(n):
            for j in range(n):
                gram.unsafe_set(i * n + j, gram.unsafe_get(i * n + j) - sums.unsafe_get(j) / Float64(n))
        sums = sum_axis2d(gram, 1)
        for i in range(n):
            for j in range(n):
                gram.unsafe_set(i * n + j, -0.5 * (gram.unsafe_get(i * n + j) - sums.unsafe_get(i) / Float64(n)))
        # SciPy eigh reads the lower triangle, including centering roundoff.
        for i in range(n):
            for j in range(i):
                var value = gram.unsafe_get(i * n + j)
                gram.unsafe_set(i * n + j, value)
                gram.unsafe_set(j * n + i, value)
        var eigen = symmetric_eigh(gram)
        var count = min(n, self.n_components)
        var values = zeros[DType.float64](Shape(count))
        var coordinates = zeros[DType.float64](Shape(n, count))
        for j in range(count):
            var value = eigen.eigenvalues.unsafe_get(j)
            values.unsafe_set(j, value)
            for i in range(n):
                coordinates.unsafe_set(i * count + j, sqrt(value) * eigen.eigenvectors.unsafe_get(i * n + j))
        self.n_features_in_ = X.shape[1]
        self.dissimilarity_matrix_ = distances^
        self.eigenvalues_ = values^
        self.embedding_ = coordinates.copy()
        return coordinates^
