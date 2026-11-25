# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense locally linear embeddings and barycentric out-of-sample extension."""
from numojo import NDArray, Shape, zeros
from std.builtin.sort import sort
from std.math import abs, isfinite
from mojo_linalg import symmetric_eigh
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie.manifold._base import OutOfSampleEmbedding
from moxie.manifold._lle_reconstruction import Matrix, barycenter, reconstruction_operator
from moxie.manifold._plumbing import validate_matrix
from moxie.neighbors._distance import at
from moxie.neighbors._unsupervised import NearestNeighbors


def _float64_snapshot[dtype: DType](X: NDArray[dtype]) raises -> Matrix:
    comptime if dtype == DType.float64:
        ref source = rebind[Matrix](X)
        return source.contiguous()
    else:
        if X.is_c_contiguous():
            return X.astype[DType.float64]()
        return X.contiguous().astype[DType.float64]()


@fieldwise_init
struct LocallyLinearEmbeddingParams(Copyable, Movable):
    var n_neighbors: Int
    var n_components: Int
    var reg: Float64
    var eigen_solver: String
    var tol: Float64
    var max_iter: Int
    var method: String
    var hessian_tol: Float64
    var modified_tol: Float64
    var neighbors_algorithm: String
    var random_state: Optional[Int]
    var n_jobs: Optional[Int]

    def _validate(self, samples: Int, features: Int) raises:
        if self.n_neighbors < 1 or self.n_neighbors >= samples:
            raise Error("ValueError: expected 1 <= n_neighbors < n_samples")
        if self.n_components < 1 or self.n_components > features:
            raise Error("ValueError: output dimension must be positive and at most input dimension")
        if self.n_components >= samples:
            raise Error("ValueError: n_components must be smaller than n_samples")
        if self.max_iter < 1:
            raise Error("ValueError: max_iter must be positive")
        if not (isfinite(self.reg) and self.reg >= 0 and isfinite(self.tol) and self.tol >= 0
                and isfinite(self.hessian_tol) and self.hessian_tol >= 0
                and isfinite(self.modified_tol) and self.modified_tol >= 0):
            raise Error("ValueError: LLE regularization and tolerances must be finite and nonnegative")
        if self.random_state and (self.random_state.value() < 0 or self.random_state.value() > 4294967295):
            raise Error("ValueError: random_state must be between 0 and 2**32 - 1")
        if self.eigen_solver == "arpack":
            raise Error("NotImplementedError: LLE eigen_solver='arpack' is unavailable")
        if self.eigen_solver != "dense" and self.eigen_solver != "auto":
            raise Error("ValueError: eigen_solver must be auto, arpack, or dense")
        if self.method == "hessian":
            var d = self.n_components
            # n_neighbors > d * (d + 3) / 2, evaluated without integer overflow.
            if Float64(self.n_neighbors) <= Float64(d) * (Float64(d) + 3) / 2:
                raise Error("ValueError: Hessian LLE requires n_neighbors > n_components * (n_components + 3) / 2")
        elif self.method == "modified" or self.method == "ltsa":
            if self.n_neighbors < self.n_components:
                raise Error("ValueError: modified LLE and LTSA require n_neighbors >= n_components")
        elif self.method != "standard":
            raise Error("ValueError: method must be standard, hessian, modified, or ltsa")


@fieldwise_init
struct LocallyLinearEmbeddingResult(Movable):
    var embedding: Matrix
    var reconstruction_error: Float64

    def _take_embedding(deinit self) -> Matrix:
        return self.embedding^


@fieldwise_init
struct _SelectedEigenpair(Comparable, Copyable, Movable):
    var magnitude: Float64
    var index: Int

    def __lt__(self, other: Self) -> Bool:
        return self.magnitude < other.magnitude or (
            self.magnitude == other.magnitude and self.index < other.index
        )


def _smallest_embedding(operator: Matrix, dimensions: Int) raises -> LocallyLinearEmbeddingResult:
    var factor = symmetric_eigh(operator)
    var values = factor.eigenvalues()
    var vectors = factor.eigenvectors()
    var selected = List[_SelectedEigenpair](capacity=dimensions)
    var error: Float64 = 0
    for d in range(1, dimensions + 1):
        var value = values.unsafe_get(d)
        error += value
        selected.append(_SelectedEigenpair(abs(value), d))
    sort(Span(selected))

    var embedding = zeros[DType.float64](Shape(operator.shape[0], dimensions))
    for i in range(operator.shape[0]):
        for d in range(dimensions):
            embedding.unsafe_set(i * dimensions + d, at(vectors, i, selected[d].index))
    return LocallyLinearEmbeddingResult(embedding^, error)


struct LocallyLinearEmbedding[dtype: DType = DType.float64](
    OutOfSampleEmbedding, FeatureNamesOut, InputFeatureCount,
):
    """Dense LLE with deterministic dense eigensolving for auto and dense."""
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = Matrix
    comptime Params = LocallyLinearEmbeddingParams

    var _params: Self.Params
    var embedding_: Optional[Self.Transformed]
    var reconstruction_error_: Float64
    var n_features_in_: Int
    var nbrs_: Optional[NearestNeighbors[DType.float64]]
    var _training: Optional[Matrix]

    def __init__(
        out self, *, n_neighbors: Int = 5, n_components: Int = 2,
        reg: Float64 = 0.001, eigen_solver: String = "auto", tol: Float64 = 1e-6,
        max_iter: Int = 100, method: String = "standard", hessian_tol: Float64 = 1e-4,
        modified_tol: Float64 = 1e-12, neighbors_algorithm: String = "auto",
        random_state: Optional[Int] = None, n_jobs: Optional[Int] = None,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self._params = Self.Params(
            n_neighbors, n_components, reg, eigen_solver.copy(), tol, max_iter,
            method.copy(), hessian_tol, modified_tol, neighbors_algorithm.copy(),
            random_state, n_jobs,
        )
        self.embedding_ = None
        self.reconstruction_error_ = 0
        self.n_features_in_ = 0
        self.nbrs_ = None
        self._training = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self._params = params.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_neighbors: Optional[Int] = None,
        n_components: Optional[Int] = None, reg: Optional[Float64] = None,
        eigen_solver: Optional[String] = None, tol: Optional[Float64] = None,
        max_iter: Optional[Int] = None, method: Optional[String] = None,
        hessian_tol: Optional[Float64] = None, modified_tol: Optional[Float64] = None,
        neighbors_algorithm: Optional[String] = None,
        random_state: Optional[Optional[Int]] = None, n_jobs: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        if n_neighbors:
            self._params.n_neighbors = n_neighbors.value()
        if n_components:
            self._params.n_components = n_components.value()
        if reg:
            self._params.reg = reg.value()
        if eigen_solver:
            self._params.eigen_solver = eigen_solver.value().copy()
        if tol:
            self._params.tol = tol.value()
        if max_iter:
            self._params.max_iter = max_iter.value()
        if method:
            self._params.method = method.value().copy()
        if hessian_tol:
            self._params.hessian_tol = hessian_tol.value()
        if modified_tol:
            self._params.modified_tol = modified_tol.value()
        if neighbors_algorithm:
            self._params.neighbors_algorithm = neighbors_algorithm.value().copy()
        if random_state:
            self._params.random_state = random_state.value()
        if n_jobs:
            self._params.n_jobs = n_jobs.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result._params = self._params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.embedding_)

    def get_n_features_in(self) raises -> Int:
        if not self.embedding_:
            raise Error("AttributeError: LocallyLinearEmbedding has no n_features_in_ before fit")
        return self.n_features_in_

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.embedding_:
            raise Error("NotFittedError: LocallyLinearEmbedding is not fitted")
        if input_features and len(input_features.value()) != self.n_features_in_:
            raise Error("ValueError: input_features length differs from n_features_in_")
        var names = List[String]()
        for d in range(self.embedding_.value().shape[1]):
            names.append("locallylinearembedding" + String(d))
        return names^

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: LocallyLinearEmbedding supports native array output only")

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        validate_matrix(X)
        ref params = self._params
        var n = X.shape[0]
        var features = X.shape[1]
        params._validate(n, features)
        var k = params.n_neighbors
        var training = _float64_snapshot(X)
        var nbrs = NearestNeighbors[DType.float64](
            k, algorithm=params.neighbors_algorithm, n_jobs=params.n_jobs,
        )
        _ = nbrs.fit(training)
        # Match the explicit sklearn query and drop its first result, including
        # sklearn's convention for a training point with coincident neighbors.
        var nearest = nbrs.kneighbors(training, n_neighbors=k + 1, return_distance=False)
        var neighbors = List[Int](capacity=n * k)
        for row in range(n):
            for j in range(1, k + 1):
                neighbors.append(Int(nearest.indices.unsafe_get(row * (k + 1) + j)))
        var operator = reconstruction_operator(
            training, Span(neighbors), k, params.n_components, params.method,
            params.reg, params.hessian_tol, params.modified_tol,
        )
        var result = _smallest_embedding(operator, params.n_components)

        var error = result.reconstruction_error
        var embedding = result^._take_embedding()
        self.reconstruction_error_ = error
        self.embedding_ = embedding^
        self.n_features_in_ = features
        self.nbrs_ = nbrs^
        self._training = training^
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.embedding_.value().copy()

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if not self.embedding_:
            raise Error("NotFittedError: LocallyLinearEmbedding is not fitted")
        validate_matrix(X)
        comptime if Self.dtype == DType.float64:
            return self._transform(rebind[Matrix](X))
        else:
            return self._transform(_float64_snapshot(X))

    def _transform(self, query: Matrix) raises -> Self.Transformed:
        ref nbrs = self.nbrs_.value()
        ref training = self._training.value()
        ref embedding = self.embedding_.value()
        var nearest = nbrs.kneighbors(query, n_neighbors=self._params.n_neighbors, return_distance=False)
        var k = nearest.k
        var dimensions = embedding.shape[1]
        var result = zeros[DType.float64](Shape(query.shape[0], dimensions))
        var indices = List[Int](capacity=k)
        for row in range(query.shape[0]):
            indices.clear()
            for j in range(k):
                indices.append(Int(nearest.indices.unsafe_get(row * k + j)))
            var weights = barycenter(query, row, training, Span(indices), self._params.reg)
            for d in range(dimensions):
                var value: Float64 = 0
                for j in range(k):
                    value += weights.unsafe_get(j) * at(embedding, indices[j], d)
                result.unsafe_set(row * dimensions + d, value)
        return result^


def locally_linear_embedding[dtype: DType](
    X: NDArray[dtype], *, n_neighbors: Int, n_components: Int,
    reg: Float64 = 0.001, eigen_solver: String = "auto", tol: Float64 = 1e-6,
    max_iter: Int = 100, method: String = "standard", hessian_tol: Float64 = 1e-4,
    modified_tol: Float64 = 1e-12, random_state: Optional[Int] = None,
    n_jobs: Optional[Int] = None,
) raises -> LocallyLinearEmbeddingResult:
    var model = LocallyLinearEmbedding[dtype](
        n_neighbors=n_neighbors, n_components=n_components, reg=reg,
        eigen_solver=eigen_solver, tol=tol, max_iter=max_iter, method=method,
        hessian_tol=hessian_tol, modified_tol=modified_tol,
        random_state=random_state, n_jobs=n_jobs,
    )
    var embedding = model.fit_transform(X)
    return LocallyLinearEmbeddingResult(embedding^, model.reconstruction_error_)
