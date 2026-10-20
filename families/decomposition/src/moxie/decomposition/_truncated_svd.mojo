"""Dense TruncatedSVD (latent semantic analysis) without centering.

Follows sklearn.decomposition._truncated_svd (BSD-3-Clause; see
LICENSES/scikit-learn.txt). Factorizations come from the shared SVD engine.
"""
from numojo import NDArray, Shape, zeros
from std.math import isfinite
from moxie.base import UnsupervisedEstimator
from ._base import InvertibleDecomposition
from ._contracts import validate_matrix
from ._matrix import get2, set2, get1, set1, column_variances, seeded_state, feature_names
from ._svd_engine import randomized_svd, svd_flip, truncated_svd


@fieldwise_init
struct TruncatedSVDParams(Copyable, Movable):
    """Typed get_params/set_params record; field names follow sklearn."""
    var n_components: Int
    var algorithm: String
    var n_iter: Int
    var n_oversamples: Int
    var power_iteration_normalizer: String
    var random_state: Optional[Int]
    var tol: Float64


struct TruncatedSVD[dtype: DType = DType.float64](InvertibleDecomposition, UnsupervisedEstimator):
    """Dimensionality reduction by truncated SVD of the uncentered data.

    `algorithm="randomized"` (default) uses the shared Halko range finder with
    sklearn-compatible seeded draws. `algorithm="arpack"` is the dense
    equivalent of scipy's ARPACK `svds`: it returns the exact leading singular
    triplets (Jacobi SVD), which satisfies every requested `tol`.
    Fit mutates the estimator and returns nothing.
    """
    comptime Params = TruncatedSVDParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]

    var n_components: Int
    var algorithm: String
    var n_iter: Int
    var n_oversamples: Int
    var power_iteration_normalizer: String
    var random_state: Optional[Int]
    var tol: Float64
    var components_: Optional[NDArray[Self.dtype]]
    var explained_variance_: Optional[NDArray[Self.dtype]]
    var explained_variance_ratio_: Optional[NDArray[Self.dtype]]
    var singular_values_: Optional[NDArray[Self.dtype]]
    var n_features_in_: Int

    def __init__(
        out self, n_components: Int = 2, *, algorithm: String = "randomized", n_iter: Int = 5,
        n_oversamples: Int = 10, power_iteration_normalizer: String = "auto",
        random_state: Optional[Int] = None, tol: Float64 = 0.0,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.tol = tol
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_components, self.algorithm, self.n_iter, self.n_oversamples,
            self.power_iteration_normalizer, self.random_state, self.tol)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.algorithm = params.algorithm
        self.n_iter = params.n_iter
        self.n_oversamples = params.n_oversamples
        self.power_iteration_normalizer = params.power_iteration_normalizer
        self.random_state = params.random_state
        self.tol = params.tol
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_components: Optional[Int] = None, algorithm: Optional[String] = None,
        n_iter: Optional[Int] = None, n_oversamples: Optional[Int] = None,
        power_iteration_normalizer: Optional[String] = None,
        random_state: Optional[Optional[Int]] = None, tol: Optional[Float64] = None,
    ) raises -> ref[self] Self:
        """Partial update; an omitted keyword keeps its value (use Optional(None) to clear random_state)."""
        if n_components: self.n_components = n_components.value()
        if algorithm: self.algorithm = algorithm.value()
        if n_iter: self.n_iter = n_iter.value()
        if n_oversamples: self.n_oversamples = n_oversamples.value()
        if power_iteration_normalizer: self.power_iteration_normalizer = power_iteration_normalizer.value()
        if random_state: self.random_state = random_state.value()
        if tol: self.tol = tol.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.n_components, algorithm=self.algorithm, n_iter=self.n_iter,
            n_oversamples=self.n_oversamples, power_iteration_normalizer=self.power_iteration_normalizer,
            random_state=self.random_state, tol=self.tol)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.components_)

    def _validate_params(self) raises:
        if self.n_components < 1:
            raise Error("ValueError: TruncatedSVD n_components must be >= 1, got " + String(self.n_components))
        if self.algorithm != "randomized" and self.algorithm != "arpack":
            raise Error("ValueError: TruncatedSVD algorithm must be 'arpack' or 'randomized', got '" + self.algorithm + "'")
        if self.n_iter < 0:
            raise Error("ValueError: TruncatedSVD n_iter must be >= 0, got " + String(self.n_iter))
        if self.n_oversamples < 1:
            raise Error("ValueError: TruncatedSVD n_oversamples must be >= 1, got " + String(self.n_oversamples))
        var normalizer = self.power_iteration_normalizer
        if normalizer != "auto" and normalizer != "QR" and normalizer != "LU" and normalizer != "none":
            raise Error("ValueError: TruncatedSVD power_iteration_normalizer must be auto, QR, LU or none, got '" + normalizer + "'")
        if not isfinite(self.tol) or self.tol < 0:
            raise Error("ValueError: TruncatedSVD tol must be a finite value >= 0")

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _ = self.fit_transform(X)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        """Fit on X (n_samples, n_features >= 2) and return X projected on components_."""
        self._validate_params()
        validate_matrix(X)
        var n = X.shape[0]
        var p = X.shape[1]
        if p < 2:
            raise Error("ValueError: TruncatedSVD requires at least 2 features, got " + String(p))
        var rng = seeded_state(self.random_state, "TruncatedSVD")
        var u: NDArray[Self.dtype]
        var sigma: NDArray[Self.dtype]
        var vt: NDArray[Self.dtype]
        if self.algorithm == "arpack":
            if self.n_components >= min(n, p):
                raise Error("ValueError: TruncatedSVD algorithm='arpack' requires n_components < min(n_samples, n_features)="
                    + String(min(n, p)) + ", got " + String(self.n_components))
            # ARPACK draws its start vector v0 = uniform(-1, 1, min(X.shape)).
            for _ in range(min(n, p)):
                _ = rng.uniform(-1, 1)
            var factors = truncated_svd(X, self.n_components)
            u = factors.U.copy()
            sigma = factors.singular_values.copy()
            vt = factors.Vt.copy()
        else:
            if self.n_components > p:
                raise Error("ValueError: n_components(" + String(self.n_components) + ") must be <= n_features(" + String(p) + ").")
            var factors = randomized_svd(X, self.n_components, rng, n_iter=self.n_iter,
                n_oversamples=self.n_oversamples, power_iteration_normalizer=self.power_iteration_normalizer,
                flip_sign=False)
            u = factors.U.copy()
            sigma = factors.singular_values.copy()
            vt = factors.Vt.copy()
        svd_flip(u, vt, u_based_decision=False)
        var k = sigma.shape[0]
        var transformed: NDArray[Self.dtype]
        if self.algorithm == "randomized" or self.tol > 0:
            # The approximation X ~ U S Vt is inexact, so project explicitly.
            transformed = X @ vt.T()
        else:
            transformed = zeros[Self.dtype](Shape(n, k))
            for i in range(n):
                for j in range(k):
                    set2(transformed, i, j, get2(u, i, j) * get1(sigma, j))
        var explained = column_variances(transformed)
        var feature_variance = column_variances(X)
        var full_var: Float64 = 0
        for j in range(p):
            full_var += get1(feature_variance, j)
        var ratios = zeros[Self.dtype](Shape(k))
        for j in range(k):
            set1(ratios, j, get1(explained, j) / full_var)
        self.components_ = vt^
        self.explained_variance_ = explained^
        self.explained_variance_ratio_ = ratios^
        self.singular_values_ = sigma^
        self.n_features_in_ = p
        return transformed^

    def _check(self) raises:
        if not self.components_:
            raise Error("NotFittedError: This TruncatedSVD instance is not fitted yet. Call 'fit' first.")

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Project X (n, n_features_in_) onto components_: X @ components_.T."""
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: X has " + String(X.shape[1]) + " features, but TruncatedSVD is expecting "
                + String(self.n_features_in_) + " features as input.")
        return X @ self.components_.value().T()

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        """Map (n, n_components_) scores back to feature space: X @ components_."""
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.components_.value().shape[0]:
            raise Error("ValueError: inverse_transform expects " + String(self.components_.value().shape[0])
                + " columns, got " + String(X.shape[1]))
        return X @ self.components_.value()

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._check()
        return feature_names("truncatedsvd", self.components_.value().shape[0], input_features, self.n_features_in_)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: TruncatedSVD supports native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")
