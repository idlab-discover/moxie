"""Dense FactorAnalysis: Gaussian latent-variable model fitted by SVD-based EM.

Follows sklearn.decomposition._factor_analysis (BSD-3-Clause; see
LICENSES/scikit-learn.txt), itself after Barber (2012), Algorithm 21.1.
Like the reference, fitting and learned state are Float64 whatever the input
dtype. Factorizations: shared SVD engine; inverses/log-determinants: mojo-linalg.
"""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, isfinite
from moxie.base import UnsupervisedEstimator
from moxie._random import RandomState
from ._base import DecompositionTransformer
from ._contracts import validate_matrix
from ._matrix import libm_log as log
from ._matrix import (get2, set2, get1, set1, as_float64, identity, column_means, column_variances,
    subtract_row, frobenius_squared, seeded_state, feature_names, symmetric_inverse, symmetric_logdet)
from ._svd_engine import randomized_svd, dense_svd

comptime _SMALL: Float64 = 1e-12
comptime _LOG_2PI: Float64 = 1.8378770664093453


@fieldwise_init
struct FactorAnalysisParams(Copyable, Movable):
    """Typed get_params/set_params record. `copy_input` is sklearn's `copy`."""
    var n_components: Optional[Int]
    var tol: Float64
    var copy_input: Bool
    var max_iter: Int
    var noise_variance_init: Optional[List[Float64]]
    var svd_method: String
    var iterated_power: Int
    var rotation: Optional[String]
    var random_state: Optional[Int]


def ortho_rotation(components: NDArray[DType.float64], method: String, *, tol: Float64 = 1e-6, max_iter: Int = 100) raises -> NDArray[DType.float64]:
    """sklearn _ortho_rotation: varimax/quartimax rotation of (p, k) loadings.

    Returns the rotated loadings transposed, shape (k, p). The update uses the
    polar factor U @ Vt of a k x k SVD, which is independent of SVD sign choices.
    """
    var p = components.shape[0]
    var k = components.shape[1]
    var rotation = identity[DType.float64](k)
    var previous: Float64 = 0
    for _ in range(max_iter):
        var rotated = components @ rotation
        var target = zeros[DType.float64](Shape(p, k))
        for j in range(k):
            var column_power: Float64 = 0
            if method == "varimax":
                for i in range(p):
                    column_power += get2(rotated, i, j) ** 2
                column_power /= Float64(p)
            for i in range(p):
                var r = get2(rotated, i, j)
                set2(target, i, j, r * r * r - (r * column_power if method == "varimax" else 0.0))
        var factors = dense_svd(components.T() @ target, flip_sign=False)
        rotation = factors.U @ factors.Vt
        var current: Float64 = 0
        for j in range(factors.singular_values.shape[0]):
            current += get1(factors.singular_values, j)
        if previous != 0 and current < previous * (1 + tol):
            break
        previous = current
    return (components @ rotation).T()


struct FactorAnalysis[dtype: DType = DType.float64](DecompositionTransformer, UnsupervisedEstimator):
    """Factor analysis on dense Float32/Float64 input; learned state is Float64.

    `svd_method="lapack"` uses the exact economy SVD; `"randomized"` (default)
    uses the shared randomized SVD with a RandomState stream shared across EM
    iterations, as in sklearn. `rotation` may be "varimax" or "quartimax".
    """
    comptime Params = FactorAnalysisParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[DType.float64]

    var n_components: Optional[Int]
    var tol: Float64
    var copy: Bool
    var max_iter: Int
    var noise_variance_init: Optional[List[Float64]]
    var svd_method: String
    var iterated_power: Int
    var rotation: Optional[String]
    var random_state: Optional[Int]
    var components_: Optional[NDArray[DType.float64]]
    var noise_variance_: Optional[NDArray[DType.float64]]
    var mean_: Optional[NDArray[DType.float64]]
    var loglike_: List[Float64]
    var n_iter_: Int
    var n_features_in_: Int

    def __init__(
        out self, n_components: Optional[Int] = None, *, tol: Float64 = 1e-2, copy: Bool = True,
        max_iter: Int = 1000, noise_variance_init: Optional[List[Float64]] = None,
        svd_method: String = "randomized", iterated_power: Int = 3,
        rotation: Optional[String] = None, random_state: Optional[Int] = 0,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components
        self.tol = tol
        self.copy = copy
        self.max_iter = max_iter
        self.noise_variance_init = noise_variance_init.copy()
        self.svd_method = svd_method
        self.iterated_power = iterated_power
        self.rotation = rotation
        self.random_state = random_state
        self.components_ = None
        self.noise_variance_ = None
        self.mean_ = None
        self.loglike_ = List[Float64]()
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_components, self.tol, self.copy, self.max_iter,
            self.noise_variance_init.copy(), self.svd_method, self.iterated_power,
            self.rotation, self.random_state)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.tol = params.tol
        self.copy = params.copy_input
        self.max_iter = params.max_iter
        self.noise_variance_init = params.noise_variance_init.copy()
        self.svd_method = params.svd_method
        self.iterated_power = params.iterated_power
        self.rotation = params.rotation
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_components: Optional[Optional[Int]] = None, tol: Optional[Float64] = None,
        copy: Optional[Bool] = None, max_iter: Optional[Int] = None,
        noise_variance_init: Optional[Optional[List[Float64]]] = None,
        svd_method: Optional[String] = None, iterated_power: Optional[Int] = None,
        rotation: Optional[Optional[String]] = None, random_state: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        """Partial update; omitted keywords keep their values."""
        if n_components: self.n_components = n_components.value()
        if tol: self.tol = tol.value()
        if copy: self.copy = copy.value()
        if max_iter: self.max_iter = max_iter.value()
        if noise_variance_init: self.noise_variance_init = noise_variance_init.value().copy()
        if svd_method: self.svd_method = svd_method.value()
        if iterated_power: self.iterated_power = iterated_power.value()
        if rotation: self.rotation = rotation.value()
        if random_state: self.random_state = random_state.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.n_components, tol=self.tol, copy=self.copy, max_iter=self.max_iter,
            noise_variance_init=self.noise_variance_init.copy(), svd_method=self.svd_method,
            iterated_power=self.iterated_power, rotation=self.rotation, random_state=self.random_state)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.components_)

    def _validate_params(self) raises:
        if self.n_components and self.n_components.value() < 0:
            raise Error("ValueError: FactorAnalysis n_components must be >= 0 or None, got " + String(self.n_components.value()))
        if not isfinite(self.tol) or self.tol < 0:
            raise Error("ValueError: FactorAnalysis tol must be a finite value >= 0")
        if self.max_iter < 1:
            raise Error("ValueError: FactorAnalysis max_iter must be >= 1, got " + String(self.max_iter))
        if self.svd_method != "randomized" and self.svd_method != "lapack":
            raise Error("ValueError: FactorAnalysis svd_method must be 'randomized' or 'lapack', got '" + self.svd_method + "'")
        if self.iterated_power < 0:
            raise Error("ValueError: FactorAnalysis iterated_power must be >= 0, got " + String(self.iterated_power))
        if self.rotation:
            var r = self.rotation.value()
            if r != "varimax" and r != "quartimax":
                raise Error("ValueError: FactorAnalysis rotation must be 'varimax', 'quartimax' or None, got '" + r + "'")

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        """Fit by EM on a private Float64 copy of X; X is never modified."""
        if not self.copy:
            raise Error("ValueError: FactorAnalysis copy=False requires fit_inplace with a mutable input")
        var work = as_float64(X)
        self._fit(X, work)
        return self

    def fit_inplace(mut self, mut X: Self.Input, y: Optional[Self.Target] = None) raises:
        """Fit; with copy=False and Float64 input, X is left centered like sklearn."""
        var work = as_float64(X)
        self._fit(X, work)
        comptime if Self.dtype == DType.float64:
            if not self.copy:
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        set2(X, i, j, get2(work, i, j))

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fit(X)
        return self.transform(X)

    def _fit(mut self, X: Self.Input, mut work: NDArray[DType.float64]) raises:
        self._validate_params()
        validate_matrix(X)
        var n = work.shape[0]
        var p = work.shape[1]
        var k_requested = self.n_components.value() if self.n_components else p
        var means = column_means(work)
        for i in range(n):
            for j in range(p):
                set2(work, i, j, get2(work, i, j) - get1(means, j))
        var nsqrt = sqrt(Float64(n))
        var llconst = Float64(p) * _LOG_2PI + Float64(k_requested)
        var variance = column_variances(work)
        var psi = zeros[DType.float64](Shape(p))
        if self.noise_variance_init:
            var init = self.noise_variance_init.value().copy()
            if len(init) != p:
                raise Error("ValueError: noise_variance_init dimension does not with number of features : "
                    + String(len(init)) + " != " + String(p))
            for j in range(p):
                set1(psi, j, init[j])
        else:
            for j in range(p):
                set1(psi, j, 1)
        var rng = RandomState(0)
        if self.svd_method == "randomized":
            rng = seeded_state(self.random_state, "FactorAnalysis")
        var loglike = List[Float64]()
        var old_ll: Float64 = 0
        var first = True
        var w = zeros[DType.float64](Shape(0, p))
        var iterations = 0
        var scaled = zeros[DType.float64](Shape(n, p))
        var sqrt_psi = zeros[DType.float64](Shape(p))
        for iteration in range(self.max_iter):
            iterations = iteration + 1
            for j in range(p):
                set1(sqrt_psi, j, sqrt(get1(psi, j)) + _SMALL)
            for i in range(n):
                for j in range(p):
                    set2(scaled, i, j, get2(work, i, j) / (get1(sqrt_psi, j) * nsqrt))
            var s = zeros[DType.float64](Shape(0))
            var vt = zeros[DType.float64](Shape(0, p))
            var unexplained: Float64
            if k_requested == 0:
                unexplained = frobenius_squared(scaled)
            elif self.svd_method == "lapack":
                var full = dense_svd(scaled, flip_sign=False)
                var kept = min(k_requested, full.singular_values.shape[0])
                s = zeros[DType.float64](Shape(kept))
                vt = zeros[DType.float64](Shape(kept, p))
                unexplained = 0
                for j in range(full.singular_values.shape[0]):
                    var value = get1(full.singular_values, j)
                    if j < kept:
                        set1(s, j, value)
                        for f in range(p): set2(vt, j, f, get2(full.Vt, j, f))
                    else:
                        unexplained += value * value
            else:
                var factors = randomized_svd(scaled, k_requested, rng, n_iter=self.iterated_power)
                s = factors.singular_values.copy()
                vt = factors.Vt.copy()
                unexplained = frobenius_squared(scaled)
                for j in range(s.shape[0]):
                    unexplained -= get1(s, j) ** 2
            var kept = s.shape[0]
            w = zeros[DType.float64](Shape(kept, p))
            var ll = llconst
            for j in range(kept):
                var squared = get1(s, j) ** 2
                ll += log(squared)
                var weight = sqrt(max(squared - 1.0, 0.0))
                for f in range(p):
                    set2(w, j, f, weight * get2(vt, j, f) * get1(sqrt_psi, f))
            ll += unexplained
            for f in range(p):
                ll += log(get1(psi, f))
            ll *= -Float64(n) / 2.0
            loglike.append(ll)
            if not first and ll - old_ll < self.tol:
                break
            first = False
            old_ll = ll
            for f in range(p):
                var loading: Float64 = 0
                for j in range(kept):
                    loading += get2(w, j, f) ** 2
                set1(psi, f, max(get1(variance, f) - loading, _SMALL))
        # The reference warns (ConvergenceWarning) when max_iter is exhausted;
        # native callers inspect n_iter_ == max_iter instead.
        if self.rotation and w.shape[0] > 0:
            w = ortho_rotation(w.T(), self.rotation.value())
        self.components_ = w^
        self.noise_variance_ = psi^
        self.mean_ = means^
        self.loglike_ = loglike^
        self.n_iter_ = iterations
        self.n_features_in_ = p

    def _check(self) raises:
        if not self.components_:
            raise Error("NotFittedError: This FactorAnalysis instance is not fitted yet. Call 'fit' first.")

    def _centered(self, X: Self.Input) raises -> NDArray[DType.float64]:
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: X has " + String(X.shape[1]) + " features, but FactorAnalysis is expecting "
                + String(self.n_features_in_) + " features as input.")
        return subtract_row(as_float64(X), self.mean_.value())

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Posterior mean of the latent variables (Bishop 12.66), shape (n, k)."""
        var centered = self._centered(X)
        var w = self.components_.value().copy()
        var k = w.shape[0]
        if k == 0:
            return zeros[DType.float64](Shape(X.shape[0], 0))
        var psi = self.noise_variance_.value().copy()
        var w_psi = zeros[DType.float64](Shape(k, self.n_features_in_))
        for j in range(k):
            for f in range(self.n_features_in_):
                set2(w_psi, j, f, get2(w, j, f) / get1(psi, f))
        var system = w_psi @ w.T()
        for j in range(k):
            set2(system, j, j, get2(system, j, j) + 1)
        var cov_z = symmetric_inverse(system, "FactorAnalysis latent covariance")
        return (centered @ w_psi.T()) @ cov_z

    def get_covariance(self) raises -> NDArray[DType.float64]:
        """Model covariance components_.T @ components_ + diag(noise_variance_)."""
        self._check()
        var w = self.components_.value().copy()
        var covariance = w.T() @ w if w.shape[0] > 0 else zeros[DType.float64](Shape(self.n_features_in_, self.n_features_in_))
        for f in range(self.n_features_in_):
            set2(covariance, f, f, get2(covariance, f, f) + get1(self.noise_variance_.value(), f))
        return covariance^

    def get_precision(self) raises -> NDArray[DType.float64]:
        """Model precision, by the matrix inversion lemma as in the reference."""
        self._check()
        var p = self.n_features_in_
        var psi = self.noise_variance_.value().copy()
        if self.n_components and self.n_components.value() == 0:
            var diagonal = zeros[DType.float64](Shape(p, p))
            for f in range(p): set2(diagonal, f, f, 1 / get1(psi, f))
            return diagonal^
        if self.n_components and self.n_components.value() == p:
            return symmetric_inverse(self.get_covariance(), "FactorAnalysis covariance")
        var w = self.components_.value().copy()
        var k = w.shape[0]
        var w_psi = zeros[DType.float64](Shape(k, p))
        for j in range(k):
            for f in range(p): set2(w_psi, j, f, get2(w, j, f) / get1(psi, f))
        var inner = w_psi @ w.T()
        for j in range(k): set2(inner, j, j, get2(inner, j, j) + 1)
        var precision = w.T() @ (symmetric_inverse(inner, "FactorAnalysis latent precision") @ w)
        for a in range(p):
            for b in range(p):
                set2(precision, a, b, -get2(precision, a, b) / (get1(psi, a) * get1(psi, b)))
            set2(precision, a, a, get2(precision, a, a) + 1 / get1(psi, a))
        return precision^

    def score_samples(self, X: Self.Input) raises -> NDArray[DType.float64]:
        """Gaussian log-likelihood of each sample under the fitted model."""
        var centered = self._centered(X)
        var precision = self.get_precision()
        var weighted = centered @ precision
        var constant = 0.5 * (Float64(self.n_features_in_) * _LOG_2PI - symmetric_logdet(precision))
        var result = zeros[DType.float64](Shape(X.shape[0]))
        for i in range(X.shape[0]):
            var quadratic: Float64 = 0
            for f in range(self.n_features_in_):
                quadratic += get2(centered, i, f) * get2(weighted, i, f)
            set1(result, i, -0.5 * quadratic - constant)
        return result^

    def score(self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Float64:
        """Average log-likelihood of the samples."""
        var values = self.score_samples(X)
        var total: Float64 = 0
        for i in range(values.shape[0]): total += get1(values, i)
        return total / Float64(values.shape[0])

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._check()
        return feature_names("factoranalysis", self.components_.value().shape[0], input_features, self.n_features_in_)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: FactorAnalysis supports native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")
