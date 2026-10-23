"""Dense CPU PCA; formulas follow sklearn.decomposition (BSD-3-Clause).

Public arrays use NuMojo. A numerical backend supplies factorization and matrix
products (mojo-linalg by default; an opt-in static Numax backend is described
in docs/pca.md). Scalar math uses the Mojo standard library, except `log`, which
goes through the platform libm as NumPy does.
"""
from std.utils import Variant
from std.memory import bitcast
from std.math import abs, isfinite, isnan, sqrt, lgamma
from std.sys import llvm_intrinsic
from numojo import NDArray, Shape, zeros, mean
from ._pca_backend import PCABackend, RuntimePCABackend
from moxie.base import UnsupervisedEstimator
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie._random import entropy_state, RandomState
from ._base import InvertibleDecomposition
from ._contracts import validate_matrix
from ._svd_engine import randomized_svd, svd_flip, center_matrix, truncated_svd

comptime ComponentCount = Variant[Int, Float64, String, NoneType]
comptime IteratedPower = Variant[Int, String]


@fieldwise_init
struct PCAParams(Copyable, Movable):
    var n_components: ComponentCount
    var copy_input: Bool
    var whiten: Bool
    var svd_solver: String
    var tol: Float64
    var iterated_power: IteratedPower
    var n_oversamples: Int
    var power_iteration_normalizer: String
    var random_state: Optional[Int]


def _sqrt(x: Float64) -> Float64:
    return sqrt(x)


def _log(x: Float64) -> Float64:
    """Platform libm log; std.math.log is off by up to ~1e-10 (see _matrix.libm_log)."""
    return llvm_intrinsic["llvm.log", Float64, has_side_effect=False](x)


def _get[dtype: DType](a: NDArray[dtype], i: Int, j: Int) raises -> Float64:
    return Float64(a.unsafe_get(i * Int(a.strides[0]) + j * Int(a.strides[1])))


def _set[dtype: DType](mut a: NDArray[dtype], i: Int, j: Int, value: Float64) raises:
    a.unsafe_set(i * Int(a.strides[0]) + j * Int(a.strides[1]), SIMD[dtype, 1](value))


def _infer_dimension[dtype: DType](spectrum: NDArray[dtype], n_samples: Int) raises -> Int:
    """Minka's likelihood, as in sklearn's _assess/_infer_dimension."""
    var n = spectrum.size
    var best = 0
    var best_ll = -bitcast[DType.float64](UInt64(0x7ff0000000000000))
    comptime pi = 3.14159265358979323846
    for rank in range(1, n):
        if Float64(spectrum.unsafe_get(rank - 1)) < 1e-15:
            continue
        var pu = -Float64(rank) * _log(2.0)
        for i in range(1, rank + 1):
            var v = Float64(n - i + 1) / 2
            pu += lgamma(v) - _log(pi) * v
        var pl: Float64 = 0
        for i in range(rank):
            pl += _log(Float64(spectrum.unsafe_get(i)))
        pl *= -Float64(n_samples) / 2
        var noise: Float64 = 0
        for i in range(rank, n):
            noise += Float64(spectrum.unsafe_get(i))
        noise = max(1e-15, noise / Float64(n - rank))
        var pv = -_log(noise) * Float64(n_samples * (n - rank)) / 2
        var m = Float64(n * rank) - Float64(rank) * Float64(rank + 1) / 2
        var pp = _log(2 * pi) * (m + Float64(rank)) / 2
        var pa: Float64 = 0
        for i in range(rank):
            var si = Float64(spectrum.unsafe_get(i))
            for j in range(i + 1, n):
                var sj = Float64(spectrum.unsafe_get(j))
                var adjusted = sj if j < rank else noise
                pa += _log((si - sj) * (1 / adjusted - 1 / si)) + _log(Float64(n_samples))
        var ll = pu + pl + pv + pp - pa / 2 - Float64(rank) * _log(Float64(n_samples)) / 2
        if ll > best_ll:
            best_ll = ll
            best = rank
    return best


struct PCA[dtype: DType = DType.float64, Backend: PCABackend = RuntimePCABackend](InvertibleDecomposition, UnsupervisedEstimator, FeatureNamesOut, InputFeatureCount):
    """PCA on dense Float32/Float64 matrices; Backend may fix training dimensions.

    full, covariance_eigh, arpack and randomized are supported, including auto
    selection. Dense "arpack" computes the exact leading factors (see docs/pca.md).
    Fit mutates self and returns a borrow of it (fluent chaining).
    """
    comptime Params = PCAParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]

    var n_components: ComponentCount
    var copy: Bool
    var whiten: Bool
    var svd_solver: String
    var tol: Float64
    var iterated_power: IteratedPower
    var n_oversamples: Int
    var power_iteration_normalizer: String
    var random_state: Optional[Int]
    var components_: Optional[NDArray[Self.dtype]]
    var explained_variance_: Optional[NDArray[Self.dtype]]
    var explained_variance_ratio_: Optional[NDArray[Self.dtype]]
    var singular_values_: Optional[NDArray[Self.dtype]]
    var mean_: Optional[NDArray[Self.dtype]]
    var n_components_: Int
    var n_samples_: Int
    var n_features_in_: Int
    var noise_variance_: Float64
    var feature_names_in_: Optional[List[String]]
    var _fit_svd_solver: String

    def __init__(
        out self, n_components: ComponentCount = NoneType(), *, copy: Bool = True,
        whiten: Bool = False, svd_solver: String = "auto", tol: Float64 = 0.0,
        iterated_power: IteratedPower = "auto", n_oversamples: Int = 10,
        power_iteration_normalizer: String = "auto", random_state: Optional[Int] = None,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components.copy()
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power.copy()
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = 0
        self.n_samples_ = 0
        self.n_features_in_ = 0
        self.noise_variance_ = 0
        self.feature_names_in_ = None
        self._fit_svd_solver = ""

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_components.copy(), self.copy, self.whiten,
            self.svd_solver, self.tol, self.iterated_power.copy(),
            self.n_oversamples, self.power_iteration_normalizer, self.random_state)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_components = params.n_components.copy()
        self.copy = params.copy_input
        self.whiten = params.whiten
        self.svd_solver = params.svd_solver
        self.tol = params.tol
        self.iterated_power = params.iterated_power.copy()
        self.n_oversamples = params.n_oversamples
        self.power_iteration_normalizer = params.power_iteration_normalizer
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_components: Optional[ComponentCount] = None,
        copy: Optional[Bool] = None, whiten: Optional[Bool] = None,
        svd_solver: Optional[String] = None, tol: Optional[Float64] = None,
        iterated_power: Optional[IteratedPower] = None,
        n_oversamples: Optional[Int] = None,
        power_iteration_normalizer: Optional[String] = None,
        random_state: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        if n_components: self.n_components = n_components.value().copy()
        if copy: self.copy = copy.value()
        if whiten: self.whiten = whiten.value()
        if svd_solver: self.svd_solver = svd_solver.value()
        if tol: self.tol = tol.value()
        if iterated_power: self.iterated_power = iterated_power.value().copy()
        if n_oversamples: self.n_oversamples = n_oversamples.value()
        if power_iteration_normalizer: self.power_iteration_normalizer = power_iteration_normalizer.value()
        if random_state: self.random_state = random_state.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.n_components.copy(), copy=self.copy, whiten=self.whiten,
            svd_solver=self.svd_solver, tol=self.tol,
            iterated_power=self.iterated_power.copy(), n_oversamples=self.n_oversamples,
            power_iteration_normalizer=self.power_iteration_normalizer, random_state=self.random_state)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.components_)

    def _check(self) raises:
        if not self.components_:
            raise Error("NotFittedError: PCA has not been fitted")

    def _check_input(self, X: Self.Input) raises:
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: X has the wrong feature count")

    def _fit(mut self, mut X: Self.Input) raises -> Self.Transformed:
        validate_matrix(X)
        Self.Backend.validate_fit(X)
        var n = X.shape[0]
        var p = X.shape[1]
        var limit = min(n, p)
        var k = limit
        var fraction: Float64 = 0
        var mle = False
        if not self.n_components.isa[NoneType]():
            var count = self.n_components.copy()
            if count.isa[Int]():
                k = count[Int]
                if k < 0 or k > limit: raise Error("ValueError: n_components out of range")
            elif count.isa[Float64]():
                fraction = count[Float64]
                if not (fraction > 0 and fraction < 1):
                    raise Error("ValueError: fractional n_components must be in (0, 1)")
            else:
                if count[String] != "mle": raise Error("ValueError: n_components must be 'mle'")
                mle = True
                if n < p: raise Error("ValueError: mle requires n_samples >= n_features")
        if not isfinite(self.tol) or self.tol < 0 or self.n_oversamples < 1:
            raise Error("ValueError: invalid tol or n_oversamples")
        var iterations = -1
        if self.iterated_power.isa[Int]():
            iterations = self.iterated_power[Int]
            if iterations < 0: raise Error("ValueError: iterated_power must be nonnegative")
        elif self.iterated_power[String] != "auto":
            raise Error("ValueError: iterated_power must be 'auto' or an integer")
        var normalizer = self.power_iteration_normalizer
        if normalizer != "auto" and normalizer != "none" and normalizer != "QR" and normalizer != "LU":
            raise Error("ValueError: invalid power_iteration_normalizer")
        if self.random_state:
            if self.random_state.value() < 0 or UInt64(self.random_state.value()) > 4294967295:
                raise Error("ValueError: random_state must be a uint32 seed")
        var solver = self.svd_solver
        if solver == "auto":
            if p <= 1000 and n >= 10 * p: solver = "covariance_eigh"
            elif max(n, p) <= 500 or mle: solver = "full"
            elif fraction == 0 and k >= 1 and Float64(k) < 0.8 * Float64(limit): solver = "randomized"
            else: solver = "full"
        if solver != "full" and solver != "covariance_eigh" and solver != "randomized" and solver != "arpack":
            raise Error("ValueError: invalid svd_solver " + solver)
        var truncated = solver == "randomized" or solver == "arpack"
        # sklearn: an omitted count means min(X.shape) - 1 for ARPACK only.
        if solver == "arpack" and self.n_components.isa[NoneType](): k = limit - 1
        if truncated and (mle or fraction > 0 or k < 1):
            raise Error("ValueError: PCA svd_solver='" + solver + "' requires an integer n_components >= 1")
        if solver == "arpack" and k >= limit:
            raise Error("ValueError: PCA svd_solver='arpack' requires n_components < min(n_samples, n_features)="
                + String(limit) + ", got " + String(k))

        var centered_and_mean = center_matrix(X)
        var centered = centered_and_mean[0].copy()
        var means = centered_and_mean[1].copy()
        if not self.copy and solver != "covariance_eigh":
            var view = X.view()
            for i in range(n):
                for j in range(p): _set(view, i, j, _get(centered, i, j))
        var u = zeros[Self.dtype](Shape(n, limit))
        var values = zeros[Self.dtype](Shape(limit))
        var vt = zeros[Self.dtype](Shape(limit, p))
        var variance = zeros[Self.dtype](Shape(limit))
        var total_var: Float64 = 0
        if solver == "covariance_eigh":
            u = zeros[Self.dtype](Shape(n, p))
            values = zeros[Self.dtype](Shape(p))
            vt = zeros[Self.dtype](Shape(p, p))
            variance = zeros[Self.dtype](Shape(p))
            var covariance = Self.Backend.matmul(X.T(), X)
            for i in range(p):
                for j in range(p):
                    var v = (_get(covariance, i, j) - Float64(n) * Float64(means.unsafe_get(i)) * Float64(means.unsafe_get(j))) / Float64(n - 1)
                    _set(covariance, i, j, v)
            var eigen = Self.Backend.eigh(covariance)
            var ev = eigen[0].copy()
            var vectors = eigen[1].copy()
            # Backends return ascending eigenvalues; PCA orders them descending.
            for j in range(p):
                var v = max(0.0, Float64(ev.unsafe_get(p - 1 - j)))
                variance.unsafe_set(j, SIMD[Self.dtype, 1](v))
                values.unsafe_set(j, SIMD[Self.dtype, 1](_sqrt(v * Float64(n - 1))))
                for f in range(p): _set(vt, j, f, _get(vectors, f, p - 1 - j))
        else:
            if solver == "full":
                var factors = Self.Backend.full(centered)
                u = factors[0].copy()
                values = factors[1].copy()
                vt = factors[2].copy()
            elif solver == "arpack":
                # ARPACK converges to the leading singular triplets; the dense
                # equivalent computes them exactly (tol is met trivially).
                var factors = truncated_svd(centered, k)
                u = factors.U.copy()
                values = factors.singular_values.copy()
                vt = factors.Vt.copy()
            else:
                var rng = entropy_state()
                if self.random_state: rng = RandomState(self.random_state.value())
                var factors = randomized_svd(centered, k, rng,
                    n_oversamples=self.n_oversamples, n_iter=iterations,
                    power_iteration_normalizer=normalizer, flip_sign=False)
                u = factors.U.copy()
                values = factors.singular_values.copy()
                vt = factors.Vt.copy()
            if truncated:
                variance = zeros[Self.dtype](Shape(k))
                for i in range(n):
                    for j in range(p):
                        var v = _get(centered, i, j)
                        total_var += v * v / Float64(n - 1)
                # sklearn's truncated copy=False path leaves squared residuals.
                if not self.copy:
                    var view = X.view()
                    for i in range(n):
                        for j in range(p):
                            var v = _get(centered, i, j)
                            _set(view, i, j, v * v)
            for j in range(values.size):
                var s = values.unsafe_get(j)
                variance.unsafe_set(j, s * s / SIMD[Self.dtype, 1](n - 1))
        svd_flip(u, vt, u_based_decision=False)
        if not truncated:
            for j in range(variance.size): total_var += Float64(variance.unsafe_get(j))
        if mle: k = _infer_dimension(variance, n)
        elif fraction > 0:
            var cumulative: Float64 = 0
            k = 0
            while k < variance.size:
                cumulative += Float64(variance.unsafe_get(k)) / total_var
                k += 1
                if cumulative > fraction: break
        var noise: Float64 = 0
        if k < limit:
            if truncated:
                noise = total_var
                for j in range(k): noise -= Float64(variance.unsafe_get(j))
            else:
                for j in range(k, variance.size): noise += Float64(variance.unsafe_get(j))
            noise /= Float64(limit - k) if truncated else Float64(variance.size - k)
        var components = zeros[Self.dtype](Shape(k, p))
        var explained = zeros[Self.dtype](Shape(k))
        var ratios = zeros[Self.dtype](Shape(k))
        var singular = zeros[Self.dtype](Shape(k))
        var transformed = zeros[Self.dtype](Shape(n, k))
        for j in range(k):
            for f in range(p): _set(components, j, f, _get(vt, j, f))
            explained.unsafe_set(j, variance.unsafe_get(j))
            ratios.unsafe_set(j, SIMD[Self.dtype, 1](Float64(variance.unsafe_get(j)) / total_var))
            singular.unsafe_set(j, values.unsafe_get(j))
            if solver != "covariance_eigh":
                var scale = _sqrt(Float64(n - 1)) if self.whiten else Float64(values.unsafe_get(j))
                for i in range(n): _set(transformed, i, j, _get(u, i, j) * scale)
        self.components_ = components^
        self.explained_variance_ = explained^
        self.explained_variance_ratio_ = ratios^
        self.singular_values_ = singular^
        self.mean_ = means^
        self.n_components_ = k
        self.n_samples_ = n
        self.n_features_in_ = p
        self.noise_variance_ = noise
        self._fit_svd_solver = solver
        if solver == "covariance_eigh": return self.transform(X)
        return transformed^

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        if not self.copy:
            raise Error("ValueError: copy=False requires fit_inplace with a mutable input")
        var owned = X.copy()
        _ = self._fit(owned)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        if not self.copy:
            raise Error("ValueError: copy=False requires fit_transform_inplace with a mutable input")
        var owned = X.copy()
        return self._fit(owned)

    def fit_inplace(mut self, mut X: Self.Input, y: Optional[Self.Target] = None) raises:
        _ = self._fit(X)

    def fit_transform_inplace(mut self, mut X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        return self._fit(X)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check_input(X)
        return _pca_transform[Self.dtype, Self.Backend](X, self.components_.value(), self.mean_.value(), self.explained_variance_.value(), self.whiten)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        self._check()
        return _pca_inverse_transform[Self.dtype, Self.Backend](X, self.components_.value(), self.mean_.value(), self.explained_variance_.value(), self.whiten)

    def get_covariance(self) raises -> Self.Input:
        self._check()
        return _pca_get_covariance[Self.dtype, Self.Backend](self.components_.value(), self.explained_variance_.value(), self.whiten, self.noise_variance_)

    def get_precision(self) raises -> Self.Input:
        self._check()
        var p = self.n_features_in_
        var identity = zeros[Self.dtype](Shape(p, p))
        for i in range(p): _set(identity, i, i, 1)
        if self.n_components_ == 0:
            for i in range(p):
                for j in range(p): _set(identity, i, j, _get(identity, i, j) / self.noise_variance_)
            return identity^
        return Self.Backend.inverse(self.get_covariance())

    def score_samples(self, X: Self.Input) raises -> NDArray[Self.dtype]:
        self._check_input(X)
        var precision = self.get_precision()
        var determinant = Self.Backend.slogdet(precision)
        var centered = X - self.mean_.value()
        var weighted = Self.Backend.matmul(centered, precision)
        var result = zeros[Self.dtype](Shape(X.shape[0]))
        var logdet = determinant[1] if determinant[0] > 0 else -bitcast[DType.float64](UInt64(0x7ff0000000000000))
        var constant = Float64(self.n_features_in_) * _log(2 * 3.14159265358979323846) - logdet
        for i in range(X.shape[0]):
            var squared: Float64 = 0
            for j in range(self.n_features_in_): squared += _get(centered, i, j) * _get(weighted, i, j)
            result.unsafe_set(i, SIMD[Self.dtype, 1](-0.5 * (squared + constant)))
        return result^

    def score(self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Float64:
        var values = self.score_samples(X)
        var total: Float64 = 0
        for i in range(values.size): total += Float64(values.unsafe_get(i))
        return total / Float64(values.size)

    def get_n_features_in(self) raises -> Int:
        """sklearn's `n_features_in_` attribute, which exists only after fit."""
        if not self.__sklearn_is_fitted__():
            raise Error("AttributeError: PCA has no n_features_in_ before fit")
        return self.n_features_in_

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._check()
        if input_features:
            if len(input_features.value()) != self.n_features_in_:
                raise Error("ValueError: input_features length differs from n_features_in_")
        var names = List[String]()
        for i in range(self.n_components_): names.append("pca" + String(i))
        return names^

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: PCA supports native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")

def _pca_transform[dtype: DType, Backend: PCABackend, mean_dtype: DType](X: NDArray[dtype], components_in: NDArray[dtype], mean_in: NDArray[mean_dtype], variance_in: NDArray[dtype], whiten: Bool) raises -> NDArray[dtype]:
    if components_in.shape[0] == 0: return zeros[dtype](Shape(X.shape[0], 0))
    var components = components_in.copy()
    var result = Backend.matmul(X, components.T())
    var means = mean_in.astype[dtype]().reshape(Shape(1, components_in.shape[1]))
    var offset = Backend.matmul(means, components.T())
    var epsilon: Float64 = 2.220446049250313e-16
    comptime if dtype == DType.float32: epsilon = 1.1920928955078125e-7
    for i in range(X.shape[0]):
        for j in range(components_in.shape[0]):
            var v = _get(result, i, j) - _get(offset, 0, j)
            if whiten:
                v /= max(_sqrt(Float64(variance_in.unsafe_get(j))), epsilon)
            _set(result, i, j, v)
    return result^


def _pca_inverse_transform[dtype: DType, Backend: PCABackend, mean_dtype: DType](X: NDArray[dtype], components_in: NDArray[dtype], mean_in: NDArray[mean_dtype], variance_in: NDArray[dtype], whiten: Bool) raises -> NDArray[dtype]:
    if X.ndim != 2 or X.shape[1] != components_in.shape[0] or X.shape[0] < 1:
        raise Error("ValueError: inverse_transform shape mismatch")
    if components_in.shape[0] > 0: validate_matrix(X)
    var result = zeros[dtype](Shape(X.shape[0], components_in.shape[1]))
    if components_in.shape[0] > 0:
        var components = components_in.copy()
        if whiten:
            for i in range(components_in.shape[0]):
                var scale = _sqrt(Float64(variance_in.unsafe_get(i)))
                for j in range(components_in.shape[1]): _set(components, i, j, _get(components, i, j) * scale)
        result = Backend.matmul(X, components)
    for i in range(X.shape[0]):
        for j in range(components_in.shape[1]):
            _set(result, i, j, _get(result, i, j) + Float64(mean_in.unsafe_get(j)))
    return result^


def _pca_get_covariance[dtype: DType, Backend: PCABackend](components_in: NDArray[dtype], variance_in: NDArray[dtype], whiten: Bool, noise: Float64) raises -> NDArray[dtype]:
    var p = components_in.shape[1]
    var result = zeros[dtype](Shape(p, p))
    if components_in.shape[0] > 0:
        var components = components_in.copy()
        var weighted = components.copy()
        for i in range(components_in.shape[0]):
            var variance = Float64(variance_in.unsafe_get(i))
            var scale = _sqrt(variance) if whiten else Float64(1)
            for j in range(p):
                var v = _get(components, i, j) * scale
                _set(components, i, j, v)
                _set(weighted, i, j, v * max(0.0, variance - noise))
        result = Backend.matmul(components.T(), weighted)
    for i in range(p): _set(result, i, i, _get(result, i, i) + noise)
    return result^
