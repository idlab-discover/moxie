"""BayesianGaussianMixture: variational inference for Gaussian mixtures.

Native port of sklearn.mixture.BayesianGaussianMixture (pinned 4358542a):
Dirichlet-process (stick-breaking) and Dirichlet-distribution weight priors,
Gaussian-Wishart priors for all four covariance types, variational updates
(`_estimate_weights/_means/_precisions`, `_estimate_wishart_*`), expected log
weights/densities and the evidence lower bound. The EM lifecycle, Gaussian
sufficient statistics, log densities, precision factors, responsibilities,
initialization and sampling are consumed from the shared Moxie mixture modules
(`moxie.mixture._em`, `_gaussian_kernels`, `_precision`, `_responsibilities`,
`_gaussian_mixture.sample_gaussian_mixture`); only the Bayesian updates live
here. digamma/gammaln/betaln come from `moxie._mixture_linalg.special_gamma`.

Mojo adaptations (docs/mixture/bayesian/api.md): compile-time dtype; `random_state:
Optional[Int]` plus `*_with_rng` methods borrowing a RandomState; scalar priors
are `Optional[Float64]`; array priors are `Optional[NDArray[dtype]]`, with a
spherical `covariance_prior` given as a shape-(1,) array; `weight_concentration_`
is a List of one (Dirichlet distribution) or two (Dirichlet process) arrays;
tied `degrees_of_freedom_` has shape (1,); resolved scalar priors are Float64.
"""
from numojo import NDArray, Shape, zeros
from std.math import abs, isfinite
from std.utils.numerics import inf
from moxie._mixture_linalg.special_gamma import digamma, gammaln, betaln
from moxie._random import RandomState, entropy_state
from moxie.mixture._base import MixtureDensity
from moxie.mixture._contracts import EMControl, check_input
from moxie.mixture._em import MixtureEMModel, EMSettings, run_em, weighted_log_prob
from moxie.mixture._gaussian_kernels import estimate_gaussian_parameters, estimate_log_gaussian_prob, rounded, Matrix
from moxie.mixture._precision import (
    compute_precision_cholesky, log_det_cholesky, precisions_from_cholesky,
    check_precision_matrix, compact,
)
from moxie.mixture._responsibilities import log_responsibilities, logsumexp_rows, exp_array, mean_vector, argmax_rows
from moxie.mixture._gaussian_mixture import sample_gaussian_mixture
from moxie.mixture._scalar_math import log64, exp64

comptime LOG_2: Float64 = 0.6931471805599453094172321214581765680755


def _empty[dtype: DType]() raises -> NDArray[dtype]:
    return zeros[dtype](Shape(0))


struct BayesianState[dtype: DType](Copyable, Movable):
    """Variational parameters restored for the best restart (`_get_parameters`)."""

    var concentration_a: NDArray[Self.dtype]
    var concentration_b: NDArray[Self.dtype]  # empty for dirichlet_distribution
    var mean_precision: NDArray[Self.dtype]
    var means: NDArray[Self.dtype]
    var degrees_of_freedom: NDArray[Self.dtype]  # (k,), tied: (1,)
    var covariances: NDArray[Self.dtype]
    var precisions_cholesky: NDArray[Self.dtype]

    def __init__(out self) raises:
        self.concentration_a = _empty[Self.dtype]()
        self.concentration_b = _empty[Self.dtype]()
        self.mean_precision = _empty[Self.dtype]()
        self.means = _empty[Self.dtype]()
        self.degrees_of_freedom = _empty[Self.dtype]()
        self.covariances = _empty[Self.dtype]()
        self.precisions_cholesky = _empty[Self.dtype]()


struct BayesianPriors[dtype: DType](Copyable, Movable):
    """Resolved priors (`*_prior_` attributes after `_check_parameters`)."""

    var weight_concentration_prior: Float64
    var mean_precision_prior: Float64
    var mean_prior: NDArray[Self.dtype]
    var degrees_of_freedom_prior: Float64
    var covariance_prior: NDArray[Self.dtype]  # full/tied (d,d), diag (d,), spherical (1,)

    def __init__(out self, weight_concentration_prior: Float64, mean_precision_prior: Float64, var mean_prior: NDArray[Self.dtype], degrees_of_freedom_prior: Float64, var covariance_prior: NDArray[Self.dtype]):
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior^
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior^


def _vector[dtype: DType](values: List[Float64]) raises -> NDArray[dtype]:
    var out = zeros[dtype](Shape(len(values)))
    for i in range(len(values)):
        out.unsafe_set(i, values[i].cast[dtype]())
    return out^


def _get[dtype: DType](x: NDArray[dtype], i: Int) -> Float64:
    return Float64(x.unsafe_get(i))


struct BayesianEngine[dtype_: DType](MixtureEMModel, Copyable):
    """Variational EM hooks for BayesianGaussianMixture."""

    comptime dtype = Self.dtype_
    comptime State = BayesianState[Self.dtype_]

    var covariance_type: String
    var dirichlet_process: Bool
    var reg_covar: Float64
    var n_components: Int
    var priors: BayesianPriors[Self.dtype]
    var state: BayesianState[Self.dtype]

    def __init__(out self, covariance_type: String, dirichlet_process: Bool, reg_covar: Float64, n_components: Int, var priors: BayesianPriors[Self.dtype], var state: BayesianState[Self.dtype]):
        self.covariance_type = covariance_type
        self.dirichlet_process = dirichlet_process
        self.reg_covar = reg_covar
        self.n_components = n_components
        self.priors = priors^
        self.state = state^

    def needs_responsibilities(self) -> Bool:
        return True

    def _update(mut self, x: NDArray[Self.dtype], resp: NDArray[Self.dtype]) raises:
        var params = estimate_gaussian_parameters(x, resp, self.reg_covar, self.covariance_type)
        self._estimate_weights(params.nk)
        self._estimate_means(params.nk, params.means)
        self._estimate_precisions(params.nk, params.means, params.covariances)

    def _estimate_weights(mut self, nk: NDArray[Self.dtype]) raises:
        var k = nk.shape[0]
        var prior = self.priors.weight_concentration_prior
        if self.dirichlet_process:
            var a = List[Float64]()
            var b = List[Float64](length=k, fill=0.0)
            for j in range(k):
                a.append(rounded[Self.dtype](1.0 + _get(nk, j)))
            # b_j = prior + sum_{l > j} nk_l, as np.cumsum(nk[::-1])[-2::-1].
            var tail = 0.0
            for j in range(k - 1, 0, -1):
                tail = rounded[Self.dtype](tail + _get(nk, j))
                b[j - 1] = rounded[Self.dtype](prior + tail)
            b[k - 1] = rounded[Self.dtype](prior + 0.0)
            self.state.concentration_a = _vector[Self.dtype](a)
            self.state.concentration_b = _vector[Self.dtype](b)
        else:
            var a = List[Float64]()
            for j in range(k):
                a.append(rounded[Self.dtype](prior + _get(nk, j)))
            self.state.concentration_a = _vector[Self.dtype](a)
            self.state.concentration_b = _empty[Self.dtype]()

    def _estimate_means(mut self, nk: NDArray[Self.dtype], xk: NDArray[Self.dtype]) raises:
        var k = xk.shape[0]
        var d = xk.shape[1]
        var beta0 = self.priors.mean_precision_prior
        var precision = List[Float64]()
        for j in range(k):
            precision.append(rounded[Self.dtype](beta0 + _get(nk, j)))
        self.state.mean_precision = _vector[Self.dtype](precision)
        var means = zeros[Self.dtype](Shape(k, d))
        for j in range(k):
            for f in range(d):
                var prior_term = rounded[Self.dtype](beta0 * _get(self.priors.mean_prior, f))
                var data_term = rounded[Self.dtype](_get(nk, j) * _get(xk, j * d + f))
                means.unsafe_set(j * d + f, (rounded[Self.dtype](prior_term + data_term) / precision[j]).cast[Self.dtype]())
        self.state.means = means^

    def _estimate_precisions(mut self, nk: NDArray[Self.dtype], xk: NDArray[Self.dtype], sk: NDArray[Self.dtype]) raises:
        var k = xk.shape[0]
        var d = xk.shape[1]
        var kind = self.covariance_type
        var beta0 = self.priors.mean_precision_prior
        var nu0 = self.priors.degrees_of_freedom_prior
        var prior = self.priors.covariance_prior.copy()
        var diff = List[Float64](length=k * d, fill=0.0)
        for j in range(k):
            for f in range(d):
                diff[j * d + f] = rounded[Self.dtype](_get(xk, j * d + f) - _get(self.priors.mean_prior, f))
        var covariances = zeros[Self.dtype](sk.shape)
        if kind == "full":
            var dof = List[Float64]()
            for j in range(k):
                dof.append(rounded[Self.dtype](nu0 + _get(nk, j)))
                var scale = rounded[Self.dtype](rounded[Self.dtype](_get(nk, j) * beta0) / _get(self.state.mean_precision, j))
                for f in range(d):
                    for g in range(d):
                        var idx = j * d * d + f * d + g
                        var value = rounded[Self.dtype](_get(prior, f * d + g) + rounded[Self.dtype](_get(nk, j) * _get(sk, idx)))
                        value = rounded[Self.dtype](value + rounded[Self.dtype](scale * rounded[Self.dtype](diff[j * d + f] * diff[j * d + g])))
                        covariances.unsafe_set(idx, (value / dof[j]).cast[Self.dtype]())
            self.state.degrees_of_freedom = _vector[Self.dtype](dof)
        elif kind == "tied":
            var total = 0.0
            for j in range(k):
                total += _get(nk, j)
            total = rounded[Self.dtype](total)
            var per_component = rounded[Self.dtype](total / Float64(k))
            var dof = rounded[Self.dtype](nu0 + per_component)
            var weights = List[Float64]()
            for j in range(k):
                weights.append(rounded[Self.dtype](_get(nk, j) / _get(self.state.mean_precision, j)))
            var factor = rounded[Self.dtype](beta0 / Float64(k))
            for f in range(d):
                for g in range(d):
                    var dot = 0.0
                    for j in range(k):
                        dot += rounded[Self.dtype](weights[j] * diff[j * d + f]) * diff[j * d + g]
                    var value = rounded[Self.dtype](_get(prior, f * d + g) + rounded[Self.dtype](rounded[Self.dtype](_get(sk, f * d + g) * total) / Float64(k)))
                    value = rounded[Self.dtype](value + rounded[Self.dtype](factor * rounded[Self.dtype](dot)))
                    covariances.unsafe_set(f * d + g, (value / dof).cast[Self.dtype]())
            self.state.degrees_of_freedom = _vector[Self.dtype]([dof])
        elif kind == "diag":
            var dof = List[Float64]()
            for j in range(k):
                dof.append(rounded[Self.dtype](nu0 + _get(nk, j)))
                var ratio = rounded[Self.dtype](beta0 / _get(self.state.mean_precision, j))
                for f in range(d):
                    var inner = rounded[Self.dtype](_get(sk, j * d + f) + rounded[Self.dtype](ratio * rounded[Self.dtype](diff[j * d + f] * diff[j * d + f])))
                    var value = rounded[Self.dtype](_get(prior, f) + rounded[Self.dtype](_get(nk, j) * inner))
                    covariances.unsafe_set(j * d + f, (value / dof[j]).cast[Self.dtype]())
            self.state.degrees_of_freedom = _vector[Self.dtype](dof)
        else:
            var dof = List[Float64]()
            for j in range(k):
                dof.append(rounded[Self.dtype](nu0 + _get(nk, j)))
                var squares = 0.0
                for f in range(d):
                    squares += rounded[Self.dtype](diff[j * d + f] * diff[j * d + f])
                var mean_square = rounded[Self.dtype](rounded[Self.dtype](squares) / Float64(d))
                var ratio = rounded[Self.dtype](beta0 / _get(self.state.mean_precision, j))
                var inner = rounded[Self.dtype](_get(sk, j) + rounded[Self.dtype](ratio * mean_square))
                var value = rounded[Self.dtype](_get(prior, 0) + rounded[Self.dtype](_get(nk, j) * inner))
                covariances.unsafe_set(j, (value / dof[j]).cast[Self.dtype]())
            self.state.degrees_of_freedom = _vector[Self.dtype](dof)
        self.state.precisions_cholesky = compute_precision_cholesky(covariances, kind)
        self.state.covariances = covariances^

    def _dof(self, j: Int) -> Float64:
        return _get(self.state.degrees_of_freedom, 0 if self.covariance_type == "tied" else j)

    def initialize(mut self, x: NDArray[Self.dtype], resp: NDArray[Self.dtype]) raises:
        self._update(x, resp)

    def m_step(mut self, x: NDArray[Self.dtype], log_resp: NDArray[Self.dtype]) raises:
        self._update(x, exp_array(log_resp))

    def estimate_log_weights(self) raises -> NDArray[Self.dtype]:
        var k = self.state.concentration_a.shape[0]
        var out = zeros[Self.dtype](Shape(k))
        if self.dirichlet_process:
            var running = 0.0
            for j in range(k):
                var a = _get(self.state.concentration_a, j)
                var b = _get(self.state.concentration_b, j)
                var psi_sum = rounded[Self.dtype](digamma(rounded[Self.dtype](a + b)))
                var value = rounded[Self.dtype](rounded[Self.dtype](digamma(a)) - psi_sum)
                # + hstack((0, cumsum(digamma_b - digamma_sum)[:-1]))
                out.unsafe_set(j, rounded[Self.dtype](value + running).cast[Self.dtype]())
                running = rounded[Self.dtype](running + rounded[Self.dtype](rounded[Self.dtype](digamma(b)) - psi_sum))
        else:
            var total = 0.0
            for j in range(k):
                total += _get(self.state.concentration_a, j)
            var psi_total = rounded[Self.dtype](digamma(rounded[Self.dtype](total)))
            for j in range(k):
                out.unsafe_set(j, (rounded[Self.dtype](digamma(_get(self.state.concentration_a, j))) - psi_total).cast[Self.dtype]())
        return out^

    def estimate_log_prob(self, x: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        var d = x.shape[1]
        var kind = self.covariance_type
        var log_det = log_det_cholesky(self.state.precisions_cholesky, kind, d)
        var log_gauss = estimate_log_gaussian_prob(x, self.state.means, self.state.precisions_cholesky, kind, log_det)
        var n = log_gauss.shape[0]
        var k = log_gauss.shape[1]
        var shift = List[Float64](length=k, fill=0.0)
        var correction = List[Float64](length=k, fill=0.0)
        for j in range(k):
            var dof = self._dof(j)
            shift[j] = rounded[Self.dtype](0.5 * Float64(d) * log64(dof))
            var psi = 0.0
            for f in range(d):
                psi += rounded[Self.dtype](digamma(rounded[Self.dtype](0.5 * (dof - Float64(f)))))
            var log_lambda = rounded[Self.dtype](Float64(d) * LOG_2 + rounded[Self.dtype](psi))
            var term = rounded[Self.dtype](log_lambda - rounded[Self.dtype](Float64(d) / _get(self.state.mean_precision, j)))
            correction[j] = rounded[Self.dtype](0.5 * term)
        for i in range(n):
            for j in range(k):
                var value = rounded[Self.dtype](Float64(log_gauss.unsafe_get(i * k + j)) - shift[j])
                log_gauss.unsafe_set(i * k + j, (value + correction[j]).cast[Self.dtype]())
        return log_gauss^

    def lower_bound(self, log_resp: NDArray[Self.dtype], mean_log_prob: Float64) raises -> Float64:
        var d = self.priors.mean_prior.shape[0]
        var kind = self.covariance_type
        var k = self.n_components
        var log_det = log_det_cholesky(self.state.precisions_cholesky, kind, d)
        var log_wishart = 0.0
        var count = 1 if kind == "tied" else k
        for j in range(count):
            var dof = self._dof(j)
            var log_det_precision = rounded[Self.dtype](log_det[j] - rounded[Self.dtype](0.5 * Float64(d) * log64(dof)))
            var gammas = 0.0
            for f in range(d):
                gammas += gammaln(rounded[Self.dtype](0.5 * (dof - Float64(f))))
            var norm = -(dof * log_det_precision + dof * Float64(d) * 0.5 * LOG_2 + rounded[Self.dtype](gammas))
            log_wishart += norm
        if kind == "tied":
            log_wishart *= Float64(k)
        var log_norm_weight: Float64
        if self.dirichlet_process:
            var total = 0.0
            for j in range(k):
                total += betaln(_get(self.state.concentration_a, j), _get(self.state.concentration_b, j))
            log_norm_weight = -total
        else:
            var total = 0.0
            var gammas = 0.0
            for j in range(k):
                total += _get(self.state.concentration_a, j)
                gammas += gammaln(_get(self.state.concentration_a, j))
            log_norm_weight = gammaln(total) - gammas
        var entropy = 0.0
        for i in range(log_resp.size):
            var value = Float64(log_resp.unsafe_get(i))
            # exp(-inf) * -inf is NaN in NumPy too; keep that propagation.
            entropy += rounded[Self.dtype](exp64(value)) * value
        var log_precision = 0.0
        for j in range(k):
            log_precision += log64(_get(self.state.mean_precision, j))
        return -entropy - log_wishart - log_norm_weight - 0.5 * Float64(d) * log_precision

    def get_state(self) raises -> Self.State:
        return self.state.copy()

    def set_state(mut self, var state: Self.State) raises:
        self.state = state^


def mixture_weights[dtype: DType](state: BayesianState[dtype], dirichlet_process: Bool) raises -> NDArray[dtype]:
    """Return weights_ from the concentration (`_set_parameters`)."""
    var k = state.concentration_a.shape[0]
    var weights = List[Float64](length=k, fill=0.0)
    var total = 0.0
    if dirichlet_process:
        var product = 1.0
        for j in range(k):
            var a = _get(state.concentration_a, j)
            var b = _get(state.concentration_b, j)
            var s = rounded[dtype](a + b)
            weights[j] = rounded[dtype](rounded[dtype](a / s) * product)
            product = rounded[dtype](product * rounded[dtype](b / s))
            total += weights[j]
    else:
        for j in range(k):
            weights[j] = _get(state.concentration_a, j)
            total += weights[j]
    total = rounded[dtype](total)
    for j in range(k):
        weights[j] = weights[j] / total
    return _vector[dtype](weights)


@fieldwise_init
struct BayesianGaussianMixtureParams[dtype: DType](Copyable, Movable):
    """Typed constructor configuration (get_params/set_params record)."""

    var n_components: Int
    var covariance_type: String
    var tol: Float64
    var reg_covar: Float64
    var max_iter: Int
    var n_init: Int
    var init_params: String
    var weight_concentration_prior_type: String
    var weight_concentration_prior: Optional[Float64]
    var mean_precision_prior: Optional[Float64]
    var mean_prior: Optional[NDArray[Self.dtype]]
    var degrees_of_freedom_prior: Optional[Float64]
    var covariance_prior: Optional[NDArray[Self.dtype]]
    var random_state: Optional[Int]
    var warm_start: Bool
    var verbose: Int
    var verbose_interval: Int


def _positive(value: Optional[Float64], name: String) raises:
    if value:
        var v = value.value()
        if not (v > 0.0) or not isfinite(v):
            raise Error(String("ValueError: The '", name, "' parameter of BayesianGaussianMixture must be None or a float in the range (0.0, inf). Got ", v, " instead."))


struct BayesianGaussianMixture[dtype: DType = DType.float64](MixtureDensity, Copyable):
    """Variational Bayesian estimation of a Gaussian mixture.

    Parameters, defaults and learned attributes follow
    sklearn.mixture.BayesianGaussianMixture; learned arrays are empty before fit.
    """

    comptime Params = BayesianGaussianMixtureParams[Self.dtype]
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Prediction = NDArray[DType.int64]
    comptime Probabilities = NDArray[Self.dtype]
    comptime SampleScores = NDArray[Self.dtype]
    comptime Samples = Tuple[NDArray[DType.float64], NDArray[DType.int64]]

    var n_components: Int
    var covariance_type: String
    var tol: Float64
    var reg_covar: Float64
    var max_iter: Int
    var n_init: Int
    var init_params: String
    var weight_concentration_prior_type: String
    var weight_concentration_prior: Optional[Float64]
    var mean_precision_prior: Optional[Float64]
    var mean_prior: Optional[NDArray[Self.dtype]]
    var degrees_of_freedom_prior: Optional[Float64]
    var covariance_prior: Optional[NDArray[Self.dtype]]
    var random_state: Optional[Int]
    var warm_start: Bool
    var verbose: Int
    var verbose_interval: Int
    # Learned attributes.
    var weights_: NDArray[Self.dtype]
    var means_: NDArray[Self.dtype]
    var covariances_: NDArray[Self.dtype]
    var precisions_: NDArray[Self.dtype]
    var precisions_cholesky_: NDArray[Self.dtype]
    var converged_: Bool
    var n_iter_: Int
    var lower_bound_: Float64
    var lower_bounds_: List[Float64]
    var weight_concentration_prior_: Float64
    var weight_concentration_: List[NDArray[Self.dtype]]
    var mean_precision_prior_: Float64
    var mean_precision_: NDArray[Self.dtype]
    var mean_prior_: NDArray[Self.dtype]
    var degrees_of_freedom_prior_: Float64
    var degrees_of_freedom_: NDArray[Self.dtype]
    var covariance_prior_: NDArray[Self.dtype]
    var n_features_in_: Int
    var convergence_warning_: String
    var _fitted: Bool

    def __init__(
        out self,
        *,
        n_components: Int = 1,
        covariance_type: String = "full",
        tol: Float64 = 1e-3,
        reg_covar: Float64 = 1e-6,
        max_iter: Int = 100,
        n_init: Int = 1,
        init_params: String = "kmeans",
        weight_concentration_prior_type: String = "dirichlet_process",
        weight_concentration_prior: Optional[Float64] = None,
        mean_precision_prior: Optional[Float64] = None,
        var mean_prior: Optional[NDArray[Self.dtype]] = None,
        degrees_of_freedom_prior: Optional[Float64] = None,
        var covariance_prior: Optional[NDArray[Self.dtype]] = None,
        random_state: Optional[Int] = None,
        warm_start: Bool = False,
        verbose: Int = 0,
        verbose_interval: Int = 10,
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "BayesianGaussianMixture supports Float32 and Float64"
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior^
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior^
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.weights_ = _empty[Self.dtype]()
        self.means_ = _empty[Self.dtype]()
        self.covariances_ = _empty[Self.dtype]()
        self.precisions_ = _empty[Self.dtype]()
        self.precisions_cholesky_ = _empty[Self.dtype]()
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = 0.0
        self.lower_bounds_ = List[Float64]()
        self.weight_concentration_prior_ = 0.0
        self.weight_concentration_ = List[NDArray[Self.dtype]]()
        self.mean_precision_prior_ = 0.0
        self.mean_precision_ = _empty[Self.dtype]()
        self.mean_prior_ = _empty[Self.dtype]()
        self.degrees_of_freedom_prior_ = 0.0
        self.degrees_of_freedom_ = _empty[Self.dtype]()
        self.covariance_prior_ = _empty[Self.dtype]()
        self.n_features_in_ = 0
        self.convergence_warning_ = String("")
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return BayesianGaussianMixtureParams[Self.dtype](
            self.n_components, self.covariance_type, self.tol, self.reg_covar, self.max_iter, self.n_init,
            self.init_params, self.weight_concentration_prior_type, self.weight_concentration_prior,
            self.mean_precision_prior, self.mean_prior.copy(), self.degrees_of_freedom_prior,
            self.covariance_prior.copy(), self.random_state, self.warm_start, self.verbose, self.verbose_interval,
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.covariance_type = params.covariance_type
        self.tol = params.tol
        self.reg_covar = params.reg_covar
        self.max_iter = params.max_iter
        self.n_init = params.n_init
        self.init_params = params.init_params
        self.weight_concentration_prior_type = params.weight_concentration_prior_type
        self.weight_concentration_prior = params.weight_concentration_prior
        self.mean_precision_prior = params.mean_precision_prior
        self.mean_prior = params.mean_prior.copy()
        self.degrees_of_freedom_prior = params.degrees_of_freedom_prior
        self.covariance_prior = params.covariance_prior.copy()
        self.random_state = params.random_state
        self.warm_start = params.warm_start
        self.verbose = params.verbose
        self.verbose_interval = params.verbose_interval
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        n_components: Optional[Int] = None,
        covariance_type: Optional[String] = None,
        tol: Optional[Float64] = None,
        reg_covar: Optional[Float64] = None,
        max_iter: Optional[Int] = None,
        n_init: Optional[Int] = None,
        init_params: Optional[String] = None,
        weight_concentration_prior_type: Optional[String] = None,
        weight_concentration_prior: Optional[Float64] = None,
        mean_precision_prior: Optional[Float64] = None,
        var mean_prior: Optional[NDArray[Self.dtype]] = None,
        degrees_of_freedom_prior: Optional[Float64] = None,
        var covariance_prior: Optional[NDArray[Self.dtype]] = None,
        random_state: Optional[Int] = None,
        warm_start: Optional[Bool] = None,
        verbose: Optional[Int] = None,
        verbose_interval: Optional[Int] = None,
    ) raises -> ref[self] Self:
        """Named partial update (sklearn `set_params(**params)`); learned state is kept.

        Omitted names keep their value; pass a full Params record to reset an
        optional prior or `random_state` to None. Values are validated at the
        next fit, as in sklearn.
        """
        if n_components:
            self.n_components = n_components.value()
        if covariance_type:
            self.covariance_type = covariance_type.value()
        if tol:
            self.tol = tol.value()
        if reg_covar:
            self.reg_covar = reg_covar.value()
        if max_iter:
            self.max_iter = max_iter.value()
        if n_init:
            self.n_init = n_init.value()
        if init_params:
            self.init_params = init_params.value()
        if weight_concentration_prior_type:
            self.weight_concentration_prior_type = weight_concentration_prior_type.value()
        if weight_concentration_prior:
            self.weight_concentration_prior = weight_concentration_prior
        if mean_precision_prior:
            self.mean_precision_prior = mean_precision_prior
        if mean_prior:
            self.mean_prior = mean_prior^
        if degrees_of_freedom_prior:
            self.degrees_of_freedom_prior = degrees_of_freedom_prior
        if covariance_prior:
            self.covariance_prior = covariance_prior^
        if random_state:
            self.random_state = random_state
        if warm_start:
            self.warm_start = warm_start.value()
        if verbose:
            self.verbose = verbose.value()
        if verbose_interval:
            self.verbose_interval = verbose_interval.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var clone = Self()
        _ = clone.set_params(self.get_params())
        return clone^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def _control(self) raises -> EMControl:
        var control = EMControl()
        control.n_components = self.n_components
        control.covariance_type = self.covariance_type
        control.tol = self.tol
        control.reg_covar = self.reg_covar
        control.max_iter = self.max_iter
        control.n_init = self.n_init
        control.init_params = self.init_params
        control.warm_start = self.warm_start
        control.verbose = self.verbose
        control.verbose_interval = self.verbose_interval
        try:
            control.validate()
        except error:
            raise Error(String("ValueError: invalid BayesianGaussianMixture parameters (n_components=", self.n_components,
                               ", covariance_type='", self.covariance_type, "', tol=", self.tol, ", reg_covar=", self.reg_covar,
                               ", max_iter=", self.max_iter, ", n_init=", self.n_init, ", init_params='", self.init_params,
                               "'): ", error))
        if self.weight_concentration_prior_type != "dirichlet_process" and self.weight_concentration_prior_type != "dirichlet_distribution":
            raise Error(String("ValueError: The 'weight_concentration_prior_type' parameter of BayesianGaussianMixture must be a str among {'dirichlet_process', 'dirichlet_distribution'}. Got '", self.weight_concentration_prior_type, "' instead."))
        _positive(self.weight_concentration_prior, "weight_concentration_prior")
        _positive(self.mean_precision_prior, "mean_precision_prior")
        _positive(self.degrees_of_freedom_prior, "degrees_of_freedom_prior")
        return control^

    def _resolve_priors(self, X: NDArray[Self.dtype]) raises -> BayesianPriors[Self.dtype]:
        """`_check_parameters`: resolve data-dependent defaults and validate priors."""
        var n = X.shape[0]
        var d = X.shape[1]
        var xv = Matrix[Self.dtype](X)
        var weight = self.weight_concentration_prior.value() if self.weight_concentration_prior else 1.0 / Float64(self.n_components)
        var beta0 = self.mean_precision_prior.value() if self.mean_precision_prior else 1.0
        var mean_prior: NDArray[Self.dtype]
        if self.mean_prior:
            var given = self.mean_prior.value().copy()
            if given.ndim != 1 or given.shape[0] != d:
                raise Error(String("ValueError: The parameter 'means' should have the shape of (", d, ",), but got ", given.shape))
            mean_prior = compact(given)
        else:
            mean_prior = zeros[Self.dtype](Shape(d))
            for f in range(d):
                var total = 0.0
                for i in range(n):
                    total += xv.at(X, i, f)
                mean_prior.unsafe_set(f, (total / Float64(n)).cast[Self.dtype]())
        var nu0: Float64
        if not self.degrees_of_freedom_prior:
            nu0 = Float64(d)
        elif self.degrees_of_freedom_prior.value() > Float64(d) - 1.0:
            nu0 = self.degrees_of_freedom_prior.value()
        else:
            raise Error(String("ValueError: The parameter 'degrees_of_freedom_prior' should be greater than ", d - 1, ", but got ", self.degrees_of_freedom_prior.value(), "."))
        var kind = self.covariance_type
        var prior: NDArray[Self.dtype]
        if self.covariance_prior:
            prior = compact(self.covariance_prior.value())
            if kind == "full" or kind == "tied":
                if prior.ndim != 2 or prior.shape[0] != d or prior.shape[1] != d:
                    raise Error(String("ValueError: The parameter '", kind, " covariance_prior' should have the shape of (", d, ", ", d, "), but got ", prior.shape))
                var work = zeros[DType.float64](Shape(d, d))
                for idx in range(d * d):
                    work.unsafe_set(idx, Float64(prior.unsafe_get(idx)))
                check_precision_matrix[Self.dtype](work, kind)
            elif kind == "diag":
                if prior.ndim != 1 or prior.shape[0] != d:
                    raise Error(String("ValueError: The parameter 'diag covariance_prior' should have the shape of (", d, ",), but got ", prior.shape))
                for f in range(d):
                    if not (Float64(prior.unsafe_get(f)) > 0.0):
                        raise Error("ValueError: 'diag precision' should be positive")
            else:
                if prior.ndim != 1 or prior.shape[0] != 1 or not (Float64(prior.unsafe_get(0)) > 0.0):
                    raise Error("ValueError: spherical covariance_prior must be a positive value given as a shape-(1,) array")
        else:
            # np.cov(X.T) (ddof=1) for full/tied, np.var(X, 0, ddof=1) for diag, its mean for spherical.
            var centered = List[Float64](length=n * d, fill=0.0)
            for f in range(d):
                var total = 0.0
                for i in range(n):
                    total += xv.at(X, i, f)
                var mean = total / Float64(n)
                for i in range(n):
                    centered[i * d + f] = xv.at(X, i, f) - mean
            var denominator = Float64(n - 1)
            if kind == "full" or kind == "tied":
                prior = zeros[Self.dtype](Shape(d, d))
                for f in range(d):
                    for g in range(d):
                        var total = 0.0
                        for i in range(n):
                            total += centered[i * d + f] * centered[i * d + g]
                        prior.unsafe_set(f * d + g, (total / denominator).cast[Self.dtype]())
            else:
                var variances = List[Float64](length=d, fill=0.0)
                for f in range(d):
                    var total = 0.0
                    for i in range(n):
                        total += centered[i * d + f] * centered[i * d + f]
                    variances[f] = rounded[Self.dtype](total / denominator)
                if kind == "diag":
                    prior = _vector[Self.dtype](variances)
                else:
                    var total = 0.0
                    for f in range(d):
                        total += variances[f]
                    prior = _vector[Self.dtype]([total / Float64(d)])
        return BayesianPriors[Self.dtype](weight, beta0, mean_prior^, nu0, prior^)

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        """Estimate the variational posterior (y is ignored)."""
        _ = self.fit_predict(X, y)
        return self

    def fit_with_rng[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, mut rng: RandomState) raises -> ref[self] Self:
        """Fit, borrowing and advancing a caller-owned RandomState."""
        _ = self.fit_predict_with_rng(X, rng)
        return self

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> NDArray[DType.int64]:
        """Fit and return labels of the final E-step (y is ignored)."""
        var rng = RandomState(self.random_state.value()) if self.random_state else entropy_state()
        return self.fit_predict_with_rng(X, rng)

    def fit_predict_with_rng(mut self, X: Self.Input, mut rng: RandomState) raises -> NDArray[DType.int64]:
        """Fit_predict, borrowing and advancing a caller-owned RandomState."""
        var control = self._control()
        check_input(X, self.n_components, fitting=True)
        var n_features = X.shape[1]
        var priors = self._resolve_priors(X)
        var do_init = not (self.warm_start and self._fitted)
        var state = BayesianState[Self.dtype]()
        var initial_lower_bound = 0.0
        if not do_init:
            if n_features != self.n_features_in_:
                raise Error(String("ValueError: X has ", n_features, " features, but BayesianGaussianMixture is expecting ", self.n_features_in_, " features as input."))
            state.concentration_a = self.weight_concentration_[0].copy()
            if len(self.weight_concentration_) == 2:
                state.concentration_b = self.weight_concentration_[1].copy()
            state.mean_precision = self.mean_precision_.copy()
            state.means = self.means_.copy()
            state.degrees_of_freedom = self.degrees_of_freedom_.copy()
            state.covariances = self.covariances_.copy()
            state.precisions_cholesky = self.precisions_cholesky_.copy()
            initial_lower_bound = self.lower_bound_
        var dirichlet_process = self.weight_concentration_prior_type == "dirichlet_process"
        var engine = BayesianEngine[Self.dtype](self.covariance_type, dirichlet_process, self.reg_covar, self.n_components, priors.copy(), state^)
        var settings = EMSettings(self.n_components, self.tol, self.max_iter, control.restart_count(self._fitted),
                                  self.init_params, self.verbose, self.verbose_interval, do_init, initial_lower_bound)
        var result = run_em(engine, X, settings, rng)
        self.weight_concentration_prior_ = priors.weight_concentration_prior
        self.mean_precision_prior_ = priors.mean_precision_prior
        self.mean_prior_ = priors.mean_prior.copy()
        self.degrees_of_freedom_prior_ = priors.degrees_of_freedom_prior
        self.covariance_prior_ = priors.covariance_prior.copy()
        self.weight_concentration_ = List[NDArray[Self.dtype]]()
        self.weight_concentration_.append(engine.state.concentration_a.copy())
        if dirichlet_process:
            self.weight_concentration_.append(engine.state.concentration_b.copy())
        self.mean_precision_ = engine.state.mean_precision.copy()
        self.means_ = engine.state.means.copy()
        self.degrees_of_freedom_ = engine.state.degrees_of_freedom.copy()
        self.covariances_ = engine.state.covariances.copy()
        self.precisions_cholesky_ = engine.state.precisions_cholesky.copy()
        self.precisions_ = precisions_from_cholesky(self.precisions_cholesky_, self.covariance_type)
        self.weights_ = mixture_weights(engine.state, dirichlet_process)
        self.converged_ = result.converged
        self.n_iter_ = result.n_iter
        self.lower_bound_ = result.lower_bound
        self.lower_bounds_ = result.lower_bounds.copy()
        self.n_features_in_ = n_features
        self.convergence_warning_ = result.warning
        self._fitted = True
        return result.labels.copy()

    def _engine(self) raises -> BayesianEngine[Self.dtype]:
        var state = BayesianState[Self.dtype]()
        state.concentration_a = self.weight_concentration_[0].copy()
        if len(self.weight_concentration_) == 2:
            state.concentration_b = self.weight_concentration_[1].copy()
        state.mean_precision = self.mean_precision_.copy()
        state.means = self.means_.copy()
        state.degrees_of_freedom = self.degrees_of_freedom_.copy()
        state.covariances = self.covariances_.copy()
        state.precisions_cholesky = self.precisions_cholesky_.copy()
        var priors = BayesianPriors[Self.dtype](self.weight_concentration_prior_, self.mean_precision_prior_, self.mean_prior_.copy(),
                                                self.degrees_of_freedom_prior_, self.covariance_prior_.copy())
        return BayesianEngine[Self.dtype](self.covariance_type, len(self.weight_concentration_) == 2, self.reg_covar, self.n_components, priors^, state^)

    def _weighted_log_prob(self, X: Self.Input) raises -> NDArray[Self.dtype]:
        if not self._fitted:
            raise Error("NotFittedError: This BayesianGaussianMixture instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        check_input(X, 1, fitting=False, features=self.n_features_in_)
        return weighted_log_prob(self._engine(), X)

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        """Component labels: argmax of expected weighted log probabilities."""
        return argmax_rows(self._weighted_log_prob(X))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        """Posterior component responsibilities (n_samples, n_components)."""
        var normalized = log_responsibilities(self._weighted_log_prob(X))
        return exp_array(normalized[1])

    def score_samples(self, X: Self.Input) raises -> Self.SampleScores:
        """Per-sample logsumexp of expected weighted log probabilities."""
        return logsumexp_rows(self._weighted_log_prob(X))

    def score(self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Float64:
        """Mean of score_samples (not a sum)."""
        return mean_vector(self.score_samples(X))

    def sample(self, n_samples: Int = 1) raises -> Self.Samples:
        """Draw (points (m, d) Float64, labels (m,) Int64) from the fitted mixture."""
        var rng = RandomState(self.random_state.value()) if self.random_state else entropy_state()
        return self.sample_with_rng(n_samples, rng)

    def sample_with_rng(self, n_samples: Int, mut rng: RandomState) raises -> Self.Samples:
        """Sample, borrowing and advancing a caller-owned RandomState."""
        if not self._fitted:
            raise Error("NotFittedError: This BayesianGaussianMixture instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return sample_gaussian_mixture(self.weights_, self.means_, self.covariances_, self.covariance_type, n_samples, self.n_components, rng)
