"""GaussianMixture: dense EM for full, tied, diag and spherical covariances.

Native port of sklearn.mixture.GaussianMixture (pinned 4358542a). The EM
lifecycle lives in `moxie.mixture._em`; numerical kernels in
`moxie.mixture._gaussian_kernels` (re-exported here for the Bayesian mixture),
`_precision`, `_responsibilities` and `_initialization`.

Mojo adaptations (see docs/mixture/gaussian/api.md):
- `GaussianMixture[dtype]` is parametrized by Float32/Float64 at compile time;
  X, init arrays and learned arrays share that dtype. `sample` returns Float64
  points like the reference.
- `random_state: Optional[Int]`: an integer reseeds at every point sklearn
  calls check_random_state; None uses a fresh `entropy_state()` per call.
  `fit_with_rng` / `fit_predict_with_rng` / `sample_with_rng` borrow and
  advance a caller-owned RandomState (sklearn's RandomState-instance mode).
- fit mutates in place; learned state is committed only after a successful fit.
- The nonconvergence ConvergenceWarning is printed to stderr and kept in
  `convergence_warning_` (empty when none).
"""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, abs
from moxie._random import RandomState, entropy_state
from moxie.mixture._base import MixtureDensity, InformationCriteria
from moxie.mixture._contracts import EMControl, check_input, covariance_shape
from moxie.mixture._em import MixtureEMModel, EMSettings, run_em, weighted_log_prob
from moxie.mixture._gaussian_kernels import (
    GaussianParameters, estimate_gaussian_parameters, estimate_log_gaussian_prob, rounded,
)
from moxie.mixture._precision import (
    compute_precision_cholesky, precision_cholesky_from_precisions, log_det_cholesky,
    precisions_from_cholesky, check_precisions, compact,
)
from moxie.mixture._responsibilities import (
    log_responsibilities, logsumexp_rows, exp_array, mean_vector, argmax_rows,
)
from moxie.mixture._scalar_math import log64


struct GaussianState[dtype: DType](Copyable, Movable):
    """Learned arrays restored for the best restart (`_get_parameters`)."""

    var weights: NDArray[Self.dtype]
    var means: NDArray[Self.dtype]
    var covariances: NDArray[Self.dtype]
    var precisions_cholesky: NDArray[Self.dtype]
    var has_covariances: Bool

    def __init__(out self, var weights: NDArray[Self.dtype], var means: NDArray[Self.dtype], var covariances: NDArray[Self.dtype], var precisions_cholesky: NDArray[Self.dtype], has_covariances: Bool):
        self.weights = weights^
        self.means = means^
        self.covariances = covariances^
        self.precisions_cholesky = precisions_cholesky^
        self.has_covariances = has_covariances


def _empty[dtype: DType]() raises -> NDArray[dtype]:
    return zeros[dtype](Shape(0))


def _normalized_weights[dtype: DType](nk: NDArray[dtype]) -> NDArray[dtype]:
    """weights / xp.sum(weights) in the dtype."""
    var out = nk.copy()
    var total = 0.0
    for j in range(nk.size):
        total += Float64(nk.unsafe_get(j))
    var denominator = rounded[dtype](total)
    for j in range(nk.size):
        out.unsafe_set(j, (Float64(nk.unsafe_get(j)) / denominator).cast[dtype]())
    return out^


struct GaussianEngine[dtype_: DType](MixtureEMModel, Copyable):
    """EM hooks for GaussianMixture (`_initialize`, `_m_step`, estimators)."""

    comptime dtype = Self.dtype_
    comptime State = GaussianState[Self.dtype_]

    var covariance_type: String
    var reg_covar: Float64
    var weights_init: Optional[NDArray[Self.dtype]]
    var means_init: Optional[NDArray[Self.dtype]]
    var precisions_init: Optional[NDArray[Self.dtype]]
    var state: GaussianState[Self.dtype]

    def __init__(out self, covariance_type: String, reg_covar: Float64, var weights_init: Optional[NDArray[Self.dtype]], var means_init: Optional[NDArray[Self.dtype]], var precisions_init: Optional[NDArray[Self.dtype]], var state: GaussianState[Self.dtype]):
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.weights_init = weights_init^
        self.means_init = means_init^
        self.precisions_init = precisions_init^
        self.state = state^

    def needs_responsibilities(self) -> Bool:
        return not (Bool(self.weights_init) and Bool(self.means_init) and Bool(self.precisions_init))

    def initialize(mut self, x: NDArray[Self.dtype], resp: NDArray[Self.dtype]) raises:
        var n = x.shape[0]
        if resp.shape[0] > 0:
            var params = estimate_gaussian_parameters(x, resp, self.reg_covar, self.covariance_type)
            if self.weights_init:
                self.state.weights = self.weights_init.value().copy()
            else:
                var weights = params.nk.copy()
                for j in range(weights.size):
                    weights.unsafe_set(j, weights.unsafe_get(j) / Scalar[Self.dtype](n))
                self.state.weights = weights^
            if self.means_init:
                self.state.means = self.means_init.value().copy()
            else:
                self.state.means = params.means.copy()
            if self.precisions_init:
                self.state.precisions_cholesky = precision_cholesky_from_precisions(self.precisions_init.value(), self.covariance_type)
            else:
                self.state.precisions_cholesky = compute_precision_cholesky(params.covariances, self.covariance_type)
                self.state.covariances = params.covariances.copy()
                self.state.has_covariances = True
        else:
            self.state.weights = self.weights_init.value().copy()
            self.state.means = self.means_init.value().copy()
            self.state.precisions_cholesky = precision_cholesky_from_precisions(self.precisions_init.value(), self.covariance_type)

    def estimate_log_prob(self, x: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        var log_det = log_det_cholesky(self.state.precisions_cholesky, self.covariance_type, x.shape[1])
        return estimate_log_gaussian_prob(x, self.state.means, self.state.precisions_cholesky, self.covariance_type, log_det)

    def estimate_log_weights(self) raises -> NDArray[Self.dtype]:
        var out = self.state.weights.copy()
        for j in range(out.size):
            out.unsafe_set(j, log64(Float64(out.unsafe_get(j))).cast[Self.dtype]())
        return out^

    def m_step(mut self, x: NDArray[Self.dtype], log_resp: NDArray[Self.dtype]) raises:
        var params = estimate_gaussian_parameters(x, exp_array(log_resp), self.reg_covar, self.covariance_type)
        self.state.weights = _normalized_weights(params.nk)
        self.state.means = params.means.copy()
        self.state.precisions_cholesky = compute_precision_cholesky(params.covariances, self.covariance_type)
        self.state.covariances = params.covariances.copy()
        self.state.has_covariances = True

    def lower_bound(self, log_resp: NDArray[Self.dtype], mean_log_prob: Float64) raises -> Float64:
        return mean_log_prob

    def get_state(self) raises -> Self.State:
        return self.state.copy()

    def set_state(mut self, var state: Self.State) raises:
        self.state = state^


@fieldwise_init
struct GaussianMixtureParams[dtype: DType](Copyable, Movable):
    """Typed constructor configuration (get_params/set_params record)."""

    var n_components: Int
    var covariance_type: String
    var tol: Float64
    var reg_covar: Float64
    var max_iter: Int
    var n_init: Int
    var init_params: String
    var weights_init: Optional[NDArray[Self.dtype]]
    var means_init: Optional[NDArray[Self.dtype]]
    var precisions_init: Optional[NDArray[Self.dtype]]
    var random_state: Optional[Int]
    var warm_start: Bool
    var verbose: Int
    var verbose_interval: Int


def _check_weights[dtype: DType](weights: NDArray[dtype], n_components: Int) raises -> NDArray[dtype]:
    """sklearn `_check_weights`: shape (k,), values in [0, 1], normalized."""
    if weights.ndim != 1 or weights.shape[0] != n_components:
        raise Error(String("ValueError: The parameter 'weights' should have the shape of (", n_components, ",), but got ", weights.shape))
    var values = compact(weights)
    var minimum = Float64(values.unsafe_get(0))
    var maximum = minimum
    var total = Scalar[dtype](0)
    for j in range(n_components):
        var w = Float64(values.unsafe_get(j))
        if not (w == w) or w > 1.7976931348623157e308 or w < -1.7976931348623157e308:
            raise Error("ValueError: Input weights contains NaN or infinity.")
        if w < minimum:
            minimum = w
        if w > maximum:
            maximum = w
        total += values.unsafe_get(j)
    if minimum < 0.0 or maximum > 1.0:
        raise Error(String("ValueError: The parameter 'weights' should be in the range [0, 1], but got max value ", maximum, ", min value ", minimum))
    var atol = 1e-6 if dtype == DType.float32 else 1e-8
    # np.allclose(|1 - sum|, 0, atol): |a - b| <= atol + rtol * |b| with b = 0.
    if not (abs(1.0 - Float64(total)) <= atol):
        raise Error(String("ValueError: The parameter 'weights' should be normalized, but got sum(weights) = ", Float64(total)))
    return values^


def _check_means[dtype: DType](means: NDArray[dtype], n_components: Int, n_features: Int) raises -> NDArray[dtype]:
    if means.ndim != 2 or means.shape[0] != n_components or means.shape[1] != n_features:
        raise Error(String("ValueError: The parameter 'means' should have the shape of (", n_components, ", ", n_features, "), but got ", means.shape))
    var values = compact(means)
    for i in range(values.size):
        var v = Float64(values.unsafe_get(i))
        if not (v == v) or abs(v) > 1.7976931348623157e308:
            raise Error("ValueError: Input means contains NaN or infinity.")
    return values^


struct GaussianMixture[dtype: DType = DType.float64](MixtureDensity, InformationCriteria, Copyable):
    """Gaussian mixture model fitted by expectation-maximization.

    Parameters, defaults and learned attributes follow
    sklearn.mixture.GaussianMixture. Learned arrays are empty before fit.
    """

    comptime Params = GaussianMixtureParams[Self.dtype]
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Prediction = NDArray[DType.int64]
    comptime Probabilities = NDArray[Self.dtype]
    comptime SampleScores = NDArray[Self.dtype]
    comptime Samples = Tuple[NDArray[DType.float64], NDArray[DType.int64]]

    # Constructor parameters.
    var n_components: Int
    var covariance_type: String
    var tol: Float64
    var reg_covar: Float64
    var max_iter: Int
    var n_init: Int
    var init_params: String
    var weights_init: Optional[NDArray[Self.dtype]]
    var means_init: Optional[NDArray[Self.dtype]]
    var precisions_init: Optional[NDArray[Self.dtype]]
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
    var n_features_in_: Int
    var convergence_warning_: String
    var _fitted: Bool

    def __init__(
        out self,
        n_components: Int = 1,
        *,
        covariance_type: String = "full",
        tol: Float64 = 1e-3,
        reg_covar: Float64 = 1e-6,
        max_iter: Int = 100,
        n_init: Int = 1,
        init_params: String = "kmeans",
        var weights_init: Optional[NDArray[Self.dtype]] = None,
        var means_init: Optional[NDArray[Self.dtype]] = None,
        var precisions_init: Optional[NDArray[Self.dtype]] = None,
        random_state: Optional[Int] = None,
        warm_start: Bool = False,
        verbose: Int = 0,
        verbose_interval: Int = 10,
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "GaussianMixture supports Float32 and Float64"
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init^
        self.means_init = means_init^
        self.precisions_init = precisions_init^
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
        self.n_features_in_ = 0
        self.convergence_warning_ = String("")
        self._fitted = False

    # ----- configuration -----
    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return GaussianMixtureParams[Self.dtype](
            self.n_components, self.covariance_type, self.tol, self.reg_covar, self.max_iter,
            self.n_init, self.init_params, self.weights_init.copy(), self.means_init.copy(),
            self.precisions_init.copy(), self.random_state, self.warm_start, self.verbose,
            self.verbose_interval,
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.covariance_type = params.covariance_type
        self.tol = params.tol
        self.reg_covar = params.reg_covar
        self.max_iter = params.max_iter
        self.n_init = params.n_init
        self.init_params = params.init_params
        self.weights_init = params.weights_init.copy()
        self.means_init = params.means_init.copy()
        self.precisions_init = params.precisions_init.copy()
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
        var weights_init: Optional[NDArray[Self.dtype]] = None,
        var means_init: Optional[NDArray[Self.dtype]] = None,
        var precisions_init: Optional[NDArray[Self.dtype]] = None,
        random_state: Optional[Int] = None,
        warm_start: Optional[Bool] = None,
        verbose: Optional[Int] = None,
        verbose_interval: Optional[Int] = None,
    ) raises -> ref[self] Self:
        """Named partial update (sklearn `set_params(**params)`); learned state is kept.

        Omitted names keep their value; pass a full Params record to reset an
        optional parameter (`*_init`, `random_state`) to None. Values are
        validated at the next fit, as in sklearn.
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
        if weights_init:
            self.weights_init = weights_init^
        if means_init:
            self.means_init = means_init^
        if precisions_init:
            self.precisions_init = precisions_init^
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
        var params = self.get_params()
        return Self(
            params.n_components, covariance_type=params.covariance_type, tol=params.tol,
            reg_covar=params.reg_covar, max_iter=params.max_iter, n_init=params.n_init,
            init_params=params.init_params, weights_init=params.weights_init.copy(),
            means_init=params.means_init.copy(), precisions_init=params.precisions_init.copy(),
            random_state=params.random_state, warm_start=params.warm_start,
            verbose=params.verbose, verbose_interval=params.verbose_interval,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    # ----- fitting -----
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
            raise Error(String("ValueError: invalid GaussianMixture parameters (n_components=", self.n_components,
                               ", covariance_type='", self.covariance_type, "', tol=", self.tol, ", reg_covar=",
                               self.reg_covar, ", max_iter=", self.max_iter, ", n_init=", self.n_init,
                               ", init_params='", self.init_params, "', verbose=", self.verbose,
                               ", verbose_interval=", self.verbose_interval, "): ", error))
        return control^

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        """Estimate model parameters with EM (y is ignored)."""
        _ = self.fit_predict(X, y)
        return self

    def fit_with_rng[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, mut rng: RandomState) raises -> ref[self] Self:
        """Fit, borrowing and advancing a caller-owned RandomState."""
        _ = self.fit_predict_with_rng(X, rng)
        return self

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> NDArray[DType.int64]:
        """Fit and return argmax labels of the final E-step (y is ignored)."""
        var rng = RandomState(self.random_state.value()) if self.random_state else entropy_state()
        return self.fit_predict_with_rng(X, rng)

    def fit_predict_with_rng(mut self, X: Self.Input, mut rng: RandomState) raises -> NDArray[DType.int64]:
        """Fit_predict, borrowing and advancing a caller-owned RandomState."""
        var control = self._control()
        check_input(X, self.n_components, fitting=True)
        var n_features = X.shape[1]
        var weights_init = Optional[NDArray[Self.dtype]](None)
        var means_init = Optional[NDArray[Self.dtype]](None)
        var precisions_init = Optional[NDArray[Self.dtype]](None)
        if self.weights_init:
            weights_init = _check_weights(self.weights_init.value(), self.n_components)
        if self.means_init:
            means_init = _check_means(self.means_init.value(), self.n_components, n_features)
        if self.precisions_init:
            check_precisions(self.precisions_init.value(), self.covariance_type, self.n_components, n_features)
            precisions_init = compact(self.precisions_init.value())
        var do_init = not (self.warm_start and self._fitted)
        var state: GaussianState[Self.dtype]
        var initial_lower_bound = 0.0
        if do_init:
            state = GaussianState[Self.dtype](_empty[Self.dtype](), _empty[Self.dtype](), _empty[Self.dtype](), _empty[Self.dtype](), False)
        else:
            if n_features != self.n_features_in_:
                raise Error(String("ValueError: X has ", n_features, " features, but GaussianMixture is expecting ", self.n_features_in_, " features as input."))
            state = GaussianState[Self.dtype](self.weights_.copy(), self.means_.copy(), self.covariances_.copy(), self.precisions_cholesky_.copy(), True)
            initial_lower_bound = self.lower_bound_
        var engine = GaussianEngine[Self.dtype](self.covariance_type, self.reg_covar, weights_init^, means_init^, precisions_init^, state^)
        var settings = EMSettings(self.n_components, self.tol, self.max_iter, control.restart_count(self._fitted),
                                  self.init_params, self.verbose, self.verbose_interval, do_init, initial_lower_bound)
        var result = run_em(engine, X, settings, rng)
        if not engine.state.has_covariances:
            raise Error("AttributeError: 'GaussianMixture' object has no attribute 'covariances_' (precisions_init with max_iter=0 never estimates covariances, as in the reference)")
        # Commit learned state only after success (`_set_parameters`).
        self.weights_ = engine.state.weights.copy()
        self.means_ = engine.state.means.copy()
        self.covariances_ = engine.state.covariances.copy()
        self.precisions_cholesky_ = engine.state.precisions_cholesky.copy()
        self.precisions_ = precisions_from_cholesky(self.precisions_cholesky_, self.covariance_type)
        self.converged_ = result.converged
        self.n_iter_ = result.n_iter
        self.lower_bound_ = result.lower_bound
        self.lower_bounds_ = result.lower_bounds.copy()
        self.n_features_in_ = n_features
        self.convergence_warning_ = result.warning
        self._fitted = True
        return result.labels.copy()

    # ----- inference -----
    def _check_inference(self, X: Self.Input) raises:
        if not self._fitted:
            raise Error("NotFittedError: This GaussianMixture instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        check_input(X, 1, fitting=False, features=self.n_features_in_)

    def _engine(self) raises -> GaussianEngine[Self.dtype]:
        var state = GaussianState[Self.dtype](self.weights_.copy(), self.means_.copy(), self.covariances_.copy(), self.precisions_cholesky_.copy(), True)
        return GaussianEngine[Self.dtype](self.covariance_type, self.reg_covar, None, None, None, state^)

    def _weighted_log_prob(self, X: Self.Input) raises -> NDArray[Self.dtype]:
        self._check_inference(X)
        return weighted_log_prob(self._engine(), X)

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        """Component labels: argmax of weighted log probabilities."""
        return argmax_rows(self._weighted_log_prob(X))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        """Posterior component probabilities (n_samples, n_components)."""
        var normalized = log_responsibilities(self._weighted_log_prob(X))
        return exp_array(normalized[1])

    def score_samples(self, X: Self.Input) raises -> Self.SampleScores:
        """Per-sample log-likelihood log p(x)."""
        return logsumexp_rows(self._weighted_log_prob(X))

    def score(self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Float64:
        """Mean per-sample log-likelihood (not a sum)."""
        return mean_vector(self.score_samples(X))

    def _n_parameters(self) raises -> Int:
        var d = Float64(self.means_.shape[1]) if self.means_.ndim == 2 else 0.0
        var k = Float64(self.n_components)
        var cov_params: Float64
        if self.covariance_type == "full":
            cov_params = k * d * (d + 1.0) / 2.0
        elif self.covariance_type == "diag":
            cov_params = k * d
        elif self.covariance_type == "tied":
            cov_params = d * (d + 1.0) / 2.0
        else:
            cov_params = k
        return Int(cov_params + d * k + k - 1.0)

    def bic(self, X: Self.Input) raises -> Float64:
        """Bayesian information criterion: -2 log L + p log n."""
        var n = Float64(X.shape[0])
        return -2.0 * self.score(X) * n + Float64(self._n_parameters()) * log64(n)

    def aic(self, X: Self.Input) raises -> Float64:
        """Akaike information criterion: -2 log L + 2 p."""
        var n = Float64(X.shape[0])
        return -2.0 * self.score(X) * n + 2.0 * Float64(self._n_parameters())

    def sample(self, n_samples: Int = 1) raises -> Self.Samples:
        """Draw (points (m, d) Float64, labels (m,) Int64), grouped by component."""
        var rng = RandomState(self.random_state.value()) if self.random_state else entropy_state()
        return self.sample_with_rng(n_samples, rng)

    def sample_with_rng(self, n_samples: Int, mut rng: RandomState) raises -> Self.Samples:
        """Sample, borrowing and advancing a caller-owned RandomState."""
        if not self._fitted:
            raise Error("NotFittedError: This GaussianMixture instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return sample_gaussian_mixture(self.weights_, self.means_, self.covariances_, self.covariance_type, n_samples, self.n_components, rng)


def sample_gaussian_mixture[dtype: DType](weights: NDArray[dtype], means: NDArray[dtype], covariances: NDArray[dtype], kind: String, n_samples: Int, n_components: Int, mut rng: RandomState) raises -> Tuple[NDArray[DType.float64], NDArray[DType.int64]]:
    """BaseMixture.sample: multinomial counts, then per-component draws.

    Full/tied use RandomState.multivariate_normal on the symmetric part of the
    covariance (SVD factor; exact values can differ from NumPy's LAPACK SVD by
    factor orientation, while draw consumption and Mahalanobis norms match);
    diag/spherical use
    mean + standard_normal * sqrt(covariance).
    """
    if n_samples < 1:
        raise Error(String("ValueError: Invalid value for 'n_samples': ", n_components, " . The sampling requires at least one sample."))
    var k = means.shape[0]
    var d = means.shape[1]
    var p = zeros[DType.float64](Shape(k))
    for j in range(k):
        p.unsafe_set(j, Float64(weights.unsafe_get(j)))
    var counts = rng.multinomial(n_samples, p, 1)
    var points = zeros[DType.float64](Shape(n_samples, d))
    var labels = zeros[DType.int64](Shape(n_samples))
    var row = 0
    for j in range(k):
        var count = Int(counts.unsafe_get(j))
        var mean = zeros[dtype](Shape(d))
        for f in range(d):
            mean.unsafe_set(f, means.unsafe_get(j * d + f))
        if kind == "full" or kind == "tied":
            # The expanded covariance formulas are symmetric only up to dtype
            # rounding; NumPy warns-and-continues on that, while the native
            # sampler validates symmetry at 1e-8. Sample from the symmetric part.
            var cov = zeros[DType.float64](Shape(d, d))
            var base = j * d * d if kind == "full" else 0
            for a in range(d):
                for b in range(d):
                    var upper = Float64(covariances.unsafe_get(base + a * d + b))
                    var lower = Float64(covariances.unsafe_get(base + b * d + a))
                    cov.unsafe_set(a * d + b, 0.5 * (upper + lower))
            var draws = rng.multivariate_normal(mean, cov, count)
            for i in range(count):
                for f in range(d):
                    points.unsafe_set((row + i) * d + f, draws.unsafe_get(i * d + f))
        else:
            for i in range(count):
                for f in range(d):
                    var variance = covariances.unsafe_get(j * d + f) if kind == "diag" else covariances.unsafe_get(j)
                    var scale = Float64(sqrt(variance))
                    points.unsafe_set((row + i) * d + f, Float64(mean.unsafe_get(f)) + rng.standard_normal() * scale)
        for i in range(count):
            labels.unsafe_set(row + i, Int64(j))
        row += count
    return (points^, labels^)
