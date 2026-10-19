# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Gaussian process regression (sklearn.gaussian_process.GaussianProcessRegressor).

Algorithm 2.1 of Rasmussen & Williams (2006): Cholesky of K + alpha I, posterior
mean/covariance by triangular solves, log marginal likelihood and its analytic
log-theta gradient, hyperparameters by L-BFGS-B with optional random restarts.
All numerical work uses Float64 (see docs/api.md for Float32 input).
"""
from numojo import NDArray, Shape, zeros
from std.math import isfinite, isnan, pi, sqrt
from std.sys import stderr
from std.utils import Variant

from moxie._random import RandomState, check_random_state
from moxie.gaussian_process._base import (
    GaussianProcessPosterior,
    GaussianProcessRegressorAPI,
    GaussianUncertainty,
)
from moxie.gaussian_process._plumbing import (
    Dense64,
    HyperparameterOptimizer,
    LikelihoodObjective,
    ObjectiveEvaluation,
    OptimizationResult,
    PredictionResult,
    validate_uncertainty,
)
from moxie.gaussian_process._algebra import CholeskyAlgebra, identity
from moxie.gaussian_process._kernel_math import as_dense64, flog
from moxie.gaussian_process._optimization import (
    LBFGSBOptimizer,
    check_restart_bounds,
    restart_initial_theta,
    select_best,
)
from moxie.gaussian_process.kernels import ConstantKernel, Kernel, RBF

comptime ScalarOrVector = Variant[Float64, List[Float64]]
"""sklearn `alpha`: a float or one value per training sample."""


def _warn(message: String):
    print("UserWarning: " + message, file=stderr)


# ===----------------------------------------------------------------------=== #
# Log marginal likelihood
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct LikelihoodResult(Movable):
    """Log marginal likelihood and, when requested, its log-theta gradient."""

    var value: Float64
    var gradient: Optional[Dense64]


def _diagonal_noise(alpha: ScalarOrVector, n: Int) raises -> List[Float64]:
    var out = List[Float64]()
    if alpha.isa[Float64]():
        for _ in range(n):
            out.append(alpha[Float64])
        return out^
    ref values = alpha[List[Float64]]
    if len(values) == 1:
        for _ in range(n):
            out.append(values[0])
        return out^
    if len(values) != n:
        raise Error(
            "ValueError: alpha must be a scalar or an array with same number of"
            " entries as y. ("
            + String(len(values))
            + " != "
            + String(n)
            + ")"
        )
    return values.copy()


def log_marginal_likelihood_of(
    kernel: Kernel,
    X_train: Dense64,
    y_train: Dense64,
    noise: List[Float64],
    eval_gradient: Bool,
) raises -> LikelihoodResult:
    """sklearn `GaussianProcessRegressor.log_marginal_likelihood` body.

    y_train is `(n, t)`; returns `-inf` (and a zero gradient) when K + alpha I
    is not positive definite, exactly like sklearn.
    """
    var n = X_train.shape[0]
    var t = y_train.shape[1]
    var evaluation = kernel.evaluate(X_train, eval_gradient=eval_gradient)
    var K = evaluation.covariance.copy()
    for i in range(n):
        K.unsafe_set(i * n + i, K.unsafe_get(i * n + i) + noise[i])
    var algebra = CholeskyAlgebra()
    var L: Dense64
    try:
        L = algebra.cholesky(K)
    except:
        var p = kernel.n_dims()
        if eval_gradient:
            return LikelihoodResult(value=-Float64.MAX * 2.0, gradient=zeros[DType.float64](Shape(p)))
        return LikelihoodResult(value=-Float64.MAX * 2.0, gradient=None)
    var alpha = algebra.cho_solve(L, y_train)
    var total: Float64 = 0.0
    var log_diag: Float64 = 0.0
    for i in range(n):
        log_diag += flog(L.unsafe_get(i * n + i))
    for k in range(t):
        var fit: Float64 = 0.0
        for i in range(n):
            fit += y_train.unsafe_get(i * t + k) * alpha.unsafe_get(i * t + k)
        total += -0.5 * fit - log_diag - Float64(n) / 2.0 * flog(2.0 * pi)
    if not eval_gradient:
        return LikelihoodResult(value=total, gradient=None)
    ref G = evaluation.gradient.value()
    var p = G.shape[2]
    var K_inv = algebra.cho_solve(L, identity(n))
    var gradient = zeros[DType.float64](Shape(p))
    var g = G.unsafe_ptr()
    for i in range(n):
        for j in range(n):
            # W_ij = sum_k alpha_ik alpha_jk - t * Kinv_ij; grad_p += W_ij G_jip
            var w: Float64 = 0.0
            for k in range(t):
                w += alpha.unsafe_get(i * t + k) * alpha.unsafe_get(j * t + k)
            w -= Float64(t) * K_inv.unsafe_get(i * n + j)
            var base = (j * n + i) * p
            for c in range(p):
                gradient.unsafe_set(c, gradient.unsafe_get(c) + w * g[unsafe_offset = base + c])
    for c in range(p):
        gradient.unsafe_set(c, 0.5 * gradient.unsafe_get(c))
    return LikelihoodResult(value=total, gradient=gradient^)


struct LogMarginalLikelihood(Copyable, LikelihoodObjective, Movable):
    """Optimizer objective: negative LML of the training data at log theta.

    Mirrors sklearn's `obj_func(theta, eval_gradient=True)`; the kernel's
    theta is replaced on each call (clone_kernel=False semantics).
    """

    var kernel: Kernel
    var X_train: Dense64
    var y_train: Dense64
    var noise: List[Float64]
    var evaluations: Int

    def __init__(
        out self, var kernel: Kernel, X_train: Dense64, y_train: Dense64, var noise: List[Float64]
    ):
        self.kernel = kernel^
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.noise = noise^
        self.evaluations = 0

    def evaluate(mut self, theta: Dense64, eval_gradient: Bool = True) raises -> ObjectiveEvaluation:
        self.evaluations += 1
        self.kernel.set_theta(theta)
        var result = log_marginal_likelihood_of(
            self.kernel, self.X_train, self.y_train, self.noise, eval_gradient
        )
        if not eval_gradient:
            return ObjectiveEvaluation(value=-result.value, gradient=None)
        var gradient = result.gradient.value().copy()
        for i in range(gradient.size):
            gradient.unsafe_set(i, -gradient.unsafe_get(i))
        return ObjectiveEvaluation(value=-result.value, gradient=gradient^)


trait GPROptimizer(Defaultable, Deinitable, ImplicitlyCopyable, Movable):
    """Native replacement for sklearn's optimizer callable.

    The family `HyperparameterOptimizer` contract specialized to the GPR
    objective: minimize the negative LML from `initial_theta` within log
    `bounds` and return the optimum theta and minimized value.
    """

    def minimize(
        self, mut objective: LogMarginalLikelihood, initial_theta: Dense64, bounds: Dense64
    ) raises -> OptimizationResult:
        ...


struct FminLBFGSB(GPROptimizer, HyperparameterOptimizer, ImplicitlyCopyable):
    """sklearn `optimizer="fmin_l_bfgs_b"` (SciPy L-BFGS-B defaults)."""

    comptime Objective = LogMarginalLikelihood
    var engine: LBFGSBOptimizer[LogMarginalLikelihood]

    def __init__(out self):
        self.engine = LBFGSBOptimizer[LogMarginalLikelihood]()

    def minimize(
        self, mut objective: LogMarginalLikelihood, initial_theta: Dense64, bounds: Dense64
    ) raises -> OptimizationResult:
        return self.engine.minimize(objective, initial_theta, bounds)


# ===----------------------------------------------------------------------=== #
# Estimator
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct GaussianProcessRegressorParams[Optimizer: GPROptimizer](Copyable, Movable):
    """Constructor configuration (sklearn get_params keys)."""

    var kernel: Optional[Kernel]
    var alpha: ScalarOrVector
    var optimizer: Optional[Self.Optimizer]
    var n_restarts_optimizer: Int
    var normalize_y: Bool
    var copy_X_train: Bool
    var n_targets: Optional[Int]
    var random_state: Optional[Int]


def default_kernel() -> Kernel:
    """sklearn's `C() * RBF()` used when kernel=None."""
    return ConstantKernel(1.0) * RBF(1.0)


def _prior_kernel() -> Kernel:
    return ConstantKernel(1.0, "fixed") * RBF(1.0, "fixed")


def _target_matrix[dtype: DType](y: NDArray[dtype]) raises -> Dense64:
    if y.ndim == 1:
        var out = zeros[DType.float64](Shape(y.shape[0], 1))
        var stride = Int(y.strides[0])
        for i in range(y.shape[0]):
            out.unsafe_set(i, y.unsafe_get(i * stride).cast[DType.float64]())
        return out^
    if y.ndim != 2:
        raise Error("ValueError: y must be 1-D or 2-D, got rank " + String(y.ndim))
    return as_dense64(y, "GaussianProcessRegressor y")


def _check_finite(A: Dense64, name: String) raises:
    for i in range(A.size):
        var v = A.unsafe_get(i)
        if isnan(v):
            raise Error("ValueError: Input " + name + " contains NaN.")
        if not isfinite(v):
            raise Error("ValueError: Input " + name + " contains infinity or a value too large for dtype('float64').")


def _squeeze_columns(A: Dense64) raises -> Dense64:
    """(n, 1) -> (n,); other shapes unchanged."""
    if A.ndim == 2 and A.shape[1] == 1:
        var out = zeros[DType.float64](Shape(A.shape[0]))
        for i in range(A.shape[0]):
            out.unsafe_set(i, A.unsafe_get(i))
        return out^
    return A.copy()


comptime GaussianProcessRegressor = GenericGaussianProcessRegressor[DType.float64, FminLBFGSB]
"""sklearn `GaussianProcessRegressor` for Float64 input with the default optimizer.

Use `GenericGaussianProcessRegressor[DType.float32]` for Float32 input and
`GenericGaussianProcessRegressor[dtype, MyOptimizer]` for a native custom
optimizer (a `GPROptimizer`).
"""


struct GenericGaussianProcessRegressor[
    dtype: DType, Optimizer: GPROptimizer = FminLBFGSB
](Copyable, GaussianProcessPosterior, GaussianProcessRegressorAPI, GaussianUncertainty):
    """Gaussian process regression with sklearn's constructor defaults.

    Learned attributes: `kernel_`, `X_train_`, `y_train_` (normalized Float64
    targets `(n,)` or `(n, t)`), `L_`, `alpha_`, `log_marginal_likelihood_value_`,
    `n_features_in_`, plus `convergence_warnings_` (the warnings sklearn emits
    during fit). Prediction before fit uses the GP prior, as in sklearn.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Prediction = Dense64
    comptime SampleWeight = List[Float64]
    comptime Params = GaussianProcessRegressorParams[Self.Optimizer]
    comptime Draws = Dense64
    comptime Theta = Dense64
    comptime Likelihood = LikelihoodResult
    comptime Uncertainty = PredictionResult

    var kernel: Optional[Kernel]
    var alpha: ScalarOrVector
    var optimizer: Optional[Self.Optimizer]
    var n_restarts_optimizer: Int
    var normalize_y: Bool
    var copy_X_train: Bool
    var n_targets: Optional[Int]
    var random_state: Optional[Int]

    var kernel_: Kernel
    var X_train_: NDArray[Self.dtype]
    var y_train_: Dense64
    var L_: Dense64
    var alpha_: Dense64
    var log_marginal_likelihood_value_: Float64
    var n_features_in_: Int
    var convergence_warnings_: List[String]
    var _y_train_mean: List[Float64]
    var _y_train_std: List[Float64]
    var _target_matrix: Dense64
    var _y_is_vector: Bool
    var _fitted: Bool

    def __init__(
        out self,
        kernel: Optional[Kernel] = None,
        *,
        alpha: ScalarOrVector = 1e-10,
        optimizer: Optional[Self.Optimizer] = Self.Optimizer(),
        n_restarts_optimizer: Int = 0,
        normalize_y: Bool = False,
        copy_X_train: Bool = True,
        n_targets: Optional[Int] = None,
        random_state: Optional[Int] = None,
    ) raises:
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        ), "GaussianProcessRegressor supports Float32 and Float64 input"
        self.kernel = kernel.copy()
        self.alpha = alpha.copy()
        self.optimizer = optimizer.copy()
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.n_targets = n_targets
        self.random_state = random_state
        self.kernel_ = default_kernel()
        self.X_train_ = zeros[Self.dtype](Shape(0, 0))
        self.y_train_ = zeros[DType.float64](Shape(0))
        self.L_ = zeros[DType.float64](Shape(0, 0))
        self.alpha_ = zeros[DType.float64](Shape(0))
        self.log_marginal_likelihood_value_ = 0.0
        self.n_features_in_ = 0
        self.convergence_warnings_ = List[String]()
        self._y_train_mean = List[Float64]()
        self._y_train_std = List[Float64]()
        self._target_matrix = zeros[DType.float64](Shape(0, 0))
        self._y_is_vector = True
        self._fitted = False
        self._validate_params()

    def __init__(
        out self,
        kernel: Kernel,
        *,
        alpha: ScalarOrVector = 1e-10,
        optimizer: Optional[Self.Optimizer] = Self.Optimizer(),
        n_restarts_optimizer: Int = 0,
        normalize_y: Bool = False,
        copy_X_train: Bool = True,
        n_targets: Optional[Int] = None,
        random_state: Optional[Int] = None,
    ) raises:
        """Construct with an explicit kernel (copied; sklearn clones at fit)."""
        self = Self(
            Optional[Kernel](kernel.copy()),
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            n_targets=n_targets,
            random_state=random_state,
        )

    def __init__(out self, *, copy: Self):
        self.kernel = copy.kernel.copy()
        self.alpha = copy.alpha.copy()
        self.optimizer = copy.optimizer.copy()
        self.n_restarts_optimizer = copy.n_restarts_optimizer
        self.normalize_y = copy.normalize_y
        self.copy_X_train = copy.copy_X_train
        self.n_targets = copy.n_targets
        self.random_state = copy.random_state
        self.kernel_ = copy.kernel_.copy()
        self.X_train_ = copy.X_train_.copy()
        self.y_train_ = copy.y_train_.copy()
        self.L_ = copy.L_.copy()
        self.alpha_ = copy.alpha_.copy()
        self.log_marginal_likelihood_value_ = copy.log_marginal_likelihood_value_
        self.n_features_in_ = copy.n_features_in_
        self.convergence_warnings_ = copy.convergence_warnings_.copy()
        self._y_train_mean = copy._y_train_mean.copy()
        self._y_train_std = copy._y_train_std.copy()
        self._target_matrix = copy._target_matrix.copy()
        self._y_is_vector = copy._y_is_vector
        self._fitted = copy._fitted

    def _validate_params(self) raises:
        if self.n_restarts_optimizer < 0:
            raise Error(
                "InvalidParameterError: The 'n_restarts_optimizer' parameter of"
                " GaussianProcessRegressor must be an int in the range [0, inf). Got "
                + String(self.n_restarts_optimizer)
                + " instead."
            )
        if self.n_targets and self.n_targets.value() < 1:
            raise Error(
                "InvalidParameterError: The 'n_targets' parameter of"
                " GaussianProcessRegressor must be an int in the range [1, inf) or"
                " None. Got "
                + String(self.n_targets.value())
                + " instead."
            )
        if self.alpha.isa[Float64]():
            var a = self.alpha[Float64]
            if isnan(a) or a < 0:
                raise Error(
                    "InvalidParameterError: The 'alpha' parameter of"
                    " GaussianProcessRegressor must be a float in the range [0, inf)"
                    " or an array. Got "
                    + String(a)
                    + " instead."
                )
        if self.random_state:
            var seed = self.random_state.value()
            if seed < 0 or seed > 4294967295:
                raise Error("ValueError: random_state must be None or an int in [0, 2**32 - 1]")

    # --- BaseEstimator ------------------------------------------------------

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            kernel=self.kernel.copy(),
            alpha=self.alpha.copy(),
            optimizer=self.optimizer.copy(),
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            copy_X_train=self.copy_X_train,
            n_targets=self.n_targets,
            random_state=self.random_state,
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var previous = self.get_params()
        self.kernel = params.kernel.copy()
        self.alpha = params.alpha.copy()
        self.optimizer = params.optimizer.copy()
        self.n_restarts_optimizer = params.n_restarts_optimizer
        self.normalize_y = params.normalize_y
        self.copy_X_train = params.copy_X_train
        self.n_targets = params.n_targets
        self.random_state = params.random_state
        try:
            self._validate_params()
        except e:
            self.set_params_unchecked(previous)
            raise e^
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        kernel: Optional[Optional[Kernel]] = None,
        alpha: Variant[NoneType, Float64, List[Float64]] = NoneType(),
        optimizer: Optional[Optional[Self.Optimizer]] = None,
        n_restarts_optimizer: Optional[Int] = None,
        normalize_y: Optional[Bool] = None,
        copy_X_train: Optional[Bool] = None,
        n_targets: Optional[Optional[Int]] = None,
        random_state: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        """Update supplied constructor parameters, preserving learned state.

        For nullable fields, an absent outer Optional leaves the field unchanged;
        a present outer Optional containing None clears the field.
        """
        var params = self.get_params()
        if kernel:
            params.kernel = kernel.value().copy()
        if alpha.isa[Float64]():
            params.alpha = alpha[Float64]
        elif alpha.isa[List[Float64]]():
            params.alpha = alpha[List[Float64]].copy()
        if optimizer:
            params.optimizer = optimizer.value().copy()
        if n_restarts_optimizer:
            params.n_restarts_optimizer = n_restarts_optimizer.value()
        if normalize_y:
            params.normalize_y = normalize_y.value()
        if copy_X_train:
            params.copy_X_train = copy_X_train.value()
        if n_targets:
            params.n_targets = n_targets.value()
        if random_state:
            params.random_state = random_state.value()
        return self.set_params(params)

    def set_params_unchecked(mut self, params: Self.Params):
        self.kernel = params.kernel.copy()
        self.alpha = params.alpha.copy()
        self.optimizer = params.optimizer.copy()
        self.n_restarts_optimizer = params.n_restarts_optimizer
        self.normalize_y = params.normalize_y
        self.copy_X_train = params.copy_X_train
        self.n_targets = params.n_targets
        self.random_state = params.random_state

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            self.kernel.copy(),
            alpha=self.alpha.copy(),
            optimizer=self.optimizer.copy(),
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            copy_X_train=self.copy_X_train,
            n_targets=self.n_targets,
            random_state=self.random_state,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    # --- fit ------------------------------------------------------------------

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        """Fit with a fresh seeded stream or the shared default RandomState."""
        with check_random_state(self.random_state) as rng:
            _ = self.fit(X, y, rng[])
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, mut random_state: RandomState) raises -> ref[self] Self:
        """Fit using (and advancing) a borrowed RandomState for restarts."""
        self._validate_params()
        var kernel = self.kernel.value().copy() if self.kernel else default_kernel()
        if X.ndim != 2:
            raise Error(
                "ValueError: Expected 2D array, got "
                + String(X.ndim)
                + "D array instead. Reshape your data."
            )
        var Xd = as_dense64(X, "GaussianProcessRegressor.fit X")
        var n = Xd.shape[0]
        if n < 1 or Xd.shape[1] < 1:
            raise Error(
                "ValueError: Found array with "
                + String(n)
                + " sample(s) and "
                + String(Xd.shape[1])
                + " feature(s) while a minimum of 1 is required by GaussianProcessRegressor."
            )
        _check_finite(Xd, "X")
        var Y = _target_matrix(y)
        if Y.shape[0] != n:
            raise Error(
                "ValueError: Found input variables with inconsistent numbers of samples: ["
                + String(n)
                + ", "
                + String(Y.shape[0])
                + "]"
            )
        _check_finite(Y, "y")
        var t = Y.shape[1]
        if self.n_targets and t != self.n_targets.value():
            raise Error(
                "ValueError: The number of targets seen in `y` is different from the"
                " parameter `n_targets`. Got "
                + String(t)
                + " != "
                + String(self.n_targets.value())
                + "."
            )
        var mean = List[Float64]()
        var scale = List[Float64]()
        for k in range(t):
            if not self.normalize_y:
                mean.append(0.0)
                scale.append(1.0)
                continue
            var m: Float64 = 0.0
            for i in range(n):
                m += Y.unsafe_get(i * t + k)
            m /= Float64(n)
            var v: Float64 = 0.0
            for i in range(n):
                var d = Y.unsafe_get(i * t + k) - m
                v += d * d
            var s = sqrt(v / Float64(n))
            if s < 10.0 * 2.220446049250313e-16:  # _handle_zeros_in_scale
                s = 1.0
            mean.append(m)
            scale.append(s)
        if self.normalize_y:
            for i in range(n):
                for k in range(t):
                    Y.unsafe_set(i * t + k, (Y.unsafe_get(i * t + k) - mean[k]) / scale[k])
        var noise = _diagonal_noise(self.alpha, n)
        var warnings = List[String]()
        var lml_value: Float64
        if self.optimizer and kernel.n_dims() > 0:
            var objective = LogMarginalLikelihood(kernel.copy(), Xd, Y, noise.copy())
            ref optimizer = self.optimizer.value()
            var bounds = kernel.bounds()
            var results = List[OptimizationResult]()
            results.append(optimizer.minimize(objective, kernel.theta(), bounds))
            if self.n_restarts_optimizer > 0:
                check_restart_bounds(bounds)
                for _ in range(self.n_restarts_optimizer):
                    var start = restart_initial_theta(bounds, random_state)
                    results.append(optimizer.minimize(objective, start, bounds))
            for i in range(len(results)):
                if not results[i].converged:
                    warnings.append("ConvergenceWarning: " + results[i].message)
            var best = select_best(results^)
            kernel.set_theta(best.theta)
            for w in _bounds_warnings(kernel):
                warnings.append(w)
            lml_value = -best.value
        else:
            lml_value = log_marginal_likelihood_of(kernel, Xd, Y, noise, False).value
        var K = kernel(Xd)
        for i in range(n):
            K.unsafe_set(i * n + i, K.unsafe_get(i * n + i) + noise[i])
        var algebra = CholeskyAlgebra()
        var L: Dense64
        try:
            L = algebra.cholesky(K)
        except e:
            raise Error(
                "LinAlgError: The kernel, "
                + String(kernel)
                + ", is not returning a positive definite matrix. Try gradually"
                " increasing the 'alpha' parameter of your GaussianProcessRegressor"
                " estimator. ("
                + String(e)
                + ")"
            )
        var alpha = algebra.cho_solve(L, Y)
        for w in warnings:
            _warn(w)
        # Commit learned state only after every step succeeded.
        self.kernel_ = kernel^
        # fit borrows X immutably, so copy_X_train=False cannot alias the
        # caller's buffer; the snapshot is always independent (docs/api.md).
        self.X_train_ = X.copy()
        self._y_is_vector = y.ndim == 1
        self._target_matrix = Y.copy()
        self.y_train_ = _squeeze_columns(Y) if self._y_is_vector else Y.copy()
        self.L_ = L^
        self.alpha_ = _squeeze_columns(alpha) if self._y_is_vector else alpha^
        self.log_marginal_likelihood_value_ = lml_value
        self.n_features_in_ = Xd.shape[1]
        self.convergence_warnings_ = warnings^
        self._y_train_mean = mean^
        self._y_train_std = scale^
        self._fitted = True
        return self

    def _alpha_matrix(self) raises -> Dense64:
        var t = self._target_matrix.shape[1]
        var n = self._target_matrix.shape[0]
        var out = zeros[DType.float64](Shape(n, t))
        for i in range(n * t):
            out.unsafe_set(i, self.alpha_.unsafe_get(i))
        return out^

    # --- prediction -------------------------------------------------------------

    def _checked_input(self, X: Self.Input) raises -> Dense64:
        if X.ndim != 2:
            raise Error(
                "ValueError: Expected 2D array, got "
                + String(X.ndim)
                + "D array instead. Reshape your data."
            )
        var Xd = as_dense64(X, "GaussianProcessRegressor.predict X")
        if Xd.shape[0] < 1:
            raise Error("ValueError: Found array with 0 sample(s) while a minimum of 1 is required by GaussianProcessRegressor.")
        _check_finite(Xd, "X")
        if self._fitted and Xd.shape[1] != self.n_features_in_:
            raise Error(
                "ValueError: X has "
                + String(Xd.shape[1])
                + " features, but GaussianProcessRegressor is expecting "
                + String(self.n_features_in_)
                + " features as input."
            )
        return Xd^

    def predict(self, X: Self.Input) raises -> Dense64:
        """Posterior (or prior) mean: `(n,)` for one target, `(n, t)` otherwise."""
        return self.predict(X, return_std=False).mean.copy()

    def predict(
        self, X: Self.Input, *, return_std: Bool, return_cov: Bool = False
    ) raises -> PredictionResult:
        """sklearn `predict(X, return_std, return_cov)` as a tagged result."""
        if return_std and return_cov:
            raise Error("RuntimeError: At most one of return_std or return_cov can be requested.")
        validate_uncertainty(return_std, return_cov)
        var Xd = self._checked_input(X)
        var m = Xd.shape[0]
        if not self._fitted:
            var kernel = self.kernel.value().copy() if self.kernel else _prior_kernel()
            var t = self.n_targets.value() if self.n_targets else 1
            var mean = zeros[DType.float64](Shape(m)) if t == 1 else zeros[DType.float64](Shape(m, t))
            if return_cov:
                var cov = kernel(Xd)
                if t == 1:
                    return PredictionResult(mean=mean^, std=None, covariance=cov^)
                var rep = zeros[DType.float64](Shape(m, m, t))
                for e in range(m * m):
                    for k in range(t):
                        rep.unsafe_set(e * t + k, cov.unsafe_get(e))
                return PredictionResult(mean=mean^, std=None, covariance=rep^)
            if return_std:
                var diag = kernel.diag(Xd)
                var std = zeros[DType.float64](Shape(m)) if t == 1 else zeros[DType.float64](Shape(m, t))
                for i in range(m):
                    for k in range(t):
                        std.unsafe_set(i * t + k, sqrt(diag.unsafe_get(i)))
                return PredictionResult(mean=mean^, std=std^, covariance=None)
            return PredictionResult(mean=mean^, std=None, covariance=None)
        var n = self.L_.shape[0]
        var t = self._target_matrix.shape[1]
        var X_train = as_dense64(self.X_train_, "X_train_")
        var K_trans = self.kernel_(Xd, X_train^)  # (m, n)
        var A = self._alpha_matrix()
        var mean = zeros[DType.float64](Shape(m, t))
        for i in range(m):
            for k in range(t):
                var acc: Float64 = 0.0
                for j in range(n):
                    acc += K_trans.unsafe_get(i * n + j) * A.unsafe_get(j * t + k)
                mean.unsafe_set(i * t + k, self._y_train_std[k] * acc + self._y_train_mean[k])
        var mean_out = _squeeze_columns(mean)
        if not return_std and not return_cov:
            return PredictionResult(mean=mean_out^, std=None, covariance=None)
        var K_trans_T = zeros[DType.float64](Shape(n, m))
        for i in range(m):
            for j in range(n):
                K_trans_T.unsafe_set(j * m + i, K_trans.unsafe_get(i * n + j))
        var V = CholeskyAlgebra().solve_triangular(self.L_, K_trans_T)  # (n, m)
        if return_cov:
            var cov = self.kernel_(Xd)
            for a in range(m):
                for b in range(m):
                    var acc: Float64 = 0.0
                    for j in range(n):
                        acc += V.unsafe_get(j * m + a) * V.unsafe_get(j * m + b)
                    cov.unsafe_set(a * m + b, cov.unsafe_get(a * m + b) - acc)
            if t == 1:
                var s2 = self._y_train_std[0] * self._y_train_std[0]
                for e in range(m * m):
                    cov.unsafe_set(e, cov.unsafe_get(e) * s2)
                return PredictionResult(mean=mean_out^, std=None, covariance=cov^)
            var full = zeros[DType.float64](Shape(m, m, t))
            for e in range(m * m):
                for k in range(t):
                    full.unsafe_set(e * t + k, cov.unsafe_get(e) * self._y_train_std[k] * self._y_train_std[k])
            return PredictionResult(mean=mean_out^, std=None, covariance=full^)
        var var_ = self.kernel_.diag(Xd)
        var negative = False
        for i in range(m):
            var acc: Float64 = 0.0
            for j in range(n):
                var v = V.unsafe_get(j * m + i)
                acc += v * v
            var value = var_.unsafe_get(i) - acc
            if value < 0:
                negative = True
                value = 0.0
            var_.unsafe_set(i, value)
        if negative:
            _warn("Predicted variances smaller than 0. Setting those variances to 0.")
        var std = zeros[DType.float64](Shape(m)) if t == 1 else zeros[DType.float64](Shape(m, t))
        for i in range(m):
            for k in range(t):
                std.unsafe_set(i * t + k, sqrt(var_.unsafe_get(i) * self._y_train_std[k] * self._y_train_std[k]))
        return PredictionResult(mean=mean_out^, std=std^, covariance=None)

    # --- sampling and likelihood ---------------------------------------------------

    def sample_y(self, X: Self.Input, n_samples: Int = 1, random_state: Int = 0) raises -> Dense64:
        """Draws `(n, n_samples)` or `(n, t, n_samples)`; integer seed as sklearn."""
        var rng = RandomState(random_state)
        return self.sample_y(X, n_samples, rng)

    def sample_y(self, X: Self.Input, n_samples: Int, random_state: Optional[Int]) raises -> Dense64:
        """An explicit None advances the shared default RandomState."""
        with check_random_state(random_state) as rng:
            return self.sample_y(X, n_samples, rng[])

    def sample_y(self, X: Self.Input, n_samples: Int, mut random_state: RandomState) raises -> Dense64:
        """Draws from the posterior (or prior) using a borrowed RandomState."""
        if n_samples < 0:
            raise Error("ValueError: n_samples must be nonnegative")
        var prediction = self.predict(X, return_std=False, return_cov=True)
        ref mean = prediction.mean
        ref cov = prediction.covariance.value()
        var m = mean.shape[0]
        if mean.ndim == 1:
            var draws = random_state.multivariate_normal(mean, cov, n_samples)  # (s, m)
            var out = zeros[DType.float64](Shape(m, n_samples))
            for s in range(n_samples):
                for i in range(m):
                    out.unsafe_set(i * n_samples + s, draws.unsafe_get(s * m + i))
            return out^
        var t = mean.shape[1]
        var out = zeros[DType.float64](Shape(m, t, n_samples))
        for k in range(t):
            var mk = zeros[DType.float64](Shape(m))
            var ck = zeros[DType.float64](Shape(m, m))
            for i in range(m):
                mk.unsafe_set(i, mean.unsafe_get(i * t + k))
            for e in range(m * m):
                ck.unsafe_set(e, cov.unsafe_get(e * t + k))
            var draws = random_state.multivariate_normal(mk, ck, n_samples)
            for s in range(n_samples):
                for i in range(m):
                    out.unsafe_set((i * t + k) * n_samples + s, draws.unsafe_get(s * m + i))
        return out^

    def log_marginal_likelihood(
        mut self,
        theta: Optional[Dense64] = None,
        eval_gradient: Bool = False,
        clone_kernel: Bool = True,
    ) raises -> LikelihoodResult:
        """LML of the training data at log theta (fitted value when theta=None).

        `clone_kernel=False` updates `kernel_.theta` as in scikit-learn.
        """
        if not theta:
            if eval_gradient:
                raise Error("ValueError: Gradient can only be evaluated for theta!=None")
            if not self._fitted:
                raise Error("NotFittedError: call fit before log_marginal_likelihood")
            return LikelihoodResult(value=self.log_marginal_likelihood_value_, gradient=None)
        if not self._fitted:
            raise Error("NotFittedError: call fit before log_marginal_likelihood")
        if not clone_kernel:
            self.kernel_.set_theta(theta.value())
        var kernel = self.kernel_.clone_with_theta(theta.value())
        var X_train = as_dense64(self.X_train_, "X_train_")
        var noise = _diagonal_noise(self.alpha, X_train.shape[0])
        return log_marginal_likelihood_of(kernel, X_train, self._target_matrix, noise, eval_gradient)

    # --- scoring -------------------------------------------------------------------

    def score(
        self, X: Self.Input, y: Self.Target, sample_weight: Optional[List[Float64]] = None
    ) raises -> Float64:
        """R^2 with uniform averaging over targets (sklearn RegressorMixin.score)."""
        var predicted = self.predict(X)
        var Y = _target_matrix(y)
        var n = Y.shape[0]
        var t = Y.shape[1]
        var p_cols = predicted.shape[1] if predicted.ndim == 2 else 1
        if predicted.shape[0] != n or p_cols != t:
            raise Error("ValueError: y shape does not match predictions")
        var weights = List[Float64]()
        if sample_weight:
            weights = sample_weight.value().copy()
            if len(weights) != n:
                raise Error("ValueError: sample_weight length does not match samples")
        else:
            for _ in range(n):
                weights.append(1.0)
        if n < 2:
            _warn("R^2 score is not well-defined with less than two samples.")
            return Float64("nan")
        var mass: Float64 = 0.0
        for w in weights:
            mass += w
        var total: Float64 = 0.0
        for k in range(t):
            var mean: Float64 = 0.0
            for i in range(n):
                mean += weights[i] * Y.unsafe_get(i * t + k)
            mean /= mass
            var numerator: Float64 = 0.0
            var denominator: Float64 = 0.0
            for i in range(n):
                var value = Y.unsafe_get(i * t + k)
                var r = value - predicted.unsafe_get(i * t + k)
                numerator += weights[i] * r * r
                denominator += weights[i] * (value - mean) * (value - mean)
            if denominator != 0.0:
                total += 1.0 - numerator / denominator
            elif numerator == 0.0:
                total += 1.0
        return total / Float64(t)


def _isclose(a: Float64, b: Float64) -> Bool:
    """numpy.isclose defaults (rtol=1e-5, atol=1e-8)."""
    return abs(a - b) <= 1e-8 + 1e-5 * abs(b)


def _bounds_warnings(kernel: Kernel) raises -> List[String]:
    """sklearn `Kernel._check_bounds_params` messages (ConvergenceWarning)."""
    var out = List[String]()
    var bounds = kernel.bounds()
    var theta = kernel.theta()
    var idx = 0
    for spec in kernel.hyperparameters():
        if spec.fixed:
            continue
        for dim in range(spec.n_elements):
            var t = theta.unsafe_get(idx)
            if _isclose(bounds.unsafe_get(2 * idx), t):
                out.append(
                    "ConvergenceWarning: The optimal value found for dimension "
                    + String(dim) + " of parameter " + spec.name
                    + " is close to the specified lower bound "
                    + String(spec.bounds.unsafe_get(2 * dim))
                    + ". Decreasing the bound and calling fit again may find a better value."
                )
            elif _isclose(bounds.unsafe_get(2 * idx + 1), t):
                out.append(
                    "ConvergenceWarning: The optimal value found for dimension "
                    + String(dim) + " of parameter " + spec.name
                    + " is close to the specified upper bound "
                    + String(spec.bounds.unsafe_get(2 * dim + 1))
                    + ". Increasing the bound and calling fit again may find a better value."
                )
            idx += 1
    return out^
