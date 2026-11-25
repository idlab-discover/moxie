# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense Gaussian process classification with independent Laplace posteriors."""

from numojo import NDArray, Shape, zeros
from std.math import abs
from std.sys import stderr

from moxie._labels import encode_column
from moxie._random import check_random_state
from moxie.metrics._classification import accuracy_score

from ._base import GaussianProcessClassifierAPI
from ._gpr import LikelihoodResult, _bounds_warnings, _check_finite, _prior_kernel
from ._kernel_math import as_dense64, fexp
from ._laplace import (
    _LaplaceMode,
    _LaplaceObjective,
    _latent_mean,
    _latent_variance,
    _positive_probability,
)
from ._optimization import LBFGSBOptimizer, check_restart_bounds, restart_initial_theta, select_best
from ._plumbing import Dense64, LatentMoments, OptimizationResult, validate_likelihood
from .kernels import CompoundKernel, Kernel


@fieldwise_init
struct GaussianProcessClassifierParams(Copyable, Movable):
    """Constructor configuration for a Gaussian process classifier."""

    var kernel: Optional[Kernel]
    var optimizer: Optional[String]
    var n_restarts_optimizer: Int
    var max_iter_predict: Int
    var warm_start: Bool
    var copy_X_train: Bool
    var random_state: Optional[Int]
    var multi_class: String
    var n_jobs: Optional[Int]

    def validate(self) raises:
        if self.optimizer and self.optimizer.value() != "fmin_l_bfgs_b":
            raise Error("ValueError: optimizer must be fmin_l_bfgs_b or None")
        if self.n_restarts_optimizer < 0:
            raise Error("ValueError: n_restarts_optimizer must be nonnegative")
        if self.max_iter_predict < 1:
            raise Error("ValueError: max_iter_predict must be at least 1")
        if self.random_state:
            var seed = self.random_state.value()
            if seed < 0 or seed > 4294967295:
                raise Error("ValueError: random_state must be None or an int in [0, 2**32 - 1]")
        if self.multi_class != "one_vs_rest" and self.multi_class != "one_vs_one":
            raise Error("ValueError: multi_class must be one_vs_rest or one_vs_one")


@fieldwise_init
struct _BinaryPosterior(Copyable, Movable):
    var objective: _LaplaceObjective
    var mode: _LaplaceMode
    var value: Float64
    var warnings: List[String]

    def kernel_(self) -> Kernel:
        return self.objective.kernel.copy()

    def posterior_mean(self, var X: Dense64) raises -> Dense64:
        var cross = self.objective.kernel(self.objective.X, X^)
        return _latent_mean(cross, self.mode.residual)

    def latent_mean_and_variance(self, var X: Dense64) raises -> LatentMoments:
        var diagonal = self.objective.kernel.diag(X)
        var cross = self.objective.kernel(self.objective.X, X^)
        var mean = _latent_mean(cross, self.mode.residual)
        var variance = _latent_variance(
            diagonal^, cross^, self.mode.root_weights, self.mode.factor
        )
        return LatentMoments(mean^, variance^)

    def likelihood(
        mut self, theta: Dense64, eval_gradient: Bool, clone_kernel: Bool
    ) raises -> LikelihoodResult:
        var trial = self.objective.kernel.clone_with_theta(theta)
        if not clone_kernel:
            self.objective.kernel = trial.copy()
        return self.objective.likelihood(trial, eval_gradient)


def _fit_binary(
    var X: Dense64, var y: Dense64, var kernel: Kernel, params: GaussianProcessClassifierParams
) raises -> _BinaryPosterior:
    var objective = _LaplaceObjective(
        kernel^, X^, y^, params.max_iter_predict, params.warm_start
    )
    var warnings = List[String]()
    var value: Float64
    if params.optimizer and objective.kernel.n_dims() > 0:
        var optimizer = LBFGSBOptimizer[_LaplaceObjective]()
        var bounds = objective.kernel.bounds()
        if params.n_restarts_optimizer > 0:
            check_restart_bounds(bounds)
        var results = List[OptimizationResult]()
        var theta = objective.kernel.theta()
        results.append(optimizer.minimize(objective, theta, bounds))
        if params.n_restarts_optimizer > 0:
            var rng = check_random_state(params.random_state)
            for _ in range(params.n_restarts_optimizer):
                var start = restart_initial_theta(bounds, rng[])
                # The pinned classifier exponentiates the sampled log bounds
                # before passing the start to its log-coordinate optimizer.
                for i in range(start.size):
                    start.unsafe_set(i, fexp(start.unsafe_get(i)))
                results.append(optimizer.minimize(objective, start, bounds))
        for i in range(len(results)):
            ref result = results[i]
            if not result.converged:
                warnings.append("ConvergenceWarning: " + result.message)
        var best = select_best(results^)
        objective.kernel.set_theta(best.theta)
        value = -best.value
        for warning in _bounds_warnings(objective.kernel):
            warnings.append(warning)
    else:
        var initial_kernel = objective.kernel.copy()
        value = objective.likelihood(initial_kernel, False).value
    var covariance = objective.kernel(objective.X)
    var mode = objective.mode(covariance)
    return _BinaryPosterior(objective^, mode^, value, warnings^)


def _fit_pair(
    X: Dense64,
    first: List[Int],
    second: List[Int],
    kernel: Kernel,
    params: GaussianProcessClassifierParams,
) raises -> _BinaryPosterior:
    # Merge the two sorted row lists to preserve the original training order.
    var size = len(first) + len(second)
    var rows = List[Int](capacity=size)
    var target = zeros[DType.float64](Shape(size, 1))
    var i = 0
    var j = 0
    while i < len(first) or j < len(second):
        if j == len(second) or (i < len(first) and first[i] < second[j]):
            rows.append(first[i])
            i += 1
        else:
            target.unsafe_set(len(rows), 1.0)
            rows.append(second[j])
            j += 1
    return _fit_binary(X[rows], target^, kernel.copy(), params)


@fieldwise_init
struct _ClassifierPosterior(Copyable, Movable):
    """Snapshot of the fitted binary components in classification order."""

    var estimators_: List[_BinaryPosterior]
    var one_vs_one: Bool
    var value: Float64
    var warnings: List[String]

    def kernel_(self) raises -> Kernel:
        if len(self.estimators_) == 1:
            return self.estimators_[0].kernel_()
        var kernels = List[Kernel](capacity=len(self.estimators_))
        for estimator in self.estimators_:
            kernels.append(estimator.kernel_())
        return CompoundKernel(kernels^)

    def likelihood(
        mut self, theta: Dense64, eval_gradient: Bool, clone_kernel: Bool
    ) raises -> LikelihoodResult:
        var count = len(self.estimators_)
        if count == 1:
            return self.estimators_[0].likelihood(theta, eval_gradient, clone_kernel)
        if eval_gradient:
            raise Error("NotImplementedError: likelihood gradients require binary classification")
        var width = self.estimators_[0].objective.kernel.n_dims()
        if theta.ndim != 1 or (theta.size != width and theta.size != count * width):
            raise Error("ValueError: theta must contain log parameters for one or all components")
        var trials = List[Kernel](capacity=count if not clone_kernel else 0)
        var total: Float64 = 0.0
        for i in range(count):
            ref objective = self.estimators_[i].objective
            var coordinates = (
                theta.copy()
                if theta.size == width
                else theta[Slice(i * width, (i + 1) * width)]
            )
            var trial = objective.kernel.clone_with_theta(coordinates)
            total += objective.likelihood(trial, False).value
            if not clone_kernel:
                trials.append(trial^)
        for i in range(len(trials)):
            self.estimators_[i].objective.kernel = trials[i].copy()
        return LikelihoodResult(total / Float64(count), None)

    def rest_probabilities(self, mut X: Dense64) raises -> Dense64:
        var rows = X.shape[0]
        var count = len(self.estimators_)
        var probabilities = zeros[DType.float64](Shape(rows, count))
        for c in range(count):
            var latent = self.estimators_[c].latent_mean_and_variance(X.view())
            for i in range(rows):
                probabilities.unsafe_set(
                    i * count + c,
                    _positive_probability(
                        latent.mean.unsafe_get(i), latent.variance.unsafe_get(i)
                    ),
                )
        return probabilities^

    def pair_scores(self, mut X: Dense64, classes: Int) raises -> Dense64:
        var rows = X.shape[0]
        var votes = zeros[DType.float64](Shape(rows, classes))
        var confidence = zeros[DType.float64](Shape(rows, classes))
        var k = 0
        for first in range(classes):
            for second in range(first + 1, classes):
                var latent = self.estimators_[k].latent_mean_and_variance(X.view())
                for i in range(rows):
                    var mean = latent.mean.unsafe_get(i)
                    var p = _positive_probability(mean, latent.variance.unsafe_get(i))
                    var left = i * classes + first
                    var right = i * classes + second
                    var winner = right if mean > 0.0 else left
                    votes.unsafe_set(winner, votes.unsafe_get(winner) + 1.0)
                    confidence.unsafe_set(left, confidence.unsafe_get(left) - p)
                    confidence.unsafe_set(right, confidence.unsafe_get(right) + p)
                k += 1
        for i in range(votes.size):
            # This bounded transform breaks ties without overturning a vote.
            var total = confidence.unsafe_get(i)
            votes.unsafe_set(
                i, votes.unsafe_get(i) + total / (3.0 * (abs(total) + 1.0))
            )
        return votes^


struct GaussianProcessClassifier[
    dtype: DType = DType.float64, label: DType = DType.int64
](Copyable, GaussianProcessClassifierAPI):
    """Dense binary, one-vs-rest and one-vs-one Gaussian process classification."""

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label]
    comptime Prediction = Self.Target
    comptime Probabilities = Dense64
    comptime SampleWeight = Dense64
    comptime Params = GaussianProcessClassifierParams

    var _params: Self.Params
    var _posterior: Optional[_ClassifierPosterior]
    var _classes: Self.Target
    var n_features_in_: Int

    def __init__(
        out self,
        kernel: Optional[Kernel] = None,
        *,
        optimizer: Optional[String] = "fmin_l_bfgs_b",
        n_restarts_optimizer: Int = 0,
        max_iter_predict: Int = 100,
        warm_start: Bool = False,
        copy_X_train: Bool = True,
        random_state: Optional[Int] = None,
        multi_class: String = "one_vs_rest",
        n_jobs: Optional[Int] = None,
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, (
            "GaussianProcessClassifier input must be Float32 or Float64"
        )
        comptime assert Self.label in [DType.int64, DType.float32, DType.float64], (
            "GaussianProcessClassifier targets must be Int64, Float32 or Float64"
        )
        self._params = Self.Params(
            kernel.copy(), optimizer.copy(), n_restarts_optimizer, max_iter_predict,
            warm_start, copy_X_train, random_state, multi_class, n_jobs,
        )
        self._params.validate()
        self._posterior = None
        self._classes = zeros[Self.label](Shape(0))
        self.n_features_in_ = 0

    def __init__(
        out self,
        kernel: Kernel,
        *,
        optimizer: Optional[String] = "fmin_l_bfgs_b",
        n_restarts_optimizer: Int = 0,
        max_iter_predict: Int = 100,
        warm_start: Bool = False,
        copy_X_train: Bool = True,
        random_state: Optional[Int] = None,
        multi_class: String = "one_vs_rest",
        n_jobs: Optional[Int] = None,
    ) raises:
        self = Self(
            Optional[Kernel](kernel.copy()),
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            warm_start=warm_start,
            copy_X_train=copy_X_train,
            random_state=random_state,
            multi_class=multi_class,
            n_jobs=n_jobs,
        )

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        params.validate()
        self._params = params.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        kernel: Optional[Optional[Kernel]] = None,
        optimizer: Optional[Optional[String]] = None,
        n_restarts_optimizer: Optional[Int] = None,
        max_iter_predict: Optional[Int] = None,
        warm_start: Optional[Bool] = None,
        copy_X_train: Optional[Bool] = None,
        random_state: Optional[Optional[Int]] = None,
        multi_class: Optional[String] = None,
        n_jobs: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        var params = self.get_params()
        if kernel:
            params.kernel = kernel.value().copy()
        if optimizer:
            params.optimizer = optimizer.value().copy()
        params.n_restarts_optimizer = n_restarts_optimizer.or_else(params.n_restarts_optimizer)
        params.max_iter_predict = max_iter_predict.or_else(params.max_iter_predict)
        params.warm_start = warm_start.or_else(params.warm_start)
        params.copy_X_train = copy_X_train.or_else(params.copy_X_train)
        params.random_state = random_state.or_else(params.random_state)
        if multi_class:
            params.multi_class = multi_class.value().copy()
        params.n_jobs = n_jobs.or_else(params.n_jobs)
        return self.set_params(params)

    def __sklearn_clone__(self) raises -> Self:
        var model = Self()
        _ = model.set_params(self._params)
        return model^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._posterior)

    def _require_fitted(self) raises:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: GaussianProcessClassifier is not fitted")

    def _input(self, X: Self.Input) raises -> Dense64:
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise Error("ValueError: X must be a nonempty 2-D feature array")
        var dense = as_dense64(X, "GaussianProcessClassifier X")
        _check_finite(dense, "X")
        return dense^

    def _query(self, X: Self.Input) raises -> Dense64:
        self._require_fitted()
        var query = self._input(X)
        if query.shape[1] != self.n_features_in_:
            raise Error("ValueError: X feature count does not match fit")
        return query^

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target
    ) raises -> ref[self] Self:
        var dense = self._input(X)
        if y.ndim != 1 and (y.ndim != 2 or y.shape[1] != 1):
            raise Error("ValueError: y must be rank one or a single column")
        if y.size != dense.shape[0]:
            raise Error("ValueError: X and y have different numbers of samples")
        var encoding = encode_column(y)
        var classes = encoding.classes.size
        if classes < 2:
            raise Error("ValueError: GaussianProcessClassifier requires at least two classes")
        if classes > 2 and self._params.n_jobs.or_else(1) == 0:
            raise Error("ValueError: n_jobs cannot be zero for multiclass classification")
        var kernel = _prior_kernel()
        if self._params.kernel:
            kernel = self._params.kernel.value().copy()
        if kernel.name() == "CompoundKernel":
            raise Error("ValueError: kernel cannot be a CompoundKernel")

        var features = dense.shape[1]
        ref codes = encoding.index.inverse
        var one_vs_one = classes > 2 and self._params.multi_class == "one_vs_one"
        var estimators = List[_BinaryPosterior]()
        if one_vs_one:
            var class_rows = List[List[Int]](length=classes, fill=List[Int]())
            for i in range(codes.size):
                class_rows[Int(codes.unsafe_get(i))].append(i)
            for first in range(classes):
                for second in range(first + 1, classes):
                    estimators.append(
                        _fit_pair(
                            dense, class_rows[first], class_rows[second], kernel, self._params
                        )
                    )
        else:
            # The owned feature snapshot is shared internally by OvR components.
            for c in range(1 if classes == 2 else classes):
                var positive = Int64(1 if classes == 2 else c)
                var target = zeros[DType.float64](Shape(y.size, 1))
                for i in range(y.size):
                    target.unsafe_set(i, Float64(codes.unsafe_get(i) == positive))
                estimators.append(
                    _fit_binary(dense.view(), target^, kernel.copy(), self._params)
                )

        var value: Float64 = 0.0
        var warnings = List[String]()
        for estimator in estimators:
            value += estimator.value
            for warning in estimator.warnings:
                warnings.append(warning.copy())
        for warning in warnings:
            print(warning, file=stderr)
        value /= Float64(len(estimators))
        self._posterior = _ClassifierPosterior(estimators^, one_vs_one, value, warnings^)
        self._classes = encoding.classes.copy()
        self.n_features_in_ = features
        return self

    def classes_(self) raises -> Self.Target:
        self._require_fitted()
        return self._classes.copy()

    def n_classes_(self) raises -> Int:
        self._require_fitted()
        return self._classes.size

    def base_estimator_(self) raises -> _ClassifierPosterior:
        """An independent snapshot with the fitted components in estimators_."""
        self._require_fitted()
        return self._posterior.value().copy()

    def kernel_(self) raises -> Kernel:
        self._require_fitted()
        return self._posterior.value().kernel_()

    def log_marginal_likelihood_value_(self) raises -> Float64:
        self._require_fitted()
        return self._posterior.value().value

    def log_marginal_likelihood(
        mut self, theta: Optional[Dense64] = None,
        eval_gradient: Bool = False, clone_kernel: Bool = True,
    ) raises -> LikelihoodResult:
        """Return evidence at fit or at explicit log theta, averaged for multiclass."""
        self._require_fitted()
        ref posterior = self._posterior.value()
        validate_likelihood(Bool(theta), eval_gradient)
        if theta:
            return posterior.likelihood(theta.value(), eval_gradient, clone_kernel)
        return LikelihoodResult(posterior.value, None)

    def latent_mean_and_variance(self, X: Self.Input) raises -> LatentMoments:
        """Return Float64 latent mean and variance vectors for a binary fit."""
        var query = self._query(X)
        if self._classes.size != 2:
            raise Error("ValueError: latent mean and variance require binary classification")
        return self._posterior.value().estimators_[0].latent_mean_and_variance(query^)

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var query = self._query(X)
        ref posterior = self._posterior.value()
        var classes = self._classes.size
        var rows = query.shape[0]
        var prediction = zeros[Self.label](Shape(rows))
        if classes == 2:
            var mean = posterior.estimators_[0].posterior_mean(query^)
            for i in range(rows):
                var code = Int(mean.unsafe_get(i) > 0.0)
                prediction.unsafe_set(i, self._classes.unsafe_get(code))
            return prediction^

        var scores = (
            posterior.pair_scores(query, classes)
            if posterior.one_vs_one
            else posterior.rest_probabilities(query)
        )
        for i in range(rows):
            var best = 0
            for c in range(1, classes):
                if scores.unsafe_get(i * classes + c) > scores.unsafe_get(i * classes + best):
                    best = c
            prediction.unsafe_set(i, self._classes.unsafe_get(best))
        return prediction^

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        """Return Float64 probabilities in sorted class order for binary and OvR fits."""
        var query = self._query(X)
        ref posterior = self._posterior.value()
        if posterior.one_vs_one:
            raise Error(
                "ValueError: one_vs_one multi-class mode does not support predicting probability estimates"
            )
        var classes = self._classes.size
        if classes > 2:
            var probabilities = posterior.rest_probabilities(query)
            for i in range(query.shape[0]):
                var total: Float64 = 0.0
                for c in range(classes):
                    total += probabilities.unsafe_get(i * classes + c)
                if total != 0.0:
                    for c in range(classes):
                        var position = i * classes + c
                        probabilities.unsafe_set(position, probabilities.unsafe_get(position) / total)
            return probabilities^

        var latent = posterior.estimators_[0].latent_mean_and_variance(query^)
        var probabilities = zeros[DType.float64](Shape(latent.mean.size, 2))
        for i in range(latent.mean.size):
            var p = _positive_probability(latent.mean.unsafe_get(i), latent.variance.unsafe_get(i))
            probabilities.unsafe_set(2 * i, 1.0 - p)
            probabilities.unsafe_set(2 * i + 1, p)
        return probabilities^

    def score(
        self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None
    ) raises -> Float64:
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
