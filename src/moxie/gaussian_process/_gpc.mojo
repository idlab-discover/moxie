# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Gaussian process classification with independent Laplace posteriors."""

from numojo import NDArray, Shape, zeros
from std.math import abs
from std.sys import stderr

from moxie._labels import encode_column
from moxie._random import check_random_state
from moxie.metrics._classification import accuracy_score

from ._base import GaussianProcessClassifierAPI
from ._gpr import LikelihoodResult, _bounds_warnings, _check_finite, _covariance_copy, _prior_kernel
from ._kernel_math import as_dense64, fexp
from ._laplace import (
    _LaplaceMode,
    _LaplaceObjective,
    _latent_mean,
    _latent_variance,
    _positive_probability,
)
from ._optimization import LBFGSBOptimizer, check_restart_bounds, restart_initial_theta, select_best
from ._plumbing import Dense64, KernelContract, LatentMoments, OptimizationResult, validate_likelihood
from .kernels import CompoundKernel, Kernel


comptime GaussianProcessClassifierParams = _ClassifierParams[Kernel]


@fieldwise_init
struct _ClassifierParams[K: KernelContract](Copyable, Movable):
    """Constructor configuration for a Gaussian process classifier."""

    var kernel: Optional[Self.K]
    var optimizer: Optional[String]
    var n_restarts_optimizer: Int
    var max_iter_predict: Int
    var warm_start: Bool
    var copy_X_train: Bool
    var random_state: Optional[Int]
    var multi_class: String
    var n_jobs: Optional[Int]

    def validate(self) raises:
        comptime if Self.K != Kernel:
            if not self.kernel or self.kernel.value().requires_vector_input():
                raise Error("ValueError: an explicit non-vector kernel is required")
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


def _sample_count[K: KernelContract](X: K.Input) raises -> Int:
    comptime if K == Kernel:
        return rebind[Dense64](X).shape[0]
    else:
        return len(X)


def _component_input[K: KernelContract](mut X: K.Input) raises -> K.Input:
    comptime if K == Kernel:
        ref matrix = rebind[Dense64](X)
        return rebind_var[K.Input](matrix.view())
    else:
        return X.copy()


@fieldwise_init
struct _BinaryPosterior[K: KernelContract](Copyable, Movable):
    var objective: _LaplaceObjective[Self.K]
    var mode: _LaplaceMode
    var value: Float64
    var warnings: List[String]

    def kernel_(self) -> Self.K:
        return self.objective.kernel.copy()

    def _cross_covariance(self, var X: Self.K.Input, rows: Int) raises -> Dense64:
        comptime if Self.K == Kernel:
            return rebind[Kernel](self.objective.kernel)(
                rebind[Dense64](self.objective.X), rebind_var[Dense64](X^)
            )
        else:
            var evaluation = self.objective.kernel.evaluate(
                self.objective.X, Optional[Self.K.Input](X^)
            )
            return _covariance_copy(evaluation.covariance, self.objective.y.size, rows)

    def posterior_mean(self, var X: Self.K.Input) raises -> Dense64:
        var rows = _sample_count[Self.K](X)
        var cross = self._cross_covariance(X^, rows)
        return _latent_mean(cross, self.mode.residual)

    def latent_mean_and_variance(self, var X: Self.K.Input) raises -> LatentMoments:
        var rows = _sample_count[Self.K](X)
        var diagonal = self.objective.kernel.diag(X)
        comptime if Self.K != Kernel:
            if diagonal.shape != Shape(rows):
                raise Error("ValueError: expected kernel diagonal shape (n,)")
            diagonal = diagonal.contiguous()
        var cross = self._cross_covariance(X^, rows)
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


def _fit_binary[K: KernelContract](
    var X: K.Input, var y: Dense64, var kernel: K, params: _ClassifierParams[K]
) raises -> _BinaryPosterior[K]:
    var objective = _LaplaceObjective[K](
        kernel^, X^, y^, params.max_iter_predict, params.warm_start
    )
    var warnings = List[String]()
    var value: Float64
    if params.optimizer and len(objective.kernel.theta()) > 0:
        var optimizer = LBFGSBOptimizer[_LaplaceObjective[K]]()
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
    var covariance: Dense64
    comptime if K == Kernel:
        covariance = rebind[Kernel](objective.kernel)(rebind[Dense64](objective.X))
    else:
        var evaluation = objective.kernel.evaluate(objective.X)
        var n = objective.y.size
        covariance = _covariance_copy(evaluation.covariance, n, n)
    var mode = objective.mode(covariance)
    return _BinaryPosterior[K](objective^, mode^, value, warnings^)


@fieldwise_init
struct _PairTarget(Movable):
    var rows: List[Int]
    var target: Dense64

    def take_target(deinit self) -> Dense64:
        return self.target^


def _pair_target(first: List[Int], second: List[Int]) raises -> _PairTarget:
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
    return _PairTarget(rows^, target^)


@fieldwise_init
struct _ClassifierPosterior[K: KernelContract](Copyable, Movable):
    """Snapshot of the fitted binary components in classification order."""

    var estimators_: List[_BinaryPosterior[Self.K]]
    var one_vs_one: Bool
    var value: Float64
    var warnings: List[String]

    comptime FittedKernel = Kernel if Self.K == Kernel else List[Self.K]

    def kernel_(self) raises -> Self.FittedKernel:
        comptime if Self.K == Kernel:
            if len(self.estimators_) == 1:
                return rebind_var[Self.FittedKernel](self.estimators_[0].kernel_())
        var kernels = List[Self.K](capacity=len(self.estimators_))
        for estimator in self.estimators_:
            kernels.append(estimator.kernel_())
        comptime if Self.K == Kernel:
            return rebind_var[Self.FittedKernel](CompoundKernel(rebind_var[List[Kernel]](kernels^)))
        else:
            return rebind_var[Self.FittedKernel](kernels^)

    def likelihood(
        mut self, theta: Dense64, eval_gradient: Bool, clone_kernel: Bool
    ) raises -> LikelihoodResult:
        var count = len(self.estimators_)
        if count == 1:
            return self.estimators_[0].likelihood(theta, eval_gradient, clone_kernel)
        if eval_gradient:
            raise Error("NotImplementedError: likelihood gradients require binary classification")
        var width = len(self.estimators_[0].objective.kernel.theta())
        if theta.ndim != 1 or (theta.size != width and theta.size != count * width):
            raise Error("ValueError: theta must contain log parameters for one or all components")
        var trials = List[Self.K](capacity=count if not clone_kernel else 0)
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

    def rest_probabilities(self, mut X: Self.K.Input) raises -> Dense64:
        var rows = _sample_count[Self.K](X)
        var count = len(self.estimators_)
        var probabilities = zeros[DType.float64](Shape(rows, count))
        for c in range(count):
            var latent = self.estimators_[c].latent_mean_and_variance(_component_input[Self.K](X))
            for i in range(rows):
                probabilities.unsafe_set(
                    i * count + c,
                    _positive_probability(
                        latent.mean.unsafe_get(i), latent.variance.unsafe_get(i)
                    ),
                )
        return probabilities^

    def pair_scores(self, mut X: Self.K.Input, classes: Int) raises -> Dense64:
        var rows = _sample_count[Self.K](X)
        var votes = zeros[DType.float64](Shape(rows, classes))
        var confidence = zeros[DType.float64](Shape(rows, classes))
        var k = 0
        for first in range(classes):
            for second in range(first + 1, classes):
                var latent = self.estimators_[k].latent_mean_and_variance(_component_input[Self.K](X))
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


comptime StructuredGaussianProcessClassifier[
    K: KernelContract, Sample: Copyable & Deinitable, label: DType = DType.int64
] = GaussianProcessClassifier[DType.float64, label, K, Sample]


struct GaussianProcessClassifier[
    dtype: DType = DType.float64,
    label: DType = DType.int64,
    K: KernelContract = Kernel,
    Sample: Copyable & Deinitable = String
](Copyable, GaussianProcessClassifierAPI):
    """Binary, one-vs-rest and one-vs-one Gaussian process classification."""

    comptime Input = NDArray[Self.dtype] if Self.K == Kernel else Self.K.Input
    comptime FittedKernel = _ClassifierPosterior[Self.K].FittedKernel
    comptime Target = NDArray[Self.label]
    comptime Prediction = Self.Target
    comptime Probabilities = Dense64
    comptime SampleWeight = Dense64
    comptime Params = _ClassifierParams[Self.K]

    var _params: Self.Params
    var _posterior: Optional[_ClassifierPosterior[Self.K]]
    var _classes: Self.Target
    var n_features_in_: Int

    def __init__(
        out self,
        kernel: Optional[Self.K] = None,
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
        comptime assert Self.K == Kernel or Self.K.Input == List[Self.Sample], (
            "StructuredGaussianProcessClassifier kernel input must be List[Sample]"
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
    ) raises where Self.K == Kernel:
        self = Self(
            Optional[Self.K](rebind[Self.K](kernel).copy()),
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            warm_start=warm_start,
            copy_X_train=copy_X_train,
            random_state=random_state,
            multi_class=multi_class,
            n_jobs=n_jobs,
        )

    def __init__(
        out self,
        kernel: Self.K,
        *,
        optimizer: Optional[String] = "fmin_l_bfgs_b",
        n_restarts_optimizer: Int = 0,
        max_iter_predict: Int = 100,
        warm_start: Bool = False,
        copy_X_train: Bool = True,
        random_state: Optional[Int] = None,
        multi_class: String = "one_vs_rest",
        n_jobs: Optional[Int] = None,
    ) raises where Self.K != Kernel:
        self = Self(
            Optional[Self.K](kernel.copy()),
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
        kernel: Optional[Optional[Self.K]] = None,
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
        var model = Self(
            self._params.kernel,
            optimizer=self._params.optimizer,
            n_restarts_optimizer=self._params.n_restarts_optimizer,
            max_iter_predict=self._params.max_iter_predict,
            warm_start=self._params.warm_start,
            copy_X_train=self._params.copy_X_train,
            random_state=self._params.random_state,
            multi_class=self._params.multi_class,
            n_jobs=self._params.n_jobs,
        )
        return model^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._posterior)

    def _require_fitted(self) raises:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: GaussianProcessClassifier is not fitted")

    @staticmethod
    def _input(X: Self.Input) raises -> Self.K.Input:
        comptime if Self.K == Kernel:
            ref matrix = rebind[NDArray[Self.dtype]](X)
            if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
                raise Error("ValueError: X must be a nonempty 2-D feature array")
            var dense = as_dense64(matrix, "GaussianProcessClassifier X")
            _check_finite(dense, "X")
            return rebind_var[Self.K.Input](dense^)
        else:
            if len(X) == 0:
                raise Error("ValueError: expected at least one structured sample")
            return rebind_var[Self.K.Input](X.copy())

    @staticmethod
    def _subset(X: Self.K.Input, rows: List[Int]) raises -> Self.K.Input:
        comptime if Self.K == Kernel:
            return rebind_var[Self.K.Input](rebind[Dense64](X)[rows])
        else:
            ref samples = rebind[List[Self.Sample]](X)
            var selected = List[Self.Sample](capacity=len(rows))
            for i in rows:
                selected.append(samples[i].copy())
            return rebind_var[Self.K.Input](selected^)

    def _query(self, X: Self.Input) raises -> Self.K.Input:
        self._require_fitted()
        var query = Self._input(X)
        comptime if Self.K == Kernel:
            if rebind[Dense64](query).shape[1] != self.n_features_in_:
                raise Error("ValueError: X feature count does not match fit")
        return query^

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target
    ) raises -> ref[self] Self:
        var samples = Self._input(X)
        if y.ndim != 1 and (y.ndim != 2 or y.shape[1] != 1):
            raise Error("ValueError: y must be rank one or a single column")
        if y.size != _sample_count[Self.K](samples):
            raise Error("ValueError: X and y have different numbers of samples")
        var encoding = encode_column(y)
        var classes = encoding.classes.size
        if classes < 2:
            raise Error("ValueError: GaussianProcessClassifier requires at least two classes")
        if classes > 2 and self._params.n_jobs.or_else(1) == 0:
            raise Error("ValueError: n_jobs cannot be zero for multiclass classification")
        var kernel: Self.K
        var features = 0
        comptime if Self.K == Kernel:
            kernel = self._params.kernel.value().copy() if self._params.kernel else rebind_var[Self.K](_prior_kernel())
            if rebind[Kernel](kernel).name() == "CompoundKernel":
                raise Error("ValueError: kernel cannot be a CompoundKernel")
            features = rebind[Dense64](samples).shape[1]
        else:
            kernel = self._params.kernel.value().copy()

        ref codes = encoding.index.inverse
        var one_vs_one = classes > 2 and self._params.multi_class == "one_vs_one"
        var estimators = List[_BinaryPosterior[Self.K]]()
        if one_vs_one:
            var class_rows = List[List[Int]](length=classes, fill=List[Int]())
            for i in range(codes.size):
                class_rows[Int(codes.unsafe_get(i))].append(i)
            for first in range(classes):
                for second in range(first + 1, classes):
                    var pair = _pair_target(class_rows[first], class_rows[second])
                    var pair_samples = Self._subset(samples, pair.rows)
                    estimators.append(
                        _fit_binary(pair_samples^, pair^.take_target(), kernel.copy(), self._params)
                    )
        else:
            for c in range(1 if classes == 2 else classes):
                var positive = Int64(1 if classes == 2 else c)
                var target = zeros[DType.float64](Shape(y.size, 1))
                for i in range(y.size):
                    target.unsafe_set(i, Float64(codes.unsafe_get(i) == positive))
                estimators.append(
                    _fit_binary(
                        _component_input[Self.K](samples), target^, kernel.copy(), self._params
                    )
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
        self._posterior = _ClassifierPosterior[Self.K](estimators^, one_vs_one, value, warnings^)
        self._classes = encoding.classes.copy()
        self.n_features_in_ = features
        return self

    def classes_(self) raises -> Self.Target:
        self._require_fitted()
        return self._classes.copy()

    def n_classes_(self) raises -> Int:
        self._require_fitted()
        return self._classes.size

    def base_estimator_(self) raises -> _ClassifierPosterior[Self.K]:
        """An independent snapshot with the fitted components in estimators_."""
        self._require_fitted()
        return self._posterior.value().copy()

    def kernel_(self) raises -> Self.FittedKernel:
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
        var rows = _sample_count[Self.K](query)
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
            for i in range(_sample_count[Self.K](query)):
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
