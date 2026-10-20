# Adapted from scikit-learn sklearn/linear_model/_bayes.py.
# Authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Dense Bayesian ridge regression with empirical-Bayes precision updates."""
from numojo import NDArray, Shape, zeros
from std.math import abs, isfinite, log, sqrt
from std.utils import Variant
from moxie.base import WeightedSupervisedEstimator
from moxie.utils.validation import check_is_fitted
from moxie.linear_model._base import LinearRegressor
from moxie.linear_model._linear_base import (
    preprocess_data, scalar_weights, set_intercept, linear_decision,
    public_coef, canonical_coef, public_prediction, check_predict_input, check_finite, r2_score,
)
from moxie._linear_linalg import svd


@fieldwise_init
struct BayesianRidgeParams(Copyable, Movable):
    """Constructor configuration, independent of learned posterior state."""
    var max_iter: Int
    var tol: Float64
    var alpha_1: Float64
    var alpha_2: Float64
    var lambda_1: Float64
    var lambda_2: Float64
    var alpha_init: Optional[Float64]
    var lambda_init: Optional[Float64]
    var compute_score: Bool
    var fit_intercept: Bool
    var copy_X: Bool
    var verbose: Bool


struct BayesianRidge[dtype: DType = DType.float64](Copyable, LinearRegressor, WeightedSupervisedEstimator):
    """Fit a Gaussian linear model and estimate noise and weight precisions.

    Inputs and public arrays use Float32 or Float64. Numerical intermediates
    accumulate in Float64. Prediction uncertainty includes observation noise.
    """
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Prediction = NDArray[Self.dtype]
    comptime SampleWeight = NDArray[Self.dtype]
    comptime Params = BayesianRidgeParams
    comptime UncertainPrediction = Tuple[Self.Prediction, Self.Prediction]
    comptime PredictionResult = Variant[Self.Prediction, Self.UncertainPrediction]
    var max_iter: Int
    var tol: Float64
    var alpha_1: Float64
    var alpha_2: Float64
    var lambda_1: Float64
    var lambda_2: Float64
    var alpha_init: Optional[Float64]
    var lambda_init: Optional[Float64]
    var compute_score: Bool
    var fit_intercept: Bool
    var copy_X: Bool
    var verbose: Bool
    var coef_: NDArray[Self.dtype]
    var intercept_: NDArray[Self.dtype]
    var sigma_: NDArray[Self.dtype]
    var X_offset_: NDArray[Self.dtype]
    var X_scale_: NDArray[Self.dtype]
    var scores_: NDArray[Self.dtype]
    var alpha_: Float64
    var lambda_: Float64
    var n_iter_: Int
    var n_features_in_: Int
    var _fitted: Bool

    def __init__(out self, *,
        max_iter: Int = 300,
        tol: Float64 = 1e-3,
        alpha_1: Float64 = 1e-6,
        alpha_2: Float64 = 1e-6,
        lambda_1: Float64 = 1e-6,
        lambda_2: Float64 = 1e-6,
        alpha_init: Optional[Float64] = None,
        lambda_init: Optional[Float64] = None,
        compute_score: Bool = False,
        fit_intercept: Bool = True,
        copy_X: Bool = True,
        verbose: Bool = False,
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.verbose = verbose
        self.coef_ = zeros[Self.dtype](Shape(0))
        self.intercept_ = zeros[Self.dtype](Shape(0))
        self.sigma_ = zeros[Self.dtype](Shape(0))
        self.X_offset_ = zeros[Self.dtype](Shape(0))
        self.X_scale_ = zeros[Self.dtype](Shape(0))
        self.scores_ = zeros[Self.dtype](Shape(0))
        self.alpha_ = 0
        self.lambda_ = 0
        self.n_iter_ = 0
        self.n_features_in_ = 0
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.max_iter, self.tol, self.alpha_1, self.alpha_2, self.lambda_1, self.lambda_2, self.alpha_init, self.lambda_init, self.compute_score, self.fit_intercept, self.copy_X, self.verbose)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.alpha_1 = params.alpha_1
        self.alpha_2 = params.alpha_2
        self.lambda_1 = params.lambda_1
        self.lambda_2 = params.lambda_2
        self.alpha_init = params.alpha_init
        self.lambda_init = params.lambda_init
        self.compute_score = params.compute_score
        self.fit_intercept = params.fit_intercept
        self.copy_X = params.copy_X
        self.verbose = params.verbose
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *,
        max_iter: Optional[Int] = None,
        tol: Optional[Float64] = None,
        alpha_1: Optional[Float64] = None,
        alpha_2: Optional[Float64] = None,
        lambda_1: Optional[Float64] = None,
        lambda_2: Optional[Float64] = None,
        alpha_init: Optional[Optional[Float64]] = None,
        lambda_init: Optional[Optional[Float64]] = None,
        compute_score: Optional[Bool] = None,
        fit_intercept: Optional[Bool] = None,
        copy_X: Optional[Bool] = None,
        verbose: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        if max_iter:
            self.max_iter = max_iter.value()
        if tol:
            self.tol = tol.value()
        if alpha_1:
            self.alpha_1 = alpha_1.value()
        if alpha_2:
            self.alpha_2 = alpha_2.value()
        if lambda_1:
            self.lambda_1 = lambda_1.value()
        if lambda_2:
            self.lambda_2 = lambda_2.value()
        if alpha_init:
            self.alpha_init = alpha_init.value()
        if lambda_init:
            self.lambda_init = lambda_init.value()
        if compute_score:
            self.compute_score = compute_score.value()
        if fit_intercept:
            self.fit_intercept = fit_intercept.value()
        if copy_X:
            self.copy_X = copy_X.value()
        if verbose:
            self.verbose = verbose.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(max_iter=self.max_iter, tol=self.tol, alpha_1=self.alpha_1, alpha_2=self.alpha_2, lambda_1=self.lambda_1, lambda_2=self.lambda_2, alpha_init=self.alpha_init, lambda_init=self.lambda_init, compute_score=self.compute_score, fit_intercept=self.fit_intercept, copy_X=self.copy_X, verbose=self.verbose)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def _validate_params(self) raises:
        if self.max_iter < 1 or not isfinite(self.tol) or self.tol <= 0:
            raise Error("ValueError: BayesianRidge requires max_iter >= 1 and finite tol > 0")
        for value in [self.alpha_1, self.alpha_2, self.lambda_1, self.lambda_2]:
            if not isfinite(value) or value < 0:
                raise Error("ValueError: BayesianRidge hyperpriors must be finite and nonnegative")
        for value in [self.alpha_init, self.lambda_init]:
            if value:
                if not isfinite(value.value()) or value.value() < 0:
                    raise Error("ValueError: BayesianRidge initial precisions must be finite and nonnegative")

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        self._fit(X, y, Optional(sample_weight.copy()))
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        self._fit(X, y, Optional(scalar_weights(X, sample_weight)))
        return self

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight]) raises:
        self._validate_params()
        var data = preprocess_data(X, y, sample_weight, self.fit_intercept, True, "BayesianRidge")
        if data.y.shape[1] != 1:
            raise Error("ValueError: BayesianRidge requires a single target")
        var n = data.X.shape[0]
        var p = data.X.shape[1]
        var mass = Float64(0)
        var mean = Float64(0)
        for i in range(n):
            var w = Float64(sample_weight.value().unsafe_get(i * Int(sample_weight.value().strides[0]))) if sample_weight else Float64(1)
            mass += w
            mean += w * Float64(y.unsafe_get(i * Int(y.strides[0])))
        mean /= mass
        var variance = Float64(0)
        for i in range(n):
            var w = Float64(sample_weight.value().unsafe_get(i * Int(sample_weight.value().strides[0]))) if sample_weight else Float64(1)
            var delta = Float64(y.unsafe_get(i * Int(y.strides[0]))) - mean
            variance += w * delta * delta
        variance /= mass
        var alpha = self.alpha_init.value() if self.alpha_init else 1 / (variance + 2.220446049250313e-16)
        var penalty = self.lambda_init.value() if self.lambda_init else Float64(1)
        if self.compute_score and (alpha == 0 or penalty == 0):
            raise Error("ValueError: BayesianRidge evidence requires positive initial precisions")

        # Zero rows preserve X.T @ X and expose the full right singular basis.
        # The economy SVD on a wide matrix would omit null-space uncertainty.
        var design = zeros[DType.float64](Shape(max(n, p), p))
        for i in range(n):
            for j in range(p):
                var xij = Float64(data.X.unsafe_get(i * p + j))
                design.unsafe_set(i * p + j, xij)
        var basis = zeros[DType.float64](Shape(p, p))
        var eigenvalues = zeros[DType.float64](Shape(p))
        var projected = zeros[DType.float64](Shape(p))
        try:
            var factor = svd(design)
            basis = factor.vt()
            var singular = factor.singular_values()
            var left = factor.u()
            for j in range(min(n, p)):
                eigenvalues.unsafe_set(j, singular.unsafe_get(j) ** 2)
                var value = Float64(0)
                for i in range(n):
                    value += left.unsafe_get(i * p + j) * Float64(data.y.unsafe_get(i))
                projected.unsafe_set(j, singular.unsafe_get(j) * value)
        except error:
            raise Error(String("LinAlgError: BayesianRidge SVD failed (", error.kind, " in ", error.operation, ")"))
        var coef = zeros[DType.float64](Shape(p))
        var previous = zeros[DType.float64](Shape(p))
        var scores = List[Float64]()
        var iterations = 0
        for iteration in range(self.max_iter):
            var sse = _posterior_mean(data.X, data.y, basis, eigenvalues, projected, alpha, penalty, coef)
            var norm = _squared_norm(coef)
            if self.compute_score:
                scores.append(self._log_marginal_likelihood(eigenvalues, mass, alpha, penalty, norm, sse))
            var gamma = Float64(0)
            for j in range(min(n, p)):
                var value = alpha * eigenvalues.unsafe_get(j)
                if penalty + value == 0:
                    raise Error("LinAlgError: BayesianRidge has singular initial posterior precision")
                gamma += value / (penalty + value)
            penalty = (gamma + 2 * self.lambda_1) / (norm + 2 * self.lambda_2)
            alpha = (mass - gamma + 2 * self.alpha_1) / (sse + 2 * self.alpha_2)
            if not isfinite(alpha) or not isfinite(penalty) or alpha <= 0 or penalty <= 0:
                raise Error("LinAlgError: BayesianRidge precision update is nonpositive or nonfinite")
            iterations = iteration + 1
            var change = Float64(0)
            for j in range(p):
                change += abs(coef.unsafe_get(j) - previous.unsafe_get(j))
            if iteration > 0 and change < self.tol:
                if self.verbose:
                    print("Convergence after ", iteration, " iterations")
                break
            previous = coef.copy()
        # The reference final evidence uses the last loop coefficient norm,
        # together with the recomputed residual at the final precisions.
        var last_norm = _squared_norm(coef)
        var sse = _posterior_mean(data.X, data.y, basis, eigenvalues, projected, alpha, penalty, coef)
        if self.compute_score:
            scores.append(self._log_marginal_likelihood(eigenvalues, mass, alpha, penalty, last_norm, sse))
        var sigma = zeros[Self.dtype](Shape(p, p))
        for j in range(p):
            for k in range(j + 1):
                var value = Float64(0)
                for l in range(p):
                    value += basis.unsafe_get(l * p + j) * basis.unsafe_get(l * p + k) / (alpha * eigenvalues.unsafe_get(l) + penalty)
                sigma.unsafe_set(j * p + k, Scalar[Self.dtype](value))
                sigma.unsafe_set(k * p + j, Scalar[Self.dtype](value))
        var canonical = zeros[Self.dtype](Shape(1, p))
        for j in range(p):
            canonical.unsafe_set(j, Scalar[Self.dtype](coef.unsafe_get(j)))
        var intercept = set_intercept(canonical, data.X_offset, data.y_offset, data.X_scale, self.fit_intercept)
        var score_array = zeros[Self.dtype](Shape(len(scores)))
        for i in range(len(scores)):
            score_array.unsafe_set(i, Scalar[Self.dtype](scores[i]))
        check_finite(canonical, "coef_", "BayesianRidge")
        check_finite(intercept, "intercept_", "BayesianRidge")
        check_finite(sigma, "sigma_", "BayesianRidge")
        check_finite(score_array, "scores_", "BayesianRidge")
        self.coef_ = public_coef(canonical, True)
        self.intercept_ = intercept^
        self.sigma_ = sigma^
        self.X_offset_ = data.X_offset.copy()
        self.X_scale_ = data.X_scale.copy()
        self.scores_ = score_array^
        self.alpha_ = alpha
        self.lambda_ = penalty
        self.n_iter_ = iterations
        self.n_features_in_ = p
        self._fitted = True

    def _log_marginal_likelihood(self, eigenvalues: NDArray[DType.float64], mass: Float64, alpha: Float64, penalty: Float64, norm: Float64, sse: Float64) raises -> Float64:
        var logdet = Float64(0)
        for j in range(eigenvalues.size):
            logdet -= log(penalty + alpha * eigenvalues.unsafe_get(j))
        var score = self.lambda_1 * log(penalty) - self.lambda_2 * penalty
        score += self.alpha_1 * log(alpha) - self.alpha_2 * alpha
        return score + 0.5 * (Float64(eigenvalues.size) * log(penalty) + mass * log(alpha) - alpha * sse - penalty * norm + logdet - mass * log(Float64(6.283185307179586)))

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        check_is_fitted(self)
        check_predict_input(X, self.n_features_in_, "BayesianRidge")
        return public_prediction(linear_decision(X, canonical_coef(self.coef_), self.intercept_), True)

    def predict(self, X: Self.Input, *, return_std: Bool) raises -> Self.PredictionResult:
        """Runtime flag returns a Variant containing an array or (mean, std)."""
        if return_std:
            return Self.PredictionResult(self.predict_std(X))
        return Self.PredictionResult(self.predict(X))

    def predict_std(self, X: Self.Input) raises -> Self.UncertainPrediction:
        """Typed convenience returning (mean, standard deviation) directly."""
        var mean = self.predict(X)
        var deviation = zeros[Self.dtype](Shape(X.shape[0]))
        var p = self.n_features_in_
        for i in range(X.shape[0]):
            var variance = 1 / self.alpha_
            for j in range(p):
                var xj = Float64(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))) - Float64(self.X_offset_.unsafe_get(j))
                for k in range(p):
                    var xk = Float64(X.unsafe_get(i * Int(X.strides[0]) + k * Int(X.strides[1]))) - Float64(self.X_offset_.unsafe_get(k))
                    variance += xj * Float64(self.sigma_.unsafe_get(j * p + k)) * xk
            if not isfinite(variance) or variance < 0:
                raise Error("LinAlgError: BayesianRidge predictive variance is negative or nonfinite")
            deviation.unsafe_set(i, Scalar[Self.dtype](sqrt(variance)))
        return (mean^, deviation^)

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return r2_score(y, self.predict(X), sample_weight)


def _squared_norm(a: NDArray[DType.float64]) -> Float64:
    var result = Float64(0)
    for i in range(a.size):
        result += a.unsafe_get(i) ** 2
    return result


def _posterior_mean[dtype: DType](X: NDArray[dtype], y: NDArray[dtype], basis: NDArray[DType.float64], eigenvalues: NDArray[DType.float64], projected: NDArray[DType.float64], alpha: Float64, penalty: Float64, mut coef: NDArray[DType.float64]) raises -> Float64:
    var p = coef.size
    for j in range(p):
        var value = Float64(0)
        if alpha != 0:
            for k in range(min(X.shape[0], p)):
                var denominator = eigenvalues.unsafe_get(k) + penalty / alpha
                if denominator == 0:
                    raise Error("LinAlgError: BayesianRidge has singular initial posterior precision")
                value += basis.unsafe_get(k * p + j) * projected.unsafe_get(k) / denominator
        coef.unsafe_set(j, value)
    var sse = Float64(0)
    for i in range(X.shape[0]):
        var residual = Float64(y.unsafe_get(i))
        for j in range(p):
            residual -= Float64(X.unsafe_get(i * p + j)) * coef.unsafe_get(j)
        sse += residual * residual
    return sse
