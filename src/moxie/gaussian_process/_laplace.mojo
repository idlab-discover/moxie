# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Binary Laplace posterior and likelihood (GPML Algorithms 3.1, 3.2, 5.1)."""

from numojo import Shape, zeros
from numax.core.libm import erf
from std.collections import InlineArray
from std.math import abs, inf, nan, sqrt

from ._algebra import CholeskyAlgebra
from ._gpr import LikelihoodResult, _covariance_copy
from ._kernel_math import fexp, flog
from ._plumbing import Dense64, KernelContract, LikelihoodObjective, ObjectiveEvaluation
from .kernels import Kernel


@fieldwise_init
struct _LaplaceMode(Copyable, Movable):
    var residual: Dense64
    var root_weights: Dense64
    var factor: Dense64
    var coefficients: Dense64
    var value: Float64


def _scale_rows(mut matrix: Dense64, scale: Dense64) raises:
    var cols = matrix.shape[1]
    for i in range(matrix.shape[0]):
        var weight = scale.unsafe_get(i)
        for j in range(cols):
            var index = i * cols + j
            matrix.unsafe_set(index, matrix.unsafe_get(index) * weight)


def _posterior_mode(
    covariance: Dense64, y: Dense64, max_iter: Int, mut latent: Dense64
) raises -> _LaplaceMode:
    var n = y.size
    var algebra = CholeskyAlgebra()
    var residual = zeros[DType.float64](Shape(n, 1))
    var root_weights = zeros[DType.float64](Shape(n))
    var system = zeros[DType.float64](Shape(n, n))
    var rhs = zeros[DType.float64](Shape(n, 1))
    var factor = zeros[DType.float64](Shape(0, 0))
    var coefficients = zeros[DType.float64](Shape(n, 1))
    var likelihood = -inf[DType.float64]()
    for _ in range(max_iter):
        for i in range(n):
            var f = latent.unsafe_get(i)
            var p = 1.0 / (1.0 + fexp(-f))
            var weight = p * (1.0 - p)
            var error = y.unsafe_get(i) - p
            residual.unsafe_set(i, error)
            root_weights.unsafe_set(i, sqrt(weight))
            rhs.unsafe_set(i, weight * f + error)
        for i in range(n):
            for j in range(n):
                var entry = covariance.unsafe_get(i * n + j)
                system.unsafe_set(
                    i * n + j,
                    root_weights.unsafe_get(i)
                    * entry
                    * root_weights.unsafe_get(j),
                )
            system.unsafe_set(i * n + i, system.unsafe_get(i * n + i) + 1.0)
        factor = algebra.cholesky(system)
        var weighted = covariance @ rhs
        _scale_rows(weighted, root_weights)
        var solved = algebra.cho_solve(factor, weighted)
        for i in range(n):
            coefficients.unsafe_set(
                i,
                rhs.unsafe_get(i)
                - root_weights.unsafe_get(i) * solved.unsafe_get(i),
            )
        latent = covariance @ coefficients

        var penalty: Float64 = 0.0
        var loss: Float64 = 0.0
        for i in range(n):
            var f = latent.unsafe_get(i)
            var margin = (1.0 - 2.0 * y.unsafe_get(i)) * f
            penalty += coefficients.unsafe_get(i) * f
            var z = fexp(-abs(margin))
            var sum = 1.0 + z
            # Correct the rounding of 1 + z, including when it rounds to one.
            var log_one_plus_z = flog(sum) + (z - (sum - 1.0)) / sum
            loss += max(0.0, margin) + log_one_plus_z
        var next_value = (
            -0.5 * penalty - loss - 0.5 * algebra.log_determinant(factor)
        )
        # sklearn returns the previous evidence on the stopping step; the
        # returned factors and cached latent vector come from the latest step.
        if next_value - likelihood < 1e-10:
            break
        likelihood = next_value
    return _LaplaceMode(
        residual^, root_weights^, factor^, coefficients^, likelihood
    )


def _gradient(
    covariance: Dense64, derivatives: Dense64, y: Dense64, mode: _LaplaceMode
) raises -> Dense64:
    var algebra = CholeskyAlgebra()
    var n = y.size
    var p = derivatives.shape[2]
    var result = zeros[DType.float64](Shape(p))
    if p == 0:
        return result^

    # R = W^(1/2) B^-1 W^(1/2). C = L^-1 W^(1/2) K.
    var diagonal = zeros[DType.float64](Shape(n, n))
    for i in range(n):
        diagonal.unsafe_set(i * n + i, mode.root_weights.unsafe_get(i))
    var R = algebra.cho_solve(mode.factor, diagonal)
    _scale_rows(R, mode.root_weights)
    var weighted = covariance.copy()
    _scale_rows(weighted, mode.root_weights)
    var C = algebra.solve_triangular(mode.factor, weighted)
    var s2 = zeros[DType.float64](Shape(n, 1))
    for i in range(n):
        var norm: Float64 = 0.0
        for j in range(n):
            var c = C.unsafe_get(j * n + i)
            norm += c * c
        var probability = y.unsafe_get(i) - mode.residual.unsafe_get(i)
        var root_weight = mode.root_weights.unsafe_get(i)
        s2.unsafe_set(
            i,
            -0.5
            * (covariance.unsafe_get(i * n + i) - norm)
            * root_weight
            * root_weight
            * (1.0 - 2.0 * probability),
        )

    # Applying the adjoint once gives s2.T (I - K R) for every derivative.
    var correction = R @ (covariance @ s2)
    for i in range(n):
        for j in range(n):
            var weight = 0.5 * (
                mode.coefficients.unsafe_get(i)
                * mode.coefficients.unsafe_get(j)
                - R.unsafe_get(j * n + i)
            ) + (
                s2.unsafe_get(i) - correction.unsafe_get(i)
            ) * mode.residual.unsafe_get(
                j
            )
            var offset = (i * n + j) * p
            for k in range(p):
                result.unsafe_set(
                    k,
                    result.unsafe_get(k)
                    + weight * derivatives.unsafe_get(offset + k),
                )
    return result^


struct _LaplaceObjective[K: KernelContract](Copyable, LikelihoodObjective):
    var kernel: Self.K
    var X: Self.K.Input
    var y: Dense64
    var max_iter: Int
    var warm_start: Bool
    var latent: Dense64

    def __init__(
        out self,
        var kernel: Self.K,
        var X: Self.K.Input,
        var y: Dense64,
        max_iter: Int,
        warm_start: Bool,
    ) raises:
        self.latent = zeros[DType.float64](Shape(y.size, 1))
        self.kernel = kernel^
        self.X = X^
        self.y = y^
        self.max_iter = max_iter
        self.warm_start = warm_start

    def mode(mut self, covariance: Dense64) raises -> _LaplaceMode:
        if not self.warm_start:
            for i in range(self.latent.size):
                self.latent.unsafe_set(i, 0.0)
        return _posterior_mode(covariance, self.y, self.max_iter, self.latent)

    def likelihood(
        mut self, kernel: Self.K, eval_gradient: Bool
    ) raises -> LikelihoodResult:
        var evaluation = kernel.evaluate(self.X, eval_gradient=eval_gradient)
        var n = self.y.size
        var covariance: Dense64
        comptime if Self.K == Kernel:
            covariance = evaluation.covariance.view()
        else:
            covariance = _covariance_copy(evaluation.covariance, n, n)
        var mode = self.mode(covariance)
        if not eval_gradient:
            return LikelihoodResult(mode.value, None)
        if not evaluation.gradient:
            raise Error("ValueError: the kernel did not return a requested gradient")
        ref derivatives = evaluation.gradient.value()
        if derivatives.shape != Shape(n, n, len(kernel.theta())):
            raise Error("ValueError: expected kernel gradient shape (n, n, n_dims)")
        var dense = derivatives.view() if derivatives.is_c_contiguous() else derivatives.contiguous()
        var gradient = _gradient(covariance, dense, self.y, mode)
        return LikelihoodResult(mode.value, gradient^)

    def evaluate(
        mut self, theta: Dense64, eval_gradient: Bool = True
    ) raises -> ObjectiveEvaluation:
        # Keep the fitted/initial kernel separate from trial kernels.
        var trial = self.kernel.clone_with_theta(theta)
        var result = self.likelihood(trial, eval_gradient)
        if not eval_gradient:
            return ObjectiveEvaluation(value=-result.value, gradient=None)
        var gradient = result.gradient.value().copy()
        for i in range(gradient.size):
            gradient.unsafe_set(i, -gradient.unsafe_get(i))
        return ObjectiveEvaluation(value=-result.value, gradient=gradient^)


def _latent_mean(cross: Dense64, residual: Dense64) raises -> Dense64:
    var rows = cross.shape[0]
    var cols = cross.shape[1]
    var mean = zeros[DType.float64](Shape(cols))
    for i in range(rows):
        var weight = residual.unsafe_get(i)
        for j in range(cols):
            mean.unsafe_set(
                j, mean.unsafe_get(j) + cross.unsafe_get(i * cols + j) * weight
            )
    return mean^


def _latent_variance(
    var variance: Dense64,
    var cross: Dense64,
    root_weights: Dense64,
    factor: Dense64,
) raises -> Dense64:
    _scale_rows(cross, root_weights)
    var projected = CholeskyAlgebra().solve_triangular(factor, cross)
    var cols = projected.shape[1]
    for j in range(cols):
        var norm: Float64 = 0.0
        for i in range(projected.shape[0]):
            var entry = projected.unsafe_get(i * cols + j)
            norm += entry * entry
        variance.unsafe_set(j, variance.unsafe_get(j) - norm)
    return variance^


def _positive_probability(mean: Float64, variance: Float64) -> Float64:
    if variance < 0.0:
        return nan[DType.float64]()
    var scales: InlineArray[Float64, 5] = [0.41, 0.4, 0.37, 0.44, 0.39]
    var weights: InlineArray[Float64, 5] = [
        -1854.8214151,
        3516.89893646,
        221.29346712,
        128.12323805,
        -2010.49422654,
    ]
    var integral: Float64 = 0.0
    var offset: Float64 = 0.0
    for i in range(5):
        var scale = scales[i]
        # Algebraic form of the Gaussian-erf integral, valid at variance=0.
        var argument = (
            scale * mean / sqrt(1.0 + 2.0 * variance * scale * scale)
        )
        integral += weights[i] * (0.5 * erf(argument))
        offset += weights[i]
    return integral + 0.5 * offset
