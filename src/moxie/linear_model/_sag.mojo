# Adapted from scikit-learn sklearn/linear_model/_sag.py and _sag_fast.pyx.tp.
# Authors: The scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Shared dense SAG and proximal SAGA updates."""

from std.math import abs, copysign, isfinite, max
from numojo import NDArray, Shape, zeros
from moxie._random import our_rand_r


@fieldwise_init
struct SAGSolution(Movable):
    var coefficients: NDArray[DType.float64]
    var iterations: Int


def dense_sag[
    dtype: DType,
    Gradient: def(Int, NDArray[DType.float64], mut List[Float64]) -> None,
](
    X: NDArray[dtype], sample_weights: NDArray[DType.float64],
    var coefficients: NDArray[DType.float64], outputs: Int, fit_intercept: Bool,
    alpha: Float64, beta: Float64, max_iter: Int, tol: Float64, var seed: UInt32,
    saga: Bool, max_squared_norm: Float64, curvature: Float64, row_gradient: Gradient,
) raises -> SAGSolution:
    """Apply each sampled row's unweighted loss gradient to compact dense X."""
    var rows = X.shape[0]
    var features = X.shape[1]
    if rows > Int(Int32.MAX) or features > Int(Int32.MAX):
        raise Error("ValueError: SAG requires at most 2**31 - 1 samples and features")
    var feature_parameters = features * outputs
    var alpha_scaled = alpha / Float64(rows)
    var beta_scaled = beta / Float64(rows)
    var lipschitz = curvature * (max_squared_norm + Float64(Int(fit_intercept))) + alpha_scaled
    var denominator = lipschitz
    if saga:
        denominator = 2 * lipschitz + min(2 * alpha, lipschitz)
    if denominator == 0 and max_iter == 0:
        return SAGSolution(coefficients^, 1)
    if not (denominator > 0 and isfinite(denominator)):
        raise Error("ValueError: SAG requires a finite positive step denominator")
    var step = 1 / denominator
    var shrink = 1 - step * alpha_scaled
    var threshold = step * beta_scaled
    if shrink == 0:
        raise Error("ZeroDivisionError: SAG cannot handle step_size * alpha_scaled == 1")
    if shrink < 0 or not isfinite(step) or not isfinite(threshold):
        raise Error("ValueError: SAG has an invalid step size")
    if max_iter == 0:
        # The pinned solver reports one epoch even when the loop is skipped.
        return SAGSolution(coefficients^, 1)
    if rows > Int.MAX // 8 // outputs:
        raise Error("ValueError: SAG gradient memory size overflows")

    var seen = List[Bool](length=rows, fill=False)
    var seen_count = 0
    var memory = zeros[DType.float64](Shape(rows * outputs))
    var gradient_sum = zeros[DType.float64](Shape(coefficients.size))
    var previous = zeros[DType.float64](Shape(feature_parameters))
    var scores = List[Float64](length=outputs, fill=0)
    var delta = List[Float64](length=outputs, fill=0)
    for epoch in range(max_iter):
        for _ in range(rows):
            var sample = Int(our_rand_r(seed)) % rows
            if not seen[sample]:
                seen[sample] = True
                seen_count += 1
            var row = sample * features
            row_gradient(sample, coefficients, scores)

            var sample_weight = sample_weights.unsafe_get(sample)
            for k in range(outputs):
                var index = sample * outputs + k
                var gradient = scores[k] * sample_weight
                delta[k] = gradient - memory.unsafe_get(index)
                memory.unsafe_set(index, gradient)

            var average_step = step / Float64(seen_count)
            var correction_step = step - average_step if saga else 0
            # Dense rows touch every feature, so pending gradient and proximal
            # steps can be applied eagerly without sklearn's sparse bookkeeping.
            for j in range(features):
                var value = Float64(X.unsafe_get(row + j))
                for k in range(outputs):
                    var index = j * outputs + k
                    var correction = value * delta[k]
                    var total = gradient_sum.unsafe_get(index) + correction
                    gradient_sum.unsafe_set(index, total)
                    var updated = shrink * coefficients.unsafe_get(index) - (
                        average_step * total + correction_step * correction
                    )
                    if threshold > 0:
                        var magnitude = abs(updated) - threshold
                        updated = 0 if magnitude <= 0 else copysign(magnitude, updated)
                    coefficients.unsafe_set(index, updated)
            if fit_intercept:
                for k in range(outputs):
                    var index = feature_parameters + k
                    var total = gradient_sum.unsafe_get(index) + delta[k]
                    gradient_sum.unsafe_set(index, total)
                    coefficients.unsafe_set(
                        index, coefficients.unsafe_get(index) - (
                            average_step * total + correction_step * delta[k]
                        ),
                    )

        var max_change = Float64(0)
        var max_coefficient = Float64(0)
        for j in range(coefficients.size):
            var coefficient = coefficients.unsafe_get(j)
            if not isfinite(coefficient):
                raise Error("ValueError: SAG encountered nonfinite coefficients; scale the features")
            if j < feature_parameters:
                max_coefficient = max(max_coefficient, abs(coefficient))
                max_change = max(max_change, abs(coefficient - previous.unsafe_get(j)))
                previous.unsafe_set(j, coefficient)
        if max_change == 0 or (max_coefficient > 0 and max_change / max_coefficient <= tol):
            return SAGSolution(coefficients^, epoch + 1)
    return SAGSolution(coefficients^, max_iter)
