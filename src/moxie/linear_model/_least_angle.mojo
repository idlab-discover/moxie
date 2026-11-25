# Adapted from sklearn/linear_model/_least_angle.py, scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense Gram least-angle endpoints for fixed-dictionary coding."""

from std.math import abs, isfinite, round, sqrt

from mojo_linalg import triangular_solve_inplace
from numojo import NDArray, NDArrayStrides, Shape, zeros

from ._linear_base import _entry, check_finite


def _extend_factor[dtype: DType](
    gram: NDArray[dtype], order: List[Int], atom: Int, active: Int,
    mut factor: NDArray[dtype],
) raises -> Bool:
    var width = factor.shape[1]
    var pivot = _entry(gram, atom, atom)
    if active > 0:
        var lower = factor.view_with_layout(
            Shape(active, active), NDArrayStrides(width, 1), 0
        )
        var border = zeros[dtype](Shape(active))
        for j in range(active):
            border.unsafe_set(j, _entry(gram, atom, order[j]))
        triangular_solve_inplace(lower, border)
        var squared = Scalar[dtype](0)
        for j in range(active):
            var value = border.unsafe_get(j)
            factor.unsafe_set(active * width + j, value)
            squared += value * value
        pivot -= squared
    var diagonal = sqrt(abs(pivot))
    if diagonal < Scalar[dtype](1e-7):
        return False
    if not isfinite(diagonal):
        raise Error("ValueError: LARS Cholesky pivot must remain finite")
    factor.unsafe_set(active * width + active, diagonal)
    return True


def _least_angle_target[dtype: DType](
    gram: NDArray[dtype], covariance: NDArray[dtype], sample: Int,
    n_features: Int, alpha_min: Float64, max_iter: Int, lasso: Bool,
    positive: Bool, mut code: NDArray[dtype], mut factor: NDArray[dtype],
) raises:
    var atoms = gram.shape[0]
    var row = sample * atoms
    var order = List[Int](capacity=atoms)
    var signs = zeros[dtype](Shape(factor.shape[1]))
    var correlations = zeros[dtype](Shape(atoms))
    var slopes = zeros[dtype](Shape(atoms))
    var previous = zeros[dtype](Shape(atoms))
    for j in range(atoms):
        order.append(j)
        correlations.unsafe_set(j, _entry(covariance, j, sample))

    comptime tiny = Scalar[dtype](1.1754943508222875e-38)
    comptime equality_tol = Scalar[dtype](1.1920928955078125e-7)
    comptime rounding_scale = Scalar[dtype](1e6 if dtype == DType.float32 else 1e15)
    var target = Scalar[dtype](alpha_min)
    var previous_alpha = Scalar[dtype](0)
    var active = 0
    var iteration = 0
    var crossed = False
    while True:
        var candidate = active
        var peak = Scalar[dtype](0)
        for j in range(active, atoms):
            var correlation = correlations.unsafe_get(order[j])
            var magnitude = correlation if positive else abs(correlation)
            if j == active or magnitude > peak:
                candidate = j
                peak = magnitude
        var alpha = peak / Scalar[dtype](n_features)
        if alpha <= target + equality_tol:
            if iteration > 0 and abs(alpha - target) > equality_tol:
                var fraction = (previous_alpha - target) / (previous_alpha - alpha)
                for j in range(atoms):
                    var old = previous.unsafe_get(j)
                    code.unsafe_set(row + j, old + fraction * (code.unsafe_get(row + j) - old))
            break
        if iteration >= max_iter or active == atoms:
            break
        if not crossed:
            var atom = order[candidate]
            if not _extend_factor(gram, order, atom, active, factor):
                correlations.unsafe_set(atom, 0)
                continue
            order[candidate] = order[active]
            order[active] = atom
            signs.unsafe_set(active, Scalar[dtype](1) if correlations.unsafe_get(atom) > 0 else Scalar[dtype](-1))
            active += 1
        if lasso and iteration > 0 and alpha > previous_alpha:
            break

        var direction = zeros[dtype](Shape(active))
        for j in range(active):
            direction.unsafe_set(j, signs.unsafe_get(j))
        var lower = factor.view_with_layout(
            Shape(active, active), NDArrayStrides(factor.shape[1], 1), 0
        )
        triangular_solve_inplace(lower, direction)
        triangular_solve_inplace(lower, direction, transpose=True)
        var signed_sum = Scalar[dtype](0)
        for j in range(active):
            signed_sum += direction.unsafe_get(j) * signs.unsafe_get(j)
        var angular = 1 / sqrt(signed_sum)
        if angular <= 0 or not isfinite(angular):
            raise Error("ValueError: LARS equiangular direction must remain finite")
        for j in range(active):
            direction.unsafe_set(j, direction.unsafe_get(j) * angular)

        var step = peak / angular
        for j in range(active, atoms):
            var atom = order[j]
            var slope = Scalar[dtype](0)
            for k in range(active):
                slope += _entry(gram, atom, order[k]) * direction.unsafe_get(k)
            slope = round(slope * rounding_scale) / rounding_scale
            slopes.unsafe_set(atom, slope)
            var correlation = correlations.unsafe_get(atom)
            var forward = (peak - correlation) / (angular - slope + tiny)
            if forward > 0:
                step = min(step, forward)
            if not positive:
                var backward = (peak + correlation) / (angular + slope + tiny)
                if backward > 0:
                    step = min(step, backward)

        var crossing = Scalar[dtype].MAX
        for j in range(active):
            var event = -code.unsafe_get(row + order[j]) / (direction.unsafe_get(j) + tiny)
            slopes.unsafe_set(order[j], event)
            if event > 0:
                crossing = min(crossing, event)
        crossed = crossing < step
        if crossed and lasso:
            step = crossing

        for j in range(atoms):
            previous.unsafe_set(j, code.unsafe_get(row + j))
        for j in range(active):
            var atom = order[j]
            var value = previous.unsafe_get(atom) + step * direction.unsafe_get(j)
            if not isfinite(value):
                raise Error("ValueError: LARS coefficients must remain finite")
            code.unsafe_set(row + atom, value)
        for j in range(active, atoms):
            var atom = order[j]
            var updated = correlations.unsafe_get(atom) - step * slopes.unsafe_get(atom)
            if not isfinite(updated):
                raise Error("ValueError: LARS residual correlations must remain finite")
            correlations.unsafe_set(atom, updated)

        if crossed:
            var previous_active = active
            for j in range(active - 1, -1, -1):
                var atom = order[j]
                if slopes.unsafe_get(atom) != crossing:
                    continue
                if not lasso:
                    signs.unsafe_set(j, -signs.unsafe_get(j))
                    continue
                # Move the dropped atom to the inactive prefix. Its next
                # correlation is computed from the original Gram and target.
                code.unsafe_set(row + atom, 0)
                active -= 1
                for k in range(j, active):
                    order[k] = order[k + 1]
                    signs.unsafe_set(k, signs.unsafe_get(k + 1))
                order[active] = atom
            if lasso:
                for j in range(active):
                    if not _extend_factor(gram, order, order[j], j, factor):
                        raise Error("ValueError: LARS active Gram became singular after a drop")
                for j in range(active, previous_active):
                    var atom = order[j]
                    var residual = _entry(covariance, atom, sample)
                    for k in range(active):
                        residual -= _entry(gram, atom, order[k]) * code.unsafe_get(row + order[k])
                    correlations.unsafe_set(atom, residual)
        previous_alpha = alpha
        iteration += 1


def least_angle_gram_codes[dtype: DType](
    gram: NDArray[dtype], covariance: NDArray[dtype], n_features: Int,
    penalty: Float64, max_iter: Int, *, lasso: Bool, positive: Bool = False,
) raises -> NDArray[dtype]:
    """Return endpoint rows from a borrowed Gram and one target per covariance column."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] == 0:
        raise Error("ValueError: LARS gram must have shape (atoms, atoms)")
    var atoms = gram.shape[0]
    if covariance.ndim != 2 or covariance.shape[0] != atoms or covariance.shape[1] == 0:
        raise Error("ValueError: LARS cov must have shape (atoms, samples)")
    if n_features < 1 or max_iter < 0 or penalty < 0 or not isfinite(penalty):
        raise Error("ValueError: invalid LARS feature count, iteration limit or penalty")
    if positive and not lasso:
        raise Error("ValueError: Positive constraint not supported for 'lars' coding method.")
    check_finite(gram, "gram", "LARS")
    check_finite(covariance, "cov", "LARS")
    var samples = covariance.shape[1]
    var capacity = min(atoms, max_iter)
    var factor = zeros[dtype](Shape(capacity, capacity))
    var code = zeros[dtype](Shape(samples, atoms))
    for sample in range(samples):
        _least_angle_target(
            gram, covariance, sample, n_features, penalty / Float64(n_features),
            max_iter, lasso, positive, code, factor,
        )
    return code^
