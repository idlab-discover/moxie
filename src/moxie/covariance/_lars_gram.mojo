# Adapted from scikit-learn sklearn/linear_model/_least_angle.py;
# authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Private Float64 least-angle Gram path used by graphical lasso.

Matches its method='lars', return_path=False call, including interpolation,
500 path steps, eps-floored Cholesky pivots and degenerate-column rejection.
This is not a public linear_model path or a LassoLars implementation.
"""
from numojo import NDArray, Shape, zeros
from mojo_linalg import triangular_solve
from std.math import abs, sqrt, isfinite, round

comptime _TINY32: Float64 = 1.1754943508222875e-38
comptime _EQUALITY_TOL: Float64 = 1.1920928955078125e-7


def lars_gram(gram: NDArray[DType.float64], xy: NDArray[DType.float64],
              penalty: Float64, eps: Float64) raises -> NDArray[DType.float64]:
    """Solve the graphical row path on compact (p,p)/(p,) inputs; preserve both.

    penalty is unnormalized: sklearn alpha_min=penalty/p, n_samples=p.
    """
    var p = xy.size
    var coef = zeros[DType.float64](Shape(p))
    if p == 0:
        return coef^
    var previous = coef.copy()
    var correlations = xy.copy()
    var indices = List[Int]()
    for j in range(p):
        indices.append(j)
    var signs = List[Float64]()
    var lower = zeros[DType.float64](Shape(0, 0))
    var active = 0
    var iteration = 0
    var crossing = False
    var previous_alpha: Float64 = 0
    var alpha_min = penalty / Float64(p)
    while True:
        var candidate = active
        var C: Float64 = 0
        for j in range(active, p):
            if abs(correlations.unsafe_get(j)) > C:
                C = abs(correlations.unsafe_get(j))
                candidate = j
        var alpha = C / Float64(p)
        if alpha <= alpha_min + _EQUALITY_TOL:
            if abs(alpha - alpha_min) > _EQUALITY_TOL and iteration > 0:
                var fraction = (previous_alpha - alpha_min) / (previous_alpha - alpha)
                for j in range(p):
                    coef.unsafe_set(j, previous.unsafe_get(j) + fraction * (coef.unsafe_get(j) - previous.unsafe_get(j)))
            break
        if iteration >= 500 or active >= p:
            break
        if not crossing:
            var selected = correlations.unsafe_get(candidate)
            correlations.unsafe_set(candidate, correlations.unsafe_get(active))
            correlations.unsafe_set(active, selected)
            var original = indices[candidate]
            indices[candidate] = indices[active]
            indices[active] = original
            var border = zeros[DType.float64](Shape(active))
            for j in range(active):
                border.unsafe_set(j, gram.unsafe_get(original * p + indices[j]))
            var solved_border = triangular_solve(lower, border)
            var squared: Float64 = 0
            for j in range(active):
                squared += solved_border.unsafe_get(j) * solved_border.unsafe_get(j)
            var diagonal = max(sqrt(abs(gram.unsafe_get(original * p + original) - squared)), eps)
            if diagonal < 1e-7:
                # Preserve sklearn's permutation and zero this rejected correlation.
                correlations.unsafe_set(active, correlations.unsafe_get(candidate))
                correlations.unsafe_set(candidate, 0)
                continue
            var extended = zeros[DType.float64](Shape(active + 1, active + 1))
            for r in range(active):
                for c in range(r + 1):
                    extended.unsafe_set(r * (active + 1) + c, lower.unsafe_get(r * active + c))
                extended.unsafe_set(active * (active + 1) + r, solved_border.unsafe_get(r))
            extended.unsafe_set(active * (active + 1) + active, diagonal)
            lower = extended^
            signs.append(1.0 if selected > 0 else -1.0 if selected < 0 else 0.0)
            active += 1
        var sign_array = zeros[DType.float64](Shape(active))
        for j in range(active):
            sign_array.unsafe_set(j, signs[j])
        var direction = triangular_solve(lower, triangular_solve(lower, sign_array), transpose=True)
        var signed_sum: Float64 = 0
        for j in range(active):
            signed_sum += direction.unsafe_get(j) * signs[j]
        var angular: Float64 = 1
        if active == 1 and direction.unsafe_get(0) == 0:
            direction.unsafe_set(0, 1)
        else:
            angular = 1.0 / sqrt(signed_sum)
            if not isfinite(angular):
                var regularized = lower.copy()
                var bump = eps
                while not isfinite(angular):
                    if bump <= 0 or not isfinite(bump):
                        raise Error("FloatingPointError: ill-conditioned LARS equiangular direction")
                    for j in range(active):
                        var offset = j * active + j
                        regularized.unsafe_set(offset, regularized.unsafe_get(offset) + bump)
                    direction = triangular_solve(regularized, triangular_solve(regularized, sign_array), transpose=True)
                    signed_sum = 0
                    for j in range(active):
                        signed_sum += direction.unsafe_get(j) * signs[j]
                    angular = 1.0 / sqrt(max(signed_sum, eps))
                    bump *= 2
            for j in range(active):
                direction.unsafe_set(j, direction.unsafe_get(j) * angular)
        var equicorrelation = zeros[DType.float64](Shape(p - active))
        var step = C / angular
        for j in range(active, p):
            var value: Float64 = 0
            for k in range(active):
                value += gram.unsafe_get(indices[k] * p + indices[j]) * direction.unsafe_get(k)
            # np.around(..., decimals=np.finfo(float64).precision).
            value = round(value * 1e15) / 1e15
            equicorrelation.unsafe_set(j - active, value)
            var first = (C - correlations.unsafe_get(j)) / (angular - value + _TINY32)
            var second = (C + correlations.unsafe_get(j)) / (angular + value + _TINY32)
            if first > 0:
                step = min(step, first)
            if second > 0:
                step = min(step, second)
        var zero_step = Float64.MAX
        for j in range(active):
            var crossing_step = -coef.unsafe_get(indices[j]) / (direction.unsafe_get(j) + _TINY32)
            if crossing_step > 0:
                zero_step = min(zero_step, crossing_step)
        crossing = zero_step < step
        if crossing:
            for j in range(active):
                if -coef.unsafe_get(indices[j]) / (direction.unsafe_get(j) + _TINY32) == zero_step:
                    signs[j] = -signs[j]
        previous = coef.copy()
        previous_alpha = alpha
        for j in range(active):
            coef.unsafe_set(indices[j], previous.unsafe_get(indices[j]) + step * direction.unsafe_get(j))
        for j in range(active, p):
            correlations.unsafe_set(j, correlations.unsafe_get(j) - step * equicorrelation.unsafe_get(j - active))
        iteration += 1
    return coef^
