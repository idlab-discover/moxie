# Adapted from sklearn/linear_model/_cd_fast.pyx, scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense cyclic elastic-net residual updates and Gap Safe screening."""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, max, isfinite
from moxie.linear_model._contracts import validate_xy
from moxie.linear_model._linear_base import check_finite, compact_vector


@fieldwise_init
struct CoordinateResult[dtype: DType](Movable):
    var coef: NDArray[Self.dtype]
    var dual_gap: Float64
    var iterations: Int
    var converged: Bool


def _gap[dtype: DType](X: NDArray[dtype], y: NDArray[dtype], w: NDArray[dtype], residual: NDArray[dtype], alpha: Scalar[dtype], beta: Scalar[dtype], positive: Bool, roundoff: Bool, mut correlations: NDArray[dtype]) raises -> Tuple[Scalar[dtype], Scalar[dtype]]:
    var n = y.shape[0]
    var p = w.shape[0]
    var r2 = Scalar[dtype](0)
    var ry = Scalar[dtype](0)
    var w2 = Scalar[dtype](0)
    var w1 = Scalar[dtype](0)
    var dualnorm = Scalar[dtype](0)
    for i in range(n):
        var r = residual.unsafe_get(i)
        r2 += r * r
        ry += r * y.unsafe_get(i)
    for j in range(p):
        var v = w.unsafe_get(j)
        w2 += v * v
        w1 += abs(v)
        var corr = Scalar[dtype](0)
        for i in range(n):
            corr += X.unsafe_get(j * n + i) * residual.unsafe_get(i)
        if alpha != 0:
            corr -= beta * v
            dualnorm = max(dualnorm, corr if positive else abs(corr))
        else:
            dualnorm += corr * corr
        correlations.unsafe_set(j, corr)
    if alpha == 0 and beta == 0:
        return (dualnorm, dualnorm)
    var primal = 0.5 * (r2 + beta * w2) + alpha * w1
    var dual: Scalar[dtype]
    if alpha == 0:
        dual = -0.5 * r2 + ry - dualnorm / (2 * beta)
    else:
        var scale = alpha / dualnorm if dualnorm > alpha else Scalar[dtype](1)
        dual = -0.5 * scale * scale * (r2 + beta * w2) + scale * ry
    var gap = primal - dual
    var eps = Scalar[dtype](1.1920928955078125e-7 if dtype == DType.float32 else 2.220446049250313e-16)
    if roundoff and abs(gap) <= 2 * eps * primal:
        gap = 0
    return (gap, dualnorm)


def coordinate_descent[dtype: DType](X: NDArray[dtype], y: NDArray[dtype], initial: NDArray[dtype], alpha: Float64, beta: Float64, max_iter: Int, tol: Float64, positive: Bool = False) raises -> CoordinateResult[dtype]:
    """Solve 0.5||y-Xw||² + alpha||w||₁ + 0.5 beta||w||².

    X/y are already centered/rescaled. Penalties are unnormalized. The returned
    coefficient vector owns storage. Scratch is one column-major X plus O(n+p).
    Cyclic updates match the pinned reference's screening/early-stop policy.
    """
    validate_xy(X, y)
    if y.ndim != 1 or initial.ndim != 1 or initial.shape[0] != X.shape[1]:
        raise Error("ValueError: coordinate solver needs vector y and coef (n_features,)")
    if not isfinite(alpha) or not isfinite(beta) or alpha < 0 or beta < 0 or not isfinite(tol) or tol < 0 or max_iter < 1:
        raise Error("ValueError: invalid coordinate penalties, tolerance or iteration limit")
    check_finite(X, "X", "coordinate_descent")
    check_finite(y, "y", "coordinate_descent")
    check_finite(initial, "initial", "coordinate_descent")
    var n = X.shape[0]
    var p = X.shape[1]
    var columns = zeros[dtype](Shape(p, n))
    var norms = zeros[dtype](Shape(p))
    var w = compact_vector(initial)
    var target = compact_vector(y)
    var residual = target.copy()
    var correlations = zeros[dtype](Shape(p))
    var active = List[Int]()
    var yy = Scalar[dtype](0)
    for i in range(n):
        yy += target.unsafe_get(i) * target.unsafe_get(i)
    for j in range(p):
        var norm = Scalar[dtype](0)
        for i in range(n):
            var x = X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))
            columns.unsafe_set(j * n + i, x)
            norm += x * x
            residual.unsafe_set(i, residual.unsafe_get(i) - x * w.unsafe_get(j))
        norms.unsafe_set(j, norm)
        active.append(j)
    var a = Scalar[dtype](alpha)
    var b = Scalar[dtype](beta)
    var threshold = Scalar[dtype](tol) * yy
    var pair = _gap(columns, target, w, residual, a, b, positive, False, correlations)
    var gap = pair[0]
    var dualnorm = pair[1]
    if gap <= threshold:
        return CoordinateResult[dtype](w^, Float64(gap), 0, True)
    var iterations = 0
    var converged = False
    # Screening at initialization and after each dual-gap check.
    var screen = True
    for iteration in range(max_iter):
        if screen and a > 0:
            var retained = List[Int]()
            for j in active:
                var norm = norms.unsafe_get(j)
                var keep = False
                if norm > 0:
                    var theta = correlations.unsafe_get(j) / max(a, dualnorm)
                    var distance = (1 - abs(theta)) / sqrt(norm + b)
                    keep = distance <= sqrt(max(Scalar[dtype](0), 2 * gap)) / a
                if keep:
                    retained.append(j)
                else:
                    var old = w.unsafe_get(j)
                    for i in range(n):
                        residual.unsafe_set(i, residual.unsafe_get(i) + old * columns.unsafe_get(j * n + i))
                    w.unsafe_set(j, 0)
            active = retained^
        screen = False
        var largest = Scalar[dtype](0)
        var change = Scalar[dtype](0)
        for j in active:
            var norm = norms.unsafe_get(j)
            if norm == 0:
                continue
            var old = w.unsafe_get(j)
            var corr = Scalar[dtype](0)
            for i in range(n):
                corr += columns.unsafe_get(j * n + i) * residual.unsafe_get(i)
            corr += old * norm
            var value = Scalar[dtype](0)
            if not positive or corr >= 0:
                var sign = Scalar[dtype](1 if corr >= 0 else -1)
                value = sign * max(abs(corr) - a, Scalar[dtype](0)) / (norm + b)
            w.unsafe_set(j, value)
            if value != old:
                for i in range(n):
                    residual.unsafe_set(i, residual.unsafe_get(i) + (old - value) * columns.unsafe_get(j * n + i))
            change = max(change, abs(value - old))
            largest = max(largest, abs(value))
        iterations = iteration + 1
        if largest == 0 or change <= Scalar[dtype](tol) * largest or iterations == max_iter or len(active) <= 1:
            pair = _gap(columns, target, w, residual, a, b, positive, True, correlations)
            gap = pair[0]
            dualnorm = pair[1]
            if gap <= threshold:
                converged = True
                break
            screen = True
    check_finite(w, "coef", "coordinate_descent")
    if not isfinite(gap):
        raise Error("ValueError: coordinate descent produced a nonfinite dual gap")
    return CoordinateResult[dtype](w^, Float64(gap), iterations, converged)
