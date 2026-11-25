# Adapted from sklearn/linear_model/_cd_fast.pyx, scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense cyclic elastic-net residual updates and Gap Safe screening."""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, max, isfinite, copysign
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
    return (_dual_gap(r2, ry, w2, w1, dualnorm, alpha, beta, roundoff), dualnorm)


def _dual_gap[dtype: DType](
    r2: Scalar[dtype], ry: Scalar[dtype], w2: Scalar[dtype], w1: Scalar[dtype],
    dualnorm: Scalar[dtype], alpha: Scalar[dtype], beta: Scalar[dtype], roundoff: Bool,
) -> Scalar[dtype]:
    if alpha == 0 and beta == 0:
        return dualnorm
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
    return gap


def coordinate_descent[dtype: DType](X: NDArray[dtype], y: NDArray[dtype], initial: NDArray[dtype], alpha: Float64, beta: Float64, max_iter: Int, tol: Float64, positive: Bool = False, *, early_stopping: Bool = True) raises -> CoordinateResult[dtype]:
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
    if early_stopping and gap <= threshold:
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


def _gram_gap(
    coef: NDArray[DType.float64], product: NDArray[DType.float64], cross: NDArray[DType.float64],
    target_norm2: Float64, alpha: Float64, beta: Float64, roundoff: Bool,
    mut correlations: NDArray[DType.float64],
) -> Tuple[Float64, Float64]:
    var cross_dot = Float64(0)
    var gram_dot = Float64(0)
    var w1 = Float64(0)
    var w2 = Float64(0)
    var dualnorm = Float64(0)
    for j in range(coef.size):
        var value = coef.unsafe_get(j)
        var q = cross.unsafe_get(j)
        var qw = product.unsafe_get(j)
        cross_dot += q * value
        gram_dot += qw * value
        w1 += abs(value)
        w2 += value * value
        var correlation = q - qw
        if alpha != 0:
            correlation -= beta * value
            dualnorm = max(dualnorm, abs(correlation))
        else:
            dualnorm += correlation * correlation
        correlations.unsafe_set(j, correlation)
    var r2 = target_norm2 + gram_dot - 2 * cross_dot
    var ry = target_norm2 - cross_dot
    return (_dual_gap(r2, ry, w2, w1, dualnorm, alpha, beta, roundoff), dualnorm)


def _update_gram_product(
    gram: NDArray[DType.float64], column: Int, delta: Float64,
    mut product: NDArray[DType.float64],
):
    # The symmetric Gram is contiguous, so this column update reads a row.
    var offset = column * product.size
    for j in range(product.size):
        product.unsafe_set(j, product.unsafe_get(j) + delta * gram.unsafe_get(offset + j))


def coordinate_descent_gram(
    gram: NDArray[DType.float64], cross: NDArray[DType.float64], var coef: NDArray[DType.float64],
    target_norm2: Float64, alpha: Float64, beta: Float64, max_iter: Int, tol: Float64,
) raises -> CoordinateResult[DType.float64]:
    """Cyclic elastic-net steps on a contiguous symmetric Gram system."""
    var p = coef.size
    var product = zeros[DType.float64](Shape(p))
    var correlations = zeros[DType.float64](Shape(p))
    var active = List[Int](capacity=p)
    for j in range(p):
        var diagonal = gram.unsafe_get(j * p + j)
        if diagonal != 0 and diagonal + beta <= 0:
            raise Error("ValueError: Gram coordinate descent requires positive penalized curvature")
        active.append(j)
        var value = coef.unsafe_get(j)
        if value != 0:
            _update_gram_product(gram, j, value, product)
    var threshold = tol * target_norm2
    var gap, dualnorm = _gram_gap(coef, product, cross, target_norm2, alpha, beta, False, correlations)
    var screen = alpha > 0
    for iteration in range(max_iter):
        if screen:
            var radius = sqrt(2 * abs(gap)) / alpha
            var retained = List[Int](capacity=len(active))
            for j in active:
                var diagonal = gram.unsafe_get(j * p + j)
                var theta = correlations.unsafe_get(j) / max(alpha, dualnorm)
                if diagonal != 0 and (1 - abs(theta)) <= radius * sqrt(diagonal + beta):
                    retained.append(j)
                else:
                    var old = coef.unsafe_get(j)
                    if old != 0:
                        _update_gram_product(gram, j, -old, product)
                        coef.unsafe_set(j, 0)
            active = retained^
            screen = False
        var largest = Float64(0)
        var change = Float64(0)
        for j in active:
            var diagonal = gram.unsafe_get(j * p + j)
            if diagonal == 0:
                continue
            var old = coef.unsafe_get(j)
            var corr = cross.unsafe_get(j) - product.unsafe_get(j) + old * diagonal
            var value = copysign(max(abs(corr) - alpha, 0.0), corr) / (diagonal + beta)
            coef.unsafe_set(j, value)
            if value != old:
                _update_gram_product(gram, j, value - old, product)
            largest = max(largest, abs(value))
            change = max(change, abs(value - old))
        if largest == 0 or change <= tol * largest or len(active) <= 1 or iteration + 1 == max_iter:
            gap, dualnorm = _gram_gap(coef, product, cross, target_norm2, alpha, beta, True, correlations)
            if gap <= threshold:
                check_finite(coef, "coef", "coordinate_descent_gram")
                return CoordinateResult[DType.float64](coef^, gap, iteration + 1, True)
            screen = alpha > 0
    check_finite(coef, "coef", "coordinate_descent_gram")
    if not isfinite(gap):
        raise Error("ValueError: Gram coordinate descent produced a nonfinite dual gap")
    return CoordinateResult[DType.float64](coef^, gap, max_iter, False)
