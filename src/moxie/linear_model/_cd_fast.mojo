# Adapted from sklearn/linear_model/_cd_fast.pyx, scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense elastic-net residual updates and Gap Safe screening."""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, max, isfinite, copysign
from moxie._random import our_rand_r
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


struct _CoordinateDesign[dtype: DType](Movable):
    var columns: NDArray[Self.dtype]
    var norms: NDArray[Self.dtype]
    var target: NDArray[Self.dtype]
    var target_norm: Scalar[Self.dtype]

    def __init__(
        out self, X: NDArray[Self.dtype], y: NDArray[Self.dtype]
    ) raises:
        validate_xy(X, y)
        if y.ndim != 1:
            raise Error("ValueError: coordinate solver needs vector y")
        check_finite(X, "X", "coordinate_descent")
        check_finite(y, "y", "coordinate_descent")
        var n = X.shape[0]
        var p = X.shape[1]
        self.columns = zeros[Self.dtype](Shape(p, n))
        self.norms = zeros[Self.dtype](Shape(p))
        self.target = compact_vector(y)
        self.target_norm = 0
        for i in range(n):
            self.target_norm += self.target.unsafe_get(i) * self.target.unsafe_get(i)
        for j in range(p):
            var norm = Scalar[Self.dtype](0)
            for i in range(n):
                var x = X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))
                self.columns.unsafe_set(j * n + i, x)
                norm += x * x
            self.norms.unsafe_set(j, norm)

    def max_correlation(self, positive: Bool) raises -> Float64:
        var n = self.target.shape[0]
        var largest = Float64(0)
        for j in range(self.columns.shape[0]):
            var correlation = Float64(0)
            for i in range(n):
                correlation += Float64(self.columns.unsafe_get(j * n + i)) * Float64(
                    self.target.unsafe_get(i)
                )
            largest = max(largest, correlation if positive else abs(correlation))
        return largest

    def solve(
        self,
        initial: NDArray[Self.dtype],
        alpha: Float64,
        beta: Float64,
        max_iter: Int,
        tol: Float64,
        positive: Bool,
        *,
        early_stopping: Bool = True,
        random_seed: Optional[UInt32] = None,
    ) raises -> CoordinateResult[Self.dtype]:
        var n = self.target.shape[0]
        var p = self.columns.shape[0]
        if initial.ndim != 1 or initial.shape[0] != p:
            raise Error("ValueError: coordinate solver needs coef (n_features,)")
        if not isfinite(alpha) or not isfinite(beta) or alpha < 0 or beta < 0 or not isfinite(tol) or tol < 0 or max_iter < 1:
            raise Error("ValueError: invalid coordinate penalties, tolerance or iteration limit")
        check_finite(initial, "initial", "coordinate_descent")
        ref columns = self.columns
        ref target = self.target
        ref norms = self.norms
        var w = compact_vector(initial)
        var residual = target.copy()
        var correlations = zeros[Self.dtype](Shape(p))
        var active = List[Int]()
        for j in range(p):
            active.append(j)
            var coefficient = w.unsafe_get(j)
            if coefficient != 0:
                for i in range(n):
                    var x = columns.unsafe_get(j * n + i)
                    residual.unsafe_set(i, residual.unsafe_get(i) - x * coefficient)
        var a = Scalar[Self.dtype](alpha)
        var b = Scalar[Self.dtype](beta)
        var threshold = Scalar[Self.dtype](tol) * self.target_norm
        var pair = _gap(columns, target, w, residual, a, b, positive, False, correlations)
        var gap = pair[0]
        var dualnorm = pair[1]
        if early_stopping and gap <= threshold:
            return CoordinateResult[Self.dtype](w^, Float64(gap), 0, True)
        var iterations = 0
        var converged = False
        var random = Bool(random_seed)
        var state = random_seed.value() if random else UInt32(0)
        if a > 0:
            active = self._screen_active(active, correlations, a, b, gap, dualnorm, w, residual)
        for iteration in range(max_iter):
            var largest = Scalar[Self.dtype](0)
            var change = Scalar[Self.dtype](0)
            for position in range(len(active)):
                var index = Int(our_rand_r(state)) % len(active) if random else position
                var j = active[index]
                var norm = norms.unsafe_get(j)
                if norm == 0:
                    continue
                var old = w.unsafe_get(j)
                var corr = Scalar[Self.dtype](0)
                for i in range(n):
                    corr += columns.unsafe_get(j * n + i) * residual.unsafe_get(i)
                corr += old * norm
                var value = Scalar[Self.dtype](0)
                if not positive or corr >= 0:
                    var sign = Scalar[Self.dtype](1 if corr >= 0 else -1)
                    value = sign * max(abs(corr) - a, Scalar[Self.dtype](0)) / (norm + b)
                w.unsafe_set(j, value)
                if value != old:
                    for i in range(n):
                        residual.unsafe_set(i, residual.unsafe_get(i) + (old - value) * columns.unsafe_get(j * n + i))
                change = max(change, abs(value - old))
                largest = max(largest, abs(value))
            iterations = iteration + 1
            if largest == 0 or change <= Scalar[Self.dtype](tol) * largest or iterations == max_iter or len(active) <= 1:
                pair = _gap(columns, target, w, residual, a, b, positive, True, correlations)
                gap = pair[0]
                dualnorm = pair[1]
                if gap <= threshold:
                    converged = True
                    break
                if a > 0:
                    # The reference screens after the final failed gap check too.
                    active = self._screen_active(active, correlations, a, b, gap, dualnorm, w, residual)
        check_finite(w, "coef", "coordinate_descent")
        if not isfinite(gap):
            raise Error("ValueError: coordinate descent produced a nonfinite dual gap")
        return CoordinateResult[Self.dtype](w^, Float64(gap), iterations, converged)

    def _screen_active(
        self,
        active: List[Int],
        correlations: NDArray[Self.dtype],
        alpha: Scalar[Self.dtype],
        beta: Scalar[Self.dtype],
        gap: Scalar[Self.dtype],
        dualnorm: Scalar[Self.dtype],
        mut coef: NDArray[Self.dtype],
        mut residual: NDArray[Self.dtype],
    ) raises -> List[Int]:
        var retained = List[Int](capacity=len(active))
        var radius = sqrt(max(Scalar[Self.dtype](0), 2 * gap)) / alpha
        var normalization = max(alpha, dualnorm)
        var n = self.target.shape[0]
        for j in active:
            var norm = self.norms.unsafe_get(j)
            if norm > 0:
                var theta = correlations.unsafe_get(j) / normalization
                var distance = (1 - abs(theta)) / sqrt(norm + beta)
                if distance <= radius:
                    retained.append(j)
                    continue
            var old = coef.unsafe_get(j)
            if old != 0:
                for i in range(n):
                    residual.unsafe_set(i, residual.unsafe_get(i) + old * self.columns.unsafe_get(j * n + i))
                coef.unsafe_set(j, 0)
        return retained^


def coordinate_descent[dtype: DType](
    X: NDArray[dtype],
    y: NDArray[dtype],
    initial: NDArray[dtype],
    alpha: Float64,
    beta: Float64,
    max_iter: Int,
    tol: Float64,
    positive: Bool = False,
    *,
    early_stopping: Bool = True,
    random_seed: Optional[UInt32] = None,
) raises -> CoordinateResult[dtype]:
    """Solve 0.5||y-Xw||² + alpha||w||₁ + 0.5 beta||w||².

    X and y are already centered/rescaled. Penalties are unnormalized.
    """
    var design = _CoordinateDesign[dtype](X, y)
    return design.solve(
        initial, alpha, beta, max_iter, tol, positive,
        early_stopping=early_stopping, random_seed=random_seed,
    )


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
