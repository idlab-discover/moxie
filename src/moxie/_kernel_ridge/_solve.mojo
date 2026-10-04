"""Cholesky and minimum-norm Jacobi SVD solves in pure Mojo."""
from std.math import sqrt, isfinite, copysign
import numojo as nm
from ._validation import validate, epsilon, matrix_copy, check_system


def cholesky[
    dtype: DType
](A: nm.NDArray[dtype]) raises -> Optional[nm.NDArray[dtype]]:
    """Factor the symmetric matrix specified by A's upper triangle.

    As in scipy.linalg.solve(assume_a='pos', lower=False), the other triangle
    is ignored. A failed factorization leaves A unchanged for the fallback.
    """
    validate(A, "kernel", True)
    if A.shape[0] != A.shape[1]:
        raise Error("ValueError: kernel must be square")
    var n = A.shape[0]
    var L = nm.zeros[dtype](nm.Shape(n, n))
    for i in range(n):
        for j in range(i + 1):
            var value = A.unsafe_get(
                j * Int(A.strides[0]) + i * Int(A.strides[1])
            )
            for k in range(j):
                value -= L.unsafe_get(i * n + k) * L.unsafe_get(j * n + k)
            if i == j:
                if value <= 0 or not isfinite(value):
                    return None
                L.unsafe_set(i * n + j, sqrt(value))
            else:
                L.unsafe_set(i * n + j, value / L.unsafe_get(j * n + j))
    return L^


def triangular_solve[
    dtype: DType
](factor: nm.NDArray[dtype], y: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    check_system(factor, y)
    var L = matrix_copy(factor)
    for i in range(L.shape[0]):
        if L.unsafe_get(i * L.shape[0] + i) == 0:
            raise Error("LinAlgError: zero triangular diagonal")
    var result = matrix_copy(y)
    var targets = y.shape[1]
    var n = L.shape[0]
    for target in range(targets):
        for i in range(n):
            var value = result.unsafe_get(i * targets + target)
            for j in range(i):
                value -= L.unsafe_get(i * n + j) * result.unsafe_get(
                    j * targets + target
                )
            result.unsafe_set(
                i * targets + target, value / L.unsafe_get(i * n + i)
            )
        for step in range(n):
            var i = n - step - 1
            var value = result.unsafe_get(i * targets + target)
            for j in range(i + 1, n):
                value -= L.unsafe_get(j * n + i) * result.unsafe_get(
                    j * targets + target
                )
            result.unsafe_set(
                i * targets + target, value / L.unsafe_get(i * n + i)
            )
    validate(result, "dual coefficients")
    return result^


def least_squares[
    dtype: DType
](
    A: nm.NDArray[dtype], y: nm.NDArray[dtype], max_sweeps: Int = -1
) raises -> nm.NDArray[dtype]:
    """Minimum-norm solve via one-sided cyclic Jacobi SVD, never A.T @ A.

    Scale the matrix, orthogonalize column pairs, and retain singular values
    greater than eps * sigma_max, matching SciPy lstsq's default rank cutoff.
    Nonconvergence raises instead of returning a partial factorization.
    """
    check_system(A, y)
    var targets = y.shape[1]
    var n = A.shape[0]
    var B = matrix_copy(A)
    var V = nm.zeros[dtype](nm.Shape(n, n))
    var result = nm.zeros[dtype](nm.Shape(n, targets))
    var scale: SIMD[dtype, 1] = 0
    for i in range(n * n):
        scale = max(scale, abs(B.unsafe_get(i)))
    if scale == 0:
        return result^
    for i in range(n * n):
        B.unsafe_set(i, B.unsafe_get(i) / (scale))
    for i in range(n):
        V.unsafe_set(i * n + i, 1)
    var converged = False
    for _ in range(max(40, 6 * n) if max_sweeps < 0 else max_sweeps):
        var changed = False
        for p in range(n):
            for q in range(p + 1, n):
                var a: SIMD[dtype, 1] = 0
                var b: SIMD[dtype, 1] = 0
                var c: SIMD[dtype, 1] = 0
                for i in range(n):
                    var x = B.unsafe_get(i * n + p)
                    var z = B.unsafe_get(i * n + q)
                    a += x * x
                    b += z * z
                    c += x * z
                if (
                    a == 0
                    or b == 0
                    or abs(c) <= epsilon[dtype]() * sqrt(a) * sqrt(b)
                ):
                    continue
                changed = True
                var tau = (b - a) / (2.0 * c)
                var t: SIMD[dtype, 1]
                if abs(tau) > sqrt(SIMD[dtype, 1](1e30)):
                    t = 0.5 / tau
                else:
                    t = copysign(1.0 / (abs(tau) + sqrt(1.0 + tau * tau)), tau)
                var cosine = 1.0 / sqrt(1.0 + t * t)
                var sine = cosine * t
                for i in range(n):
                    var x = B.unsafe_get(i * n + p)
                    var z = B.unsafe_get(i * n + q)
                    B.unsafe_set(i * n + p, cosine * x - sine * z)
                    B.unsafe_set(i * n + q, sine * x + cosine * z)
                    x = V.unsafe_get(i * n + p)
                    z = V.unsafe_get(i * n + q)
                    V.unsafe_set(i * n + p, cosine * x - sine * z)
                    V.unsafe_set(i * n + q, sine * x + cosine * z)
        if not changed:
            converged = True
            break
    if not converged:
        raise Error("LinAlgError: Jacobi least-squares solver did not converge")
    var norms = List[SIMD[dtype, 1]]()
    var largest: SIMD[dtype, 1] = 0
    for j in range(n):
        var norm: SIMD[dtype, 1] = 0
        for i in range(n):
            norm += B.unsafe_get(i * n + j) ** 2
        norm = sqrt(norm)
        norms.append(norm)
        largest = max(largest, norm)
    for j in range(n):
        if norms[j] <= epsilon[dtype]() * largest:
            continue
        for target in range(targets):
            var projection: SIMD[dtype, 1] = 0
            for i in range(n):
                projection += (
                    B.unsafe_get(i * n + j) / norms[j]
                ) * y.unsafe_get(
                    i * Int(y.strides[0]) + target * Int(y.strides[1])
                )
            var coefficient = (projection / norms[j]) / scale
            for i in range(n):
                result.unsafe_set(
                    i * targets + target,
                    result.unsafe_get(i * targets + target)
                    + (V.unsafe_get(i * n + j) * coefficient),
                )
    validate(result, "least-squares coefficients")
    return result^


def _solve_matrix[
    dtype: DType
](
    K: nm.NDArray[dtype],
    y: nm.NDArray[dtype],
    alpha: List[SIMD[dtype, 1]],
    weights: List[SIMD[dtype, 1]],
) raises -> nm.NDArray[dtype]:
    var targets = y.shape[1]
    var n = K.shape[0]
    var weighted_K = matrix_copy(K)
    var weighted_y = matrix_copy(y)
    var roots = weights.copy()
    for i in range(n):
        roots[i] = sqrt(weights[i])
    for i in range(n):
        for j in range(n):
            weighted_K.unsafe_set(
                i * n + j,
                weighted_K.unsafe_get(i * n + j) * (roots[i] * roots[j]),
            )
        for j in range(targets):
            weighted_y.unsafe_set(
                i * targets + j,
                weighted_y.unsafe_get(i * targets + j) * (roots[i]),
            )
    validate(weighted_K, "weighted kernel", True)
    validate(weighted_y, "weighted targets")
    var shared = True
    for value in alpha:
        if value != alpha[0]:
            shared = False
    var result = nm.zeros[dtype](nm.Shape(n, targets))
    if shared:
        for i in range(n):
            weighted_K.unsafe_set(
                i * n + i, weighted_K.unsafe_get(i * n + i) + (alpha[0])
            )
        validate(weighted_K, "regularized kernel", True)
        var L = cholesky(weighted_K)
        if L:
            result = triangular_solve(L.value(), weighted_y)
        else:
            result = least_squares(weighted_K, weighted_y)
    else:
        # The sklearn unequal-alpha branch has no least-squares fallback.
        for target in range(targets):
            var A = matrix_copy(weighted_K)
            var b = nm.zeros[dtype](nm.Shape(n, 1))
            for i in range(n):
                A.unsafe_set(
                    i * n + i, A.unsafe_get(i * n + i) + (alpha[target])
                )
                b.unsafe_set(i, weighted_y.unsafe_get(i * targets + target))
            validate(A, "regularized kernel", True)
            var L = cholesky(A)
            if not L:
                raise Error(
                    "LinAlgError: per-target regularized kernel is not positive"
                    " definite"
                )
            var solution = triangular_solve(L.value(), b)
            for i in range(n):
                result.unsafe_set(i * targets + target, solution.unsafe_get(i))
    for i in range(n):
        for j in range(targets):
            result.unsafe_set(
                i * targets + j, result.unsafe_get(i * targets + j) * (roots[i])
            )
    validate(result, "dual coefficients")
    return result^


def solve_kernel[
    dtype: DType
](
    K: nm.NDArray[dtype],
    y: nm.NDArray[dtype],
    alpha: List[SIMD[dtype, 1]],
    weights: List[SIMD[dtype, 1]],
) raises -> nm.NDArray[dtype]:
    validate(K, "kernel", True)
    validate(y, "targets")
    var n = K.shape[0]
    var targets = y.shape[1] if y.ndim == 2 else 1
    if K.shape[1] != n or y.shape[0] != n:
        raise Error("ValueError: incompatible kernel-system shapes")
    if len(alpha) != 1 and len(alpha) != targets:
        raise Error("ValueError: alpha must have length one or target count")
    if len(weights) != 1 and len(weights) != n:
        raise Error("ValueError: weights must have length one or sample count")
    for a in alpha:
        if a < 0 or not isfinite(a):
            raise Error("ValueError: alpha must be finite and nonnegative")
    var expanded = List[SIMD[dtype, 1]]()
    for i in range(n):
        var w = weights[0 if len(weights) == 1 else i]
        if w < 0 or not isfinite(w):
            raise Error("ValueError: weights must be finite and nonnegative")
        expanded.append(w)
    var targets_matrix = matrix_copy(y)
    var result = _solve_matrix(K, targets_matrix, alpha, expanded)
    if y.ndim == 1:
        return result.reshape(nm.Shape(n))
    return result^


def solve_kernel_scalar[
    dtype: DType
](
    K: nm.NDArray[dtype],
    y: nm.NDArray[dtype],
    alpha: SIMD[dtype, 1] = 1,
    weight: SIMD[dtype, 1] = 1,
) raises -> nm.NDArray[dtype]:
    return solve_kernel(
        K, y, List[SIMD[dtype, 1]]([alpha]), List[SIMD[dtype, 1]]([weight])
    )
