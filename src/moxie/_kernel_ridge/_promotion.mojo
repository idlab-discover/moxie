"""KernelRidge's runtime dtype choices over standard NuMojo storage."""
from std.utils import Variant
from std.math import sqrt
import numojo as nm
from ._validation import dense_cast, floating_dtype, matrix_copy, validate
from ._solve import cholesky, triangular_solve, least_squares

comptime FloatingArray = Variant[
    nm.NDArray[DType.float32], nm.NDArray[DType.float64]
]
comptime FittedArray = Variant[
    nm.NDArray[DType.float32],
    nm.NDArray[DType.float64],
    nm.NDArray[DType.int32],
    nm.NDArray[DType.int64],
]


def solve_promoted[
    kdtype: DType, ydtype: DType
](
    K: nm.NDArray[kdtype],
    y: nm.NDArray[ydtype],
    alpha: List[Float64],
    weights: List[Float64],
    has_weights: Bool,
) raises -> FloatingArray:
    # Weight multiplication allocates y but updates K in its existing dtype.
    if has_weights:
        var targets = dense_cast[DType.float64](y)
        return _solve_weighted(K, targets, alpha, weights, True)
    var targets = dense_cast[floating_dtype(ydtype)](y)
    return _solve_weighted(K, targets, alpha, weights, False)


def _solve_weighted[
    kdtype: DType, ydtype: DType
](
    K: nm.NDArray[kdtype],
    y: nm.NDArray[ydtype],
    alpha: List[Float64],
    weights: List[Float64],
    has_weights: Bool,
) raises -> FloatingArray:
    comptime solve_dtype = DType.float32 if kdtype == DType.float32 and ydtype == DType.float32 else DType.float64
    var A = matrix_copy(K)
    var b = matrix_copy(y)
    var n = K.shape[0]
    if K.shape[1] != n or y.shape[0] != n:
        raise Error("ValueError: incompatible kernel-system shapes")
    var targets = b.shape[1]
    var roots = weights.copy()
    if has_weights:
        for i in range(n):
            roots[i] = sqrt(roots[i])
        for i in range(n):
            for j in range(n):
                A.unsafe_set(
                    i * n + j,
                    SIMD[kdtype, 1](
                        Float64(A.unsafe_get(i * n + j)) * (roots[i] * roots[j])
                    ),
                )
            for t in range(targets):
                b.unsafe_set(
                    i * targets + t,
                    SIMD[ydtype, 1](
                        Float64(b.unsafe_get(i * targets + t)) * roots[i]
                    ),
                )
    validate(A, "weighted kernel", True)
    validate(b, "weighted targets")
    var shared = True
    for value in alpha:
        if value != alpha[0]:
            shared = False
    if shared:
        for i in range(n):
            A.unsafe_set(
                i * n + i,
                SIMD[kdtype, 1](Float64(A.unsafe_get(i * n + i)) + alpha[0]),
            )
        var system = dense_cast[solve_dtype](A)
        var rhs = dense_cast[solve_dtype](b)
        var factor = cholesky(system)
        var result: nm.NDArray[solve_dtype]
        if factor:
            result = triangular_solve(factor.value(), rhs)
        else:
            result = least_squares(system, rhs)
        if has_weights:
            for i in range(n):
                for t in range(targets):
                    result.unsafe_set(
                        i * targets + t,
                        SIMD[solve_dtype, 1](
                            Float64(result.unsafe_get(i * targets + t))
                            * roots[i]
                        ),
                    )
        validate(result, "dual coefficients")
        if y.ndim == 1:
            return FloatingArray(result.reshape(nm.Shape(n)))
        return FloatingArray(result^)
    var result = nm.zeros[kdtype](nm.Shape(n, targets))
    for t in range(targets):
        for i in range(n):
            A.unsafe_set(
                i * n + i,
                SIMD[kdtype, 1](Float64(A.unsafe_get(i * n + i)) + alpha[t]),
            )
        var system = dense_cast[solve_dtype](A)
        var rhs = nm.zeros[solve_dtype](nm.Shape(n, 1))
        for i in range(n):
            rhs.unsafe_set(i, b.unsafe_get(i * targets + t).cast[solve_dtype]())
        var factor = cholesky(system)
        if not factor:
            raise Error(
                "LinAlgError: per-target regularized kernel is not positive"
                " definite"
            )
        var solution = triangular_solve(factor.value(), rhs)
        for i in range(n):
            # sklearn writes each solve to a K.dtype output before unweighting.
            result.unsafe_set(
                i * targets + t, solution.unsafe_get(i).cast[kdtype]()
            )
            A.unsafe_set(
                i * n + i,
                SIMD[kdtype, 1](Float64(A.unsafe_get(i * n + i)) - alpha[t]),
            )
    if has_weights:
        for i in range(n):
            for t in range(targets):
                result.unsafe_set(
                    i * targets + t,
                    SIMD[kdtype, 1](
                        Float64(result.unsafe_get(i * targets + t)) * roots[i]
                    ),
                )
    validate(result, "dual coefficients")
    return FloatingArray(result^)


def prediction_product[
    kdtype: DType, cdtype: DType
](
    K: nm.NDArray[kdtype], coefficients: nm.NDArray[cdtype]
) raises -> FloatingArray:
    comptime dtype = DType.float32 if kdtype == DType.float32 and cdtype == DType.float32 else DType.float64
    var kernel = dense_cast[dtype](K)
    var dual = dense_cast[dtype](matrix_copy(coefficients))
    var result = kernel @ dual
    if coefficients.ndim == 1:
        return FloatingArray(result.reshape(nm.Shape(K.shape[0])))
    return FloatingArray(result^)
