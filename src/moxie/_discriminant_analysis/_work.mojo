"""Float64 working arrays on NuMojo storage and checked mojo-linalg calls.

Every estimator statistic and factorization is computed in Float64 compact
row-major `NDArray[DType.float64]` buffers created here (offset 0, strides
(n, 1)); public attributes are cast back to the estimator dtype. The
`get`/`put` accessors are only valid for such compact work arrays.
"""
from numojo import NDArray, Shape, zeros
from std.math import isfinite, sqrt, abs
from mojo_linalg import (
    NumericalError,
    SVDFactor,
    SymmetricEigenFactor,
    CholeskyFactor,
    LeastSquaresResult,
    svd,
    symmetric_eigh,
    cholesky,
    triangular_solve,
    least_squares,
)

comptime Work = NDArray[DType.float64]
"""Compact row-major Float64 working matrix or vector."""


@always_inline
def get(a: Work, i: Int, j: Int) raises -> Float64:
    """Entry (i, j) of a compact work matrix."""
    return a.unsafe_get(i * a.shape[1] + j)


@always_inline
def put(mut a: Work, i: Int, j: Int, value: Float64) raises:
    """Set entry (i, j) of a compact work matrix."""
    a.unsafe_set(i * a.shape[1] + j, value)


def matrix(rows: Int, cols: Int) raises -> Work:
    """Zero-initialized compact work matrix."""
    return zeros[DType.float64](Shape(rows, cols))


def vector(n: Int) raises -> Work:
    """Zero-initialized work vector."""
    return zeros[DType.float64](Shape(n))


def input_matrix[
    dtype: DType
](X: NDArray[dtype], estimator: String, *, min_samples: Int = 1) raises -> Work:
    """Validate a 2-D finite sample matrix and return its compact Float64 copy.

    Honors arbitrary NuMojo strides/offsets; the caller's array is not retained.
    """
    comptime assert (
        dtype == DType.float32 or dtype == DType.float64
    ), "discriminant analysis supports Float32 and Float64 input"
    if X.ndim != 2:
        raise Error(
            String("ValueError: ", estimator, " expected a 2D array, got a ")
            + String(X.ndim)
            + "D array instead"
        )
    var n = X.shape[0]
    var p = X.shape[1]
    if n < min_samples:
        raise Error(
            String("ValueError: Found array with ", n, " sample(s) (shape=(", n, ", ", p)
            + String(")) while a minimum of ", min_samples, " is required by ", estimator, ".")
        )
    if p < 1:
        raise Error(
            String("ValueError: Found array with 0 feature(s) (shape=(", n, ", 0)) while a")
            + String(" minimum of 1 is required by ", estimator, ".")
        )
    var s0 = Int(X.strides[0])
    var s1 = Int(X.strides[1])
    var work = matrix(n, p)
    for i in range(n):
        for j in range(p):
            var value = Float64(X.unsafe_get(i * s0 + j * s1))
            if not isfinite(value):
                raise Error(
                    String("ValueError: Input X contains NaN or infinity (row ", i, ", column ", j)
                    + String(") for ", estimator, ".")
                )
            work.unsafe_set(i * p + j, value)
    return work^


def cast_array[dtype: DType](a: Work) raises -> NDArray[dtype]:
    """Copy a compact work vector/matrix into a new array of the estimator dtype."""
    var out: NDArray[dtype]
    if a.ndim == 1:
        out = zeros[dtype](Shape(a.shape[0]))
    else:
        out = zeros[dtype](Shape(a.shape[0], a.shape[1]))
    for i in range(a.size):
        out.unsafe_set(i, SIMD[dtype, 1](a.unsafe_get(i)))
    return out^


def widen[dtype: DType](a: NDArray[dtype]) raises -> Work:
    """Compact Float64 copy of a (possibly strided) estimator-owned vector/matrix."""
    if a.ndim == 1:
        var out = vector(a.shape[0])
        var s = Int(a.strides[0])
        for i in range(a.shape[0]):
            out.unsafe_set(i, Float64(a.unsafe_get(i * s)))
        return out^
    var rows = a.shape[0]
    var cols = a.shape[1]
    var out = matrix(rows, cols)
    var s0 = Int(a.strides[0])
    var s1 = Int(a.strides[1])
    for i in range(rows):
        for j in range(cols):
            out.unsafe_set(i * cols + j, Float64(a.unsafe_get(i * s0 + j * s1)))
    return out^


def matmul(a: Work, b: Work) raises -> Work:
    """Return a @ b for compact work matrices."""
    debug_assert(a.shape[1] == b.shape[0], "matmul inner dimension")
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]
    var out = matrix(m, n)
    for i in range(m):
        for t in range(k):
            var x = a.unsafe_get(i * k + t)
            if x == 0:
                continue
            for j in range(n):
                out.unsafe_set(i * n + j, out.unsafe_get(i * n + j) + x * b.unsafe_get(t * n + j))
    return out^


def matmul_transposed(a: Work, b: Work) raises -> Work:
    """Return a @ b.T for compact work matrices (row dot products)."""
    debug_assert(a.shape[1] == b.shape[1], "matmul_transposed inner dimension")
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[0]
    var out = matrix(m, n)
    for i in range(m):
        for j in range(n):
            var acc: Float64 = 0
            for t in range(k):
                acc += a.unsafe_get(i * k + t) * b.unsafe_get(j * k + t)
            out.unsafe_set(i * n + j, acc)
    return out^


def transpose(a: Work) raises -> Work:
    """Return a compact copy of a.T."""
    var m = a.shape[0]
    var n = a.shape[1]
    var out = matrix(n, m)
    for i in range(m):
        for j in range(n):
            out.unsafe_set(j * m + i, a.unsafe_get(i * n + j))
    return out^


def linalg_error(err: NumericalError, context: String) -> Error:
    """Convert a typed mojo-linalg failure into a descriptive LinAlgError."""
    return Error(
        String("LinAlgError: ", context, " failed in ", err.operation, " (", err.kind, ", index ")
        + String(err.index, ")")
    )


def householder_r(a: Work) raises -> Work:
    """Upper-triangular R (n, n) of an unpivoted Householder QR of a tall (m >= n) matrix.

    Only R is formed (Q is discarded). A = Q R implies A and R share singular values
    and right singular vectors, the standard LAPACK gesvd preprocessing for m >> n.
    Scaled column norms avoid overflow. Consolidation candidate for mojo-linalg.
    """
    var m = a.shape[0]
    var n = a.shape[1]
    debug_assert(m >= n, "householder_r expects a tall matrix")
    var w = a.copy()
    var v = List[Float64](length=m, fill=0.0)
    for k in range(n):
        var scale: Float64 = 0
        for i in range(k, m):
            scale = max(scale, abs(w.unsafe_get(i * n + k)))
        if scale == 0:
            continue
        var norm2: Float64 = 0
        for i in range(k, m):
            var x = w.unsafe_get(i * n + k) / scale
            v[i] = x
            norm2 += x * x
        var alpha = -sqrt(norm2) if v[k] >= 0 else sqrt(norm2)
        v[k] -= alpha
        var vnorm2: Float64 = 0
        for i in range(k, m):
            vnorm2 += v[i] * v[i]
        if vnorm2 == 0:
            continue
        for j in range(k, n):
            var dot: Float64 = 0
            for i in range(k, m):
                dot += v[i] * w.unsafe_get(i * n + j)
            var f = 2.0 * dot / vnorm2
            for i in range(k, m):
                w.unsafe_set(i * n + j, w.unsafe_get(i * n + j) - f * v[i])
    var r = matrix(n, n)
    for i in range(n):
        for j in range(i, n):
            r.unsafe_set(i * n + j, w.unsafe_get(i * n + j))
    return r^


def checked_svd(a: Work, context: String) raises -> SVDFactor[DType.float64]:
    """Economy SVD (descending singular values) with contextual errors.

    Callers here only use singular values and right singular vectors; for tall
    inputs (rows > 2 * cols) the Jacobi SVD runs on the Householder R factor.
    `u()` of the returned factor is then R's, not the input's.
    """
    try:
        if a.shape[0] > 2 * a.shape[1]:
            return svd(householder_r(a))
        return svd(a)
    except err:
        raise linalg_error(err, context)


def checked_eigh(a: Work, context: String) raises -> SymmetricEigenFactor[DType.float64]:
    """Symmetric eigendecomposition (ascending values) with contextual errors."""
    try:
        return symmetric_eigh(a)
    except err:
        raise linalg_error(err, context)


def checked_cholesky(a: Work, context: String) raises -> CholeskyFactor[DType.float64]:
    """Cholesky factor of an SPD matrix; failure reports non-positive-definiteness."""
    try:
        return cholesky(a)
    except err:
        if err.kind == "not_positive_definite":
            raise Error(
                String("LinAlgError: ", context, ": the leading minor of order ", err.index + 1)
                + " is not positive definite"
            )
        raise linalg_error(err, context)


def checked_triangular_solve(
    t: Work, b: Work, *, transpose: Bool, context: String
) raises -> Work:
    """Solve op(L) X = B for lower-triangular L."""
    try:
        return triangular_solve(t, b, lower=True, transpose=transpose)
    except err:
        raise linalg_error(err, context)


def checked_least_squares(
    a: Work, b: Work, *, rtol: Float64, context: String
) raises -> Work:
    """Minimum-norm least-squares solution retaining s > rtol * s_max."""
    try:
        var result = least_squares(a, b, rtol=rtol, atol=0.0)
        return result.solution.copy()
    except err:
        raise linalg_error(err, context)
