"""Private dense numerical helpers for Gaussian-process kernels.

All helpers operate on owned, C-contiguous Float64 NuMojo matrices produced by
`as_dense64`. They allocate only their result arrays and accumulate in Float64.
"""
from numojo import NDArray, Shape, zeros
from std.math import abs, gamma, isfinite, isnan, sqrt
from std.sys import llvm_intrinsic

from moxie._gp_linalg import bessel_kv

from moxie.gaussian_process._plumbing import Dense64


@always_inline
def fexp(x: Float64) -> Float64:
    """Correctly rounded-quality exp (libm via LLVM); std.math.exp is less accurate."""
    return llvm_intrinsic["llvm.exp", Float64, has_side_effect=False](x)


@always_inline
def flog(x: Float64) -> Float64:
    """libm-quality natural logarithm; std.math.log is less accurate."""
    return llvm_intrinsic["llvm.log", Float64, has_side_effect=False](x)


@always_inline
def fpow(x: Float64, y: Float64) -> Float64:
    """libm-quality power (numpy `**` semantics for float operands)."""
    return llvm_intrinsic["llvm.pow", Float64, has_side_effect=False](x, y)


def matern_general_values(
    nu: Float64, dists: Dense64, same: Bool, epsilon: Float64
) raises -> Dense64:
    """sklearn general-nu Matern: 2^(1-nu)/Gamma(nu) * t^nu * K_nu(t).

    t = sqrt(2 nu) * d with exact zero distances replaced by machine epsilon,
    as in sklearn. The caller overwrites the diagonal when `same`.
    """
    var n = dists.shape[0]
    var m = dists.shape[1]
    var out = zeros[DType.float64](Shape(n, m))
    var src = dists.unsafe_ptr()
    var dst = out.unsafe_ptr()
    var scale = fpow(2.0, 1.0 - nu) / gamma(nu)
    var root = sqrt(2.0 * nu)
    for i in range(n):
        for j in range(m):
            if same and i == j:
                continue
            var d = src[unsafe_offset = i * m + j]
            if d == 0.0:
                d += epsilon
            var t = root * d
            var value = scale
            value *= fpow(t, nu)
            value *= bessel_kv(nu, t)
            dst[unsafe_offset = i * m + j] = value
    return out^


def as_dense64[
    dtype: DType
](X: NDArray[dtype], context: String) raises -> Dense64:
    """Return an owned C-contiguous Float64 `(n, d)` copy of dense input.

    Rank-1 input is treated as a single sample, like `numpy.atleast_2d`.
    Float32 input is widened exactly; strided views are read through strides.
    """
    comptime assert (
        dtype == DType.float32 or dtype == DType.float64
    ), "kernel inputs must be Float32 or Float64"
    if X.ndim != 1 and X.ndim != 2:
        raise Error(
            "ValueError: "
            + context
            + " expected a 2-D dense array, got rank "
            + String(X.ndim)
        )
    var rows = X.shape[0] if X.ndim == 2 else 1
    var cols = X.shape[1] if X.ndim == 2 else X.shape[0]
    var row_stride = Int(X.strides[0]) if X.ndim == 2 else 0
    var col_stride = Int(X.strides[1]) if X.ndim == 2 else Int(X.strides[0])
    var out = zeros[DType.float64](Shape(rows, cols))
    var target = out.unsafe_ptr()
    for i in range(rows):
        for j in range(cols):
            target[unsafe_offset = i * cols + j] = X.unsafe_get(
                i * row_stride + j * col_stride
            ).cast[DType.float64]()
    return out^


def scale_columns(X: Dense64, scale: List[Float64]) raises -> Dense64:
    """Divide each column by its length scale (one scale broadcasts)."""
    var n = X.shape[0]
    var d = X.shape[1]
    var out = zeros[DType.float64](Shape(n, d))
    var src = X.unsafe_ptr()
    var dst = out.unsafe_ptr()
    for i in range(n):
        for j in range(d):
            var s = scale[0] if len(scale) == 1 else scale[j]
            dst[unsafe_offset = i * d + j] = src[unsafe_offset = i * d + j] / s
    return out^


def squared_distances(A: Dense64, B: Dense64, same: Bool) raises -> Dense64:
    """Pairwise squared Euclidean distances, exact zero diagonal when same."""
    var n = A.shape[0]
    var m = B.shape[0]
    var d = A.shape[1]
    var out = zeros[DType.float64](Shape(n, m))
    var a = A.unsafe_ptr()
    var b = B.unsafe_ptr()
    var o = out.unsafe_ptr()
    for i in range(n):
        for j in range(m):
            if same and j < i:
                o[unsafe_offset = i * m + j] = o[unsafe_offset = j * m + i]
                continue
            if same and j == i:
                continue
            var acc: Float64 = 0.0
            for k in range(d):
                var diff = a[unsafe_offset = i * d + k] - b[unsafe_offset = j * d + k]
                acc += diff * diff
            o[unsafe_offset = i * m + j] = acc
    return out^


def euclidean_from_squared(D2: Dense64) raises -> Dense64:
    """Elementwise square root of a squared-distance matrix."""
    var out = zeros[DType.float64](Shape(D2.shape[0], D2.shape[1]))
    var src = D2.unsafe_ptr()
    var dst = out.unsafe_ptr()
    for i in range(D2.size):
        dst[unsafe_offset=i] = sqrt(src[unsafe_offset=i])
    return out^


def manhattan_distances(A: Dense64, B: Dense64) raises -> Dense64:
    """Pairwise L1 distances."""
    var n = A.shape[0]
    var m = B.shape[0]
    var d = A.shape[1]
    var out = zeros[DType.float64](Shape(n, m))
    var a = A.unsafe_ptr()
    var b = B.unsafe_ptr()
    var o = out.unsafe_ptr()
    for i in range(n):
        for j in range(m):
            var acc: Float64 = 0.0
            for k in range(d):
                acc += abs(a[unsafe_offset = i * d + k] - b[unsafe_offset = j * d + k])
            o[unsafe_offset = i * m + j] = acc
    return out^


def inner_products(A: Dense64, B: Dense64) raises -> Dense64:
    """Matrix of row inner products `A @ B.T`."""
    var n = A.shape[0]
    var m = B.shape[0]
    var d = A.shape[1]
    var out = zeros[DType.float64](Shape(n, m))
    var a = A.unsafe_ptr()
    var b = B.unsafe_ptr()
    var o = out.unsafe_ptr()
    for i in range(n):
        for j in range(m):
            var acc: Float64 = 0.0
            for k in range(d):
                acc += a[unsafe_offset = i * d + k] * b[unsafe_offset = j * d + k]
            o[unsafe_offset = i * m + j] = acc
    return out^


def row_normalized(A: Dense64) raises -> Dense64:
    """Rows scaled to unit L2 norm; zero rows stay zero (sklearn normalize)."""
    var n = A.shape[0]
    var d = A.shape[1]
    var out = A.copy()
    var o = out.unsafe_ptr()
    for i in range(n):
        var acc: Float64 = 0.0
        for k in range(d):
            var v = o[unsafe_offset = i * d + k]
            acc += v * v
        var norm = sqrt(acc)
        if norm == 0.0:
            continue
        for k in range(d):
            o[unsafe_offset = i * d + k] = o[unsafe_offset = i * d + k] / norm
    return out^


def additive_chi2(A: Dense64, B: Dense64, same: Bool) raises -> Dense64:
    """`-sum((x-y)^2/(x+y))`, skipping zero denominators; nonnegative input."""
    var n = A.shape[0]
    var m = B.shape[0]
    var d = A.shape[1]
    var a = A.unsafe_ptr()
    var b = B.unsafe_ptr()
    for i in range(A.size):
        if a[unsafe_offset=i] < 0:
            raise Error("ValueError: X contains negative values.")
    if not same:
        for i in range(B.size):
            if b[unsafe_offset=i] < 0:
                raise Error("ValueError: Y contains negative values.")
    var out = zeros[DType.float64](Shape(n, m))
    var o = out.unsafe_ptr()
    for i in range(n):
        for j in range(m):
            var acc: Float64 = 0.0
            for k in range(d):
                var x = a[unsafe_offset = i * d + k]
                var y = b[unsafe_offset = j * d + k]
                var denom = x + y
                if denom != 0.0:
                    var diff = x - y
                    acc -= diff * diff / denom
            o[unsafe_offset = i * m + j] = acc
    return out^


def fill(n: Int, m: Int, value: Float64) raises -> Dense64:
    """Owned `(n, m)` matrix filled with value."""
    var out = zeros[DType.float64](Shape(n, m))
    var o = out.unsafe_ptr()
    for i in range(n * m):
        o[unsafe_offset=i] = value
    return out^


def vector_fill(n: Int, value: Float64) raises -> Dense64:
    """Owned `(n,)` vector filled with value."""
    var out = zeros[DType.float64](Shape(n))
    var o = out.unsafe_ptr()
    for i in range(n):
        o[unsafe_offset=i] = value
    return out^


def empty_gradient(n: Int) raises -> Dense64:
    """The `(n, n, 0)` gradient of a kernel without free hyperparameters."""
    return zeros[DType.float64](Shape(n, n, 0))


def multiply_gradient(mut G: Dense64, K: Dense64) raises:
    """Broadcast a covariance over the last gradient axis, as NumPy does."""
    var rows = G.shape[0]
    var cols = max(G.shape[1], K.shape[1])
    if rows != K.shape[0] or (
        G.shape[1] != K.shape[1] and G.shape[1] != 1 and K.shape[1] != 1
    ):
        raise Error("ValueError: kernel gradient and covariance cannot be broadcast")
    var width = G.shape[2]
    if G.shape[1] == K.shape[1]:
        for i in range(K.size):
            for c in range(width):
                G.unsafe_set(i * width + c, G.unsafe_get(i * width + c) * K.unsafe_get(i))
        return
    var out = zeros[DType.float64](Shape(rows, cols, width))
    for i in range(rows):
        for j in range(cols):
            var gj = 0 if G.shape[1] == 1 else j
            var kj = 0 if K.shape[1] == 1 else j
            for c in range(width):
                out.unsafe_set((i * cols + j) * width + c,
                    G.unsafe_get((i * G.shape[1] + gj) * width + c)
                    * K.unsafe_get(i * K.shape[1] + kj))
    G = out^

def concat_gradients(G1: Dense64, G2: Dense64) raises -> Dense64:
    """Stack gradients along the last axis (numpy.dstack)."""
    if G1.shape[0] != G2.shape[0] or G1.shape[1] != G2.shape[1]:
        raise Error("ValueError: kernel gradient shapes must match for concatenation")
    var n = G1.shape[0]
    var m = G1.shape[1]
    var p1 = G1.shape[2]
    var p2 = G2.shape[2]
    var p = p1 + p2
    var out = zeros[DType.float64](Shape(n, m, p))
    var o = out.unsafe_ptr()
    var g1 = G1.unsafe_ptr()
    var g2 = G2.unsafe_ptr()
    for e in range(n * m):
        for k in range(p1):
            o[unsafe_offset = e * p + k] = g1[unsafe_offset = e * p1 + k]
        for k in range(p2):
            o[unsafe_offset = e * p + p1 + k] = g2[unsafe_offset = e * p2 + k]
    return out^


def _round_half_even(x: Float64) -> Float64:
    var f = Float64(Int(x)) if x >= 0 else -Float64(Int(-x))
    var r = x - f
    if abs(r) == 0.5:
        if Int(f) % 2 == 0:
            return f
        return f + (1.0 if x > 0 else -1.0)
    if r > 0.5:
        return f + 1.0
    if r < -0.5:
        return f - 1.0
    return f


def format_g3(x: Float64) -> String:
    """Python `'{:.3g}'.format(x)` for finite and special values."""
    if isnan(x):
        return "nan"
    if not isfinite(x):
        return "inf" if x > 0 else "-inf"
    if x == 0.0:
        return "0" if not (1.0 / x < 0) else "-0"
    var sign = String("-") if x < 0 else String("")
    var v = abs(x)
    var exponent = Int(flog(v) / flog(10.0))
    if Float64(10.0) ** exponent > v:
        exponent -= 1
    if Float64(10.0) ** (exponent + 1) <= v:
        exponent += 1
    # Three significant digits: integer mantissa in [100, 999].
    var mantissa = _round_half_even(v / Float64(10.0) ** (exponent - 2))
    if mantissa >= 1000.0:
        mantissa = mantissa / 10.0
        exponent += 1
    var m = Int(mantissa)
    if exponent < -4 or exponent >= 3:
        var body = String(m // 100)
        var tail = m % 100
        if tail != 0:
            body += "." + (String(tail // 10) if tail % 10 == 0 else (String(tail) if tail >= 10 else "0" + String(tail)))
        var e_sign = String("+") if exponent >= 0 else String("-")
        var e_abs = abs(exponent)
        var e_text = String(e_abs) if e_abs >= 10 else "0" + String(e_abs)
        return sign + body + "e" + e_sign + e_text
    # Fixed notation with 2 - exponent decimals.
    var decimals = 2 - exponent
    if decimals <= 0:
        return sign + String(m * (10 ** (-decimals)))
    var power = 10**decimals
    var whole = m // power
    var frac = m % power
    var width = decimals
    while frac != 0 and frac % 10 == 0:
        frac = frac // 10
        width -= 1
    if frac == 0:
        return sign + String(whole)
    var frac_text = String(frac)
    while frac_text.byte_length() < width:
        frac_text = "0" + frac_text
    return sign + String(whole) + "." + frac_text
