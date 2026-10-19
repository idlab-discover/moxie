"""Row-wise stable logsumexp and log responsibilities (shared by both mixtures).

Formula follows sklearn.utils._array_api._logsumexp (pinned revision): with
row maximum M and multiplicity m, s = sum_{a != M} exp(a - shift) where
shift = M if finite else 0, s = s / m unless zero, result = log1p(s) + log(m) + M.
Raw logits are never exponentiated without the shift. An all -inf row yields
-inf and NaN responsibilities, exactly as the reference does; NaN propagates.
Arithmetic is Float64; results are stored in the array dtype.
"""
from numojo import NDArray, Shape, zeros
from std.math import isfinite, isnan
from std.utils.numerics import nan
from moxie.mixture._contracts import ResponsibilityEngine
from moxie.mixture._scalar_math import log64, exp64, log1p64


def _check_matrix[dtype: DType](x: NDArray[dtype], operation: String) raises:
    if x.ndim != 2:
        raise Error(String("ValueError: ", operation, " expects an (n_samples, n_components) matrix, got rank ", x.ndim))


@always_inline
def _row_logsumexp[dtype: DType](x: NDArray[dtype], row: Int, columns: Int, s0: Int, s1: Int) -> Float64:
    var base = row * s0
    var maximum = Float64(x.unsafe_get(base))
    var has_nan = False
    for j in range(columns):
        var value = Float64(x.unsafe_get(base + j * s1))
        if isnan(value):
            has_nan = True
        elif value > maximum or isnan(maximum):
            maximum = value
    if has_nan:
        return nan[DType.float64]()
    var shift = maximum if isfinite(maximum) else 0.0
    var multiplicity = 0.0
    var total = 0.0
    for j in range(columns):
        var value = Float64(x.unsafe_get(base + j * s1))
        if value == maximum:
            multiplicity += 1.0
        else:
            total += exp64(value - shift)
    if total != 0.0:
        total /= multiplicity
    return log1p64(total) + log64(multiplicity) + maximum


def logsumexp_rows[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Return the (n,) row-wise logsumexp of an (n, k) matrix (any strides)."""
    _check_matrix(x, "logsumexp_rows")
    var n = x.shape[0]
    var k = x.shape[1]
    var out = zeros[dtype](Shape(n))
    if k == 0:
        raise Error("ValueError: logsumexp_rows requires at least one component column")
    var s0 = Int(x.strides[0])
    var s1 = Int(x.strides[1])
    for i in range(n):
        out.unsafe_set(i, _row_logsumexp(x, i, k, s0, s1).cast[dtype]())
    return out^


def log_responsibilities[dtype: DType](weighted_log_prob: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Return (log_prob_norm (n,), log_resp (n,k)) from weighted log probabilities.

    log_resp = weighted - log_prob_norm[:, None], computed from the dtype-rounded
    norm like the reference.
    """
    _check_matrix(weighted_log_prob, "log_responsibilities")
    var norm = logsumexp_rows(weighted_log_prob)
    var n = weighted_log_prob.shape[0]
    var k = weighted_log_prob.shape[1]
    var s0 = Int(weighted_log_prob.strides[0])
    var s1 = Int(weighted_log_prob.strides[1])
    var log_resp = zeros[dtype](Shape(n, k))
    for i in range(n):
        var row_norm = norm.unsafe_get(i)
        for j in range(k):
            log_resp.unsafe_set(i * k + j, weighted_log_prob.unsafe_get(i * s0 + j * s1) - row_norm)
    return (norm^, log_resp^)


def exp_array[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Elementwise exp of a matrix (responsibilities from log responsibilities)."""
    _check_matrix(x, "exp_array")
    var n = x.shape[0]
    var k = x.shape[1]
    var s0 = Int(x.strides[0])
    var s1 = Int(x.strides[1])
    var out = zeros[dtype](Shape(n, k))
    for i in range(n):
        for j in range(k):
            out.unsafe_set(i * k + j, exp64(Float64(x.unsafe_get(i * s0 + j * s1))).cast[dtype]())
    return out^


def mean_vector[dtype: DType](x: NDArray[dtype]) raises -> Float64:
    """Float64-accumulated mean of a vector, rounded through the dtype."""
    if x.ndim != 1 or x.shape[0] == 0:
        raise Error("ValueError: mean_vector expects a nonempty vector")
    var total = 0.0
    var s = Int(x.strides[0])
    for i in range(x.shape[0]):
        total += Float64(x.unsafe_get(i * s))
    return Float64((total / Float64(x.shape[0])).cast[dtype]())


def argmax_rows[dtype: DType](x: NDArray[dtype]) raises -> NDArray[DType.int64]:
    """First index of the row maximum (NumPy argmax; NaN wins as in NumPy)."""
    _check_matrix(x, "argmax_rows")
    var n = x.shape[0]
    var k = x.shape[1]
    var s0 = Int(x.strides[0])
    var s1 = Int(x.strides[1])
    var labels = zeros[DType.int64](Shape(n))
    for i in range(n):
        var best = 0
        var best_value = Float64(x.unsafe_get(i * s0))
        if not isnan(best_value):
            for j in range(1, k):
                var value = Float64(x.unsafe_get(i * s0 + j * s1))
                if isnan(value):
                    best = j
                    break
                if value > best_value:
                    best = j
                    best_value = value
        labels.unsafe_set(i, Int64(best))
    return labels^


@fieldwise_init
struct StableResponsibilities[dtype_: DType](ResponsibilityEngine, Copyable):
    """ResponsibilityEngine implementation using the stable logsumexp above."""

    comptime dtype = Self.dtype_

    def normalize(self, weighted_log_prob: NDArray[Self.dtype]) raises -> Tuple[NDArray[Self.dtype], NDArray[Self.dtype]]:
        return log_responsibilities(weighted_log_prob)
