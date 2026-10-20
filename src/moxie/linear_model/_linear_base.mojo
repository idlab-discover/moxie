# Adapted from scikit-learn sklearn/linear_model/_base.py; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense implementation of the shared linear-model plumbing from sklearn `_base.py`.

One family owner for: weighted centering (`_preprocess_data`), sqrt sample-weight
rescaling (`_rescale_data`), intercept reconstruction (`LinearModel._set_intercept`),
canonical dense prediction (`LinearModel._decision_function`), target/coefficient
shape conversion and the regressor R2 score. Least-squares, ridge, Bayesian, OMP
and LARS children consume these; they must not copy them.

Canonical internal layouts (see docs/plumbing.md): X (samples, features); targets
(samples, outputs); coefficients (outputs, features); intercepts (outputs,).
All returned arrays are owned, compact C-order copies; inputs are never mutated.
"""
from numojo import NDArray, Shape, zeros, ones
from std.math import isfinite, sqrt, nan
from moxie.linear_model._contracts import (
    validate_xy,
    validate_weights,
    PreparedData,
    DensePreprocessor,
    DenseLinearPrediction,
)


@always_inline
def _entry[dtype: DType](a: NDArray[dtype], i: Int, j: Int) raises -> Scalar[dtype]:
    """Logical (i, j) element of a valid rank-2 array (strides are element counts)."""
    return a.unsafe_get(i * Int(a.strides[0]) + j * Int(a.strides[1]))


@always_inline
def _vector_entry[dtype: DType](a: NDArray[dtype], i: Int) raises -> Scalar[dtype]:
    return a.unsafe_get(i * Int(a.strides[0]))


def check_float_dtype[dtype: DType]():
    comptime assert dtype == DType.float32 or dtype == DType.float64, "linear_model supports Float32 and Float64"


def n_targets[dtype: DType](y: NDArray[dtype]) raises -> Int:
    """Number of outputs of a rank-1 (one) or rank-2 target."""
    return y.shape[1] if y.ndim == 2 else 1


def check_finite[dtype: DType](a: NDArray[dtype], name: String, owner: String) raises:
    """Reject NaN/infinity like sklearn `check_array(force_all_finite=True)`."""
    check_float_dtype[dtype]()
    if a.ndim == 1:
        for i in range(a.shape[0]):
            if not isfinite(_vector_entry(a, i)):
                raise Error(String("ValueError: ", owner, ": input ", name, " contains NaN or infinity at index ", i))
    elif a.ndim == 2:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if not isfinite(_entry(a, i, j)):
                    raise Error(String("ValueError: ", owner, ": input ", name, " contains NaN or infinity at [", i, ", ", j, "]"))
    else:
        raise Error(String("ValueError: ", owner, ": ", name, " must have rank one or two, got rank ", a.ndim))


def compact_matrix[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    """Owned C-order copy of the logical entries of a (possibly strided) matrix."""
    check_float_dtype[dtype]()
    if a.ndim != 2:
        raise Error(String("ValueError: expected a matrix, got rank ", a.ndim))
    var rows = a.shape[0]
    var cols = a.shape[1]
    var result = zeros[dtype](Shape(rows, cols))
    for i in range(rows):
        for j in range(cols):
            result.unsafe_set(i * cols + j, _entry(a, i, j))
    return result^


def compact_vector[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    """Owned contiguous copy of the logical entries of a (possibly strided) vector."""
    check_float_dtype[dtype]()
    if a.ndim != 1:
        raise Error(String("ValueError: expected a vector, got rank ", a.ndim))
    var result = zeros[dtype](Shape(a.shape[0]))
    for i in range(a.shape[0]):
        result.unsafe_set(i, _vector_entry(a, i))
    return result^


def target_matrix[dtype: DType](y: NDArray[dtype]) raises -> NDArray[dtype]:
    """Canonical (samples, outputs) owned copy; a rank-1 target becomes one column."""
    check_float_dtype[dtype]()
    if y.ndim == 1:
        var result = zeros[dtype](Shape(y.shape[0], 1))
        for i in range(y.shape[0]):
            result.unsafe_set(i, _vector_entry(y, i))
        return result^
    if y.ndim == 2:
        return compact_matrix(y)
    raise Error(String("ValueError: y must have rank one or two, got rank ", y.ndim))


def weighted_column_mean[dtype: DType](a: NDArray[dtype], sample_weight: Optional[NDArray[dtype]]) raises -> NDArray[dtype]:
    """Column means (np.average(axis=0, weights=...)) with Float64 accumulation."""
    check_float_dtype[dtype]()
    var rows = a.shape[0]
    var cols = a.shape[1]
    var result = zeros[dtype](Shape(cols))
    var mass = Float64(0)
    if sample_weight:
        for i in range(rows):
            mass += Float64(_vector_entry(sample_weight.value(), i))
    else:
        mass = Float64(rows)
    if mass == 0:
        raise Error("ValueError: weights sum to zero, cannot compute a weighted average")
    for j in range(cols):
        var total = Float64(0)
        for i in range(rows):
            var value = Float64(_entry(a, i, j))
            if sample_weight:
                value *= Float64(_vector_entry(sample_weight.value(), i))
            total += value
        result.unsafe_set(j, (total / mass).cast[dtype]())
    return result^


def scalar_weights[dtype: DType](X: NDArray[dtype], weight: Float64) raises -> NDArray[dtype]:
    """Expand a scalar fit weight at the estimator boundary."""
    if X.ndim != 2:
        raise Error("ValueError: X must be a matrix")
    if not isfinite(weight) or weight < 0:
        raise Error("ValueError: sample_weight must be finite and nonnegative")
    var result = zeros[dtype](Shape(X.shape[0]))
    for i in range(X.shape[0]):
        result.unsafe_set(i, Scalar[dtype](weight))
    return result^


def preprocess_data[
    dtype: DType
](
    X: NDArray[dtype],
    y: NDArray[dtype],
    sample_weight: Optional[NDArray[dtype]],
    fit_intercept: Bool,
    rescale_with_sw: Bool = True,
    owner: String = "linear_model",
) raises -> PreparedData[dtype]:
    """Dense sklearn `_preprocess_data` followed by optional `_rescale_data`.

    With `fit_intercept`, X and y are centered by their (weighted) column means,
    stored as X_offset/y_offset; otherwise offsets are zero. X_scale is ones. When
    sample weights are given and `rescale_with_sw` is True, the centered rows are
    multiplied by sqrt(sample_weight) and `sample_weight_sqrt` is returned.
    The target is returned in canonical (samples, outputs) layout; `vector_target`
    records whether the caller supplied a rank-1 target. Inputs are unchanged.
    Rescaling requires nonnegative weights; without rescaling signed finite
    weights are accepted (only the centering uses them).
    """
    check_float_dtype[dtype]()
    validate_xy(X, y)
    check_finite(X, "X", owner)
    check_finite(y, "y", owner)
    var n_samples = X.shape[0]
    var n_features = X.shape[1]
    if sample_weight:
        validate_weights(sample_weight.value(), n_samples, nonnegative=rescale_with_sw)
    var Xc = compact_matrix(X)
    var yc = target_matrix(y)
    var outputs = yc.shape[1]
    var X_offset = zeros[dtype](Shape(n_features))
    var y_offset = zeros[dtype](Shape(outputs))
    if fit_intercept:
        X_offset = weighted_column_mean(Xc, sample_weight)
        y_offset = weighted_column_mean(yc, sample_weight)
        for i in range(n_samples):
            for j in range(n_features):
                var k = i * n_features + j
                Xc.unsafe_set(k, Xc.unsafe_get(k) - X_offset.unsafe_get(j))
            for j in range(outputs):
                var k = i * outputs + j
                yc.unsafe_set(k, yc.unsafe_get(k) - y_offset.unsafe_get(j))
    var X_scale = ones[dtype](Shape(n_features))
    var sw_sqrt = Optional[NDArray[dtype]](None)
    if sample_weight and rescale_with_sw:
        var root = zeros[dtype](Shape(n_samples))
        for i in range(n_samples):
            root.unsafe_set(i, sqrt(_vector_entry(sample_weight.value(), i)))
        for i in range(n_samples):
            var s = root.unsafe_get(i)
            for j in range(n_features):
                var k = i * n_features + j
                Xc.unsafe_set(k, Xc.unsafe_get(k) * s)
            for j in range(outputs):
                var k = i * outputs + j
                yc.unsafe_set(k, yc.unsafe_get(k) * s)
        sw_sqrt = root^
    return PreparedData[dtype](Xc^, yc^, X_offset^, y_offset^, X_scale^, sw_sqrt^, y.ndim == 1)


def set_intercept[
    dtype: DType
](
    mut coef: NDArray[dtype],
    X_offset: NDArray[dtype],
    y_offset: NDArray[dtype],
    X_scale: NDArray[dtype],
    fit_intercept: Bool,
) raises -> NDArray[dtype]:
    """sklearn `_set_intercept` on canonical (outputs, features) coefficients.

    With `fit_intercept`, coef is first divided by X_scale in place and the
    returned intercept is y_offset - coef @ X_offset (Float64 accumulation).
    Otherwise the intercept is zero. Returns shape (outputs,).
    """
    check_float_dtype[dtype]()
    if coef.ndim != 2 or X_offset.ndim != 1 or y_offset.ndim != 1 or X_scale.ndim != 1:
        raise Error("ValueError: set_intercept expects coef (outputs, features) and rank-1 offsets")
    var outputs = coef.shape[0]
    var features = coef.shape[1]
    if X_offset.shape[0] != features or X_scale.shape[0] != features or y_offset.shape[0] != outputs:
        raise Error(String("ValueError: set_intercept shape mismatch: coef (", outputs, ", ", features, "), X_offset (", X_offset.shape[0], ",), y_offset (", y_offset.shape[0], ",)"))
    var intercept = zeros[dtype](Shape(outputs))
    if not fit_intercept:
        return intercept^
    var row = Int(coef.strides[0])
    var col = Int(coef.strides[1])
    for k in range(outputs):
        var total = Float64(0)
        for j in range(features):
            var index = k * row + j * col
            var value = coef.unsafe_get(index) / _vector_entry(X_scale, j)
            coef.unsafe_set(index, value)
            total += Float64(value) * Float64(_vector_entry(X_offset, j))
        intercept.unsafe_set(k, (Float64(_vector_entry(y_offset, k)) - total).cast[dtype]())
    return intercept^


def linear_decision[
    dtype: DType
](X: NDArray[dtype], coef: NDArray[dtype], intercept: NDArray[dtype]) raises -> NDArray[dtype]:
    """Canonical dense decision X @ coef.T + intercept, shape (samples, outputs).

    coef is (outputs, features) and intercept (outputs,). Accumulates in Float64.
    """
    check_float_dtype[dtype]()
    if X.ndim != 2 or coef.ndim != 2 or intercept.ndim != 1:
        raise Error("ValueError: decision expects X (samples, features), coef (outputs, features), intercept (outputs,)")
    var n = X.shape[0]
    var p = X.shape[1]
    var k = coef.shape[0]
    if coef.shape[1] != p or intercept.shape[0] != k:
        raise Error(String("ValueError: X has ", p, " features but coefficients have shape (", coef.shape[0], ", ", coef.shape[1], ") and intercept ", intercept.shape[0], " entries"))
    var Xc = compact_matrix(X)
    var C = compact_matrix(coef)
    var result = zeros[dtype](Shape(n, k))
    for i in range(n):
        for t in range(k):
            var total = Float64(_vector_entry(intercept, t))
            for j in range(p):
                total += Float64(Xc.unsafe_get(i * p + j)) * Float64(C.unsafe_get(t * p + j))
            result.unsafe_set(i * k + t, total.cast[dtype]())
    return result^


def gram_matrix[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]:
    """X.T @ X, shape (features, features), with Float64 accumulation.

    Streams rows of the (possibly strided) input once; the result is exactly
    symmetric. Shared by the ridge (cholesky), Bayesian, OMP and LARS children.
    """
    check_float_dtype[dtype]()
    var Xc = compact_matrix(X)
    var n = Xc.shape[0]
    var p = Xc.shape[1]
    var acc = List[Float64](length=p * p, fill=0)
    for i in range(n):
        var row = i * p
        for j in range(p):
            var xij = Float64(Xc.unsafe_get(row + j))
            if xij == 0:
                continue
            for l in range(j, p):
                acc[j * p + l] += xij * Float64(Xc.unsafe_get(row + l))
    var result = zeros[dtype](Shape(p, p))
    for j in range(p):
        for l in range(j, p):
            var value = acc[j * p + l].cast[dtype]()
            result.unsafe_set(j * p + l, value)
            result.unsafe_set(l * p + j, value)
    return result^


def outer_gram[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]:
    """X @ X.T, shape (samples, samples), with Float64 accumulation (kernel form)."""
    check_float_dtype[dtype]()
    var Xc = compact_matrix(X)
    var n = Xc.shape[0]
    var p = Xc.shape[1]
    var result = zeros[dtype](Shape(n, n))
    for i in range(n):
        for k in range(i, n):
            var total = Float64(0)
            for j in range(p):
                total += Float64(Xc.unsafe_get(i * p + j)) * Float64(Xc.unsafe_get(k * p + j))
            var value = total.cast[dtype]()
            result.unsafe_set(i * n + k, value)
            result.unsafe_set(k * n + i, value)
    return result^


def transpose_product[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype]) raises -> NDArray[dtype]:
    """X.T @ Y for X (samples, features) and Y (samples, k) or (samples,).

    Returns (features, k) for a matrix Y and (features,) for a vector Y, with
    Float64 accumulation.
    """
    check_float_dtype[dtype]()
    var Xc = compact_matrix(X)
    var Yc = target_matrix(Y)
    var n = Xc.shape[0]
    var p = Xc.shape[1]
    var k = Yc.shape[1]
    if Yc.shape[0] != n:
        raise Error(String("ValueError: transpose_product: X has ", n, " rows but Y has ", Yc.shape[0]))
    var acc = List[Float64](length=p * k, fill=0)
    for i in range(n):
        for j in range(p):
            var xij = Float64(Xc.unsafe_get(i * p + j))
            for t in range(k):
                acc[j * k + t] += xij * Float64(Yc.unsafe_get(i * k + t))
    var result = zeros[dtype](Shape(p, k) if Y.ndim == 2 else Shape(p))
    for i in range(p * k):
        result.unsafe_set(i, acc[i].cast[dtype]())
    return result^


def public_coef[dtype: DType](coef: NDArray[dtype], vector_target: Bool) raises -> NDArray[dtype]:
    """Public `coef_`: (features,) for a rank-1 target, else (outputs, features)."""
    var C = compact_matrix(coef)
    if vector_target:
        if C.shape[0] != 1:
            raise Error("ValueError: vector target requires one output row")
        return C.reshape(Shape(C.shape[1]))
    return C^


def canonical_coef[dtype: DType](coef: NDArray[dtype]) raises -> NDArray[dtype]:
    """Inverse of `public_coef`: always (outputs, features)."""
    if coef.ndim == 1:
        var result = zeros[dtype](Shape(1, coef.shape[0]))
        for j in range(coef.shape[0]):
            result.unsafe_set(j, _vector_entry(coef, j))
        return result^
    return compact_matrix(coef)


def public_prediction[dtype: DType](decision: NDArray[dtype], vector_target: Bool) raises -> NDArray[dtype]:
    """(samples,) for a rank-1 fitted target, else the (samples, outputs) matrix."""
    if vector_target:
        if decision.shape[1] != 1:
            raise Error("ValueError: vector target requires one output column")
        return decision.reshape(Shape(decision.shape[0]))
    return decision.copy()


def check_predict_input[dtype: DType](X: NDArray[dtype], n_features_in: Int, owner: String) raises:
    """Validate inference input: dense matrix, finite, fitted feature count."""
    if X.ndim != 2:
        raise Error(String("ValueError: ", owner, ": expected a 2-D array X, got rank ", X.ndim))
    if X.shape[0] == 0:
        raise Error(String("ValueError: ", owner, ": X has 0 samples; a minimum of 1 is required"))
    if X.shape[1] != n_features_in:
        raise Error(String("ValueError: ", owner, ": X has ", X.shape[1], " features, but it is expecting ", n_features_in, " features as input"))
    check_finite(X, "X", owner)


def r2_score[
    dtype: DType
](y_true: NDArray[dtype], y_pred: NDArray[dtype], sample_weight: Optional[NDArray[dtype]] = None) raises -> Float64:
    """sklearn `r2_score` with multioutput='uniform_average' and force_finite=True.

    Returns NaN for fewer than two samples (sklearn warns and returns NaN).
    """
    check_float_dtype[dtype]()
    check_finite(y_true, "y_true", "score")
    check_finite(y_pred, "y_pred", "score")
    var yt = target_matrix(y_true)
    var yp = target_matrix(y_pred)
    if yt.shape[0] != yp.shape[0] or yt.shape[1] != yp.shape[1]:
        raise Error(String("ValueError: y_true shape (", yt.shape[0], ", ", yt.shape[1], ") does not match y_pred (", yp.shape[0], ", ", yp.shape[1], ")"))
    var n = yt.shape[0]
    var k = yt.shape[1]
    if n == 0 or k == 0:
        raise Error("ValueError: score requires nonempty targets")
    if sample_weight:
        validate_weights(sample_weight.value(), n, nonnegative=False)
    if n < 2:
        return nan[DType.float64]()
    var weights = List[Float64](capacity=n)
    var mass = Float64(0)
    for i in range(n):
        var w = Float64(_vector_entry(sample_weight.value(), i)) if sample_weight else 1.0
        weights.append(w)
        mass += w
    if mass == 0:
        raise Error("ValueError: sample weights sum to zero")
    var total = Float64(0)
    for t in range(k):
        var mean = Float64(0)
        for i in range(n):
            mean += weights[i] * Float64(yt.unsafe_get(i * k + t))
        mean /= mass
        var numerator = Float64(0)
        var denominator = Float64(0)
        for i in range(n):
            var value = Float64(yt.unsafe_get(i * k + t))
            var residual = value - Float64(yp.unsafe_get(i * k + t))
            numerator += weights[i] * residual * residual
            denominator += weights[i] * (value - mean) * (value - mean)
        if denominator != 0:
            total += 1.0 - numerator / denominator
        elif numerator == 0:
            total += 1.0
    return total / Float64(k)


struct DenseCentering[data_type: DType](DensePreprocessor, Movable):
    """`DensePreprocessor` provider delegating to `preprocess_data`.

    The trait's `weights` argument is always present; pass ones for unweighted
    data or call `preprocess_data` with `None` to reproduce sklearn's no-weight
    branch (no rescaling, `sample_weight_sqrt=None`).
    """

    comptime dtype = Self.data_type

    def __init__(out self):
        pass

    def prepare(self, X: NDArray[Self.dtype], y: NDArray[Self.dtype], weights: NDArray[Self.dtype], fit_intercept: Bool, rescale_with_sw: Bool) raises -> PreparedData[Self.dtype]:
        return preprocess_data(X, y, Optional(weights.copy()), fit_intercept, rescale_with_sw)


struct DenseDecision[data_type: DType](DenseLinearPrediction, Movable):
    """`DenseLinearPrediction` provider delegating to `linear_decision`."""

    comptime dtype = Self.data_type

    def __init__(out self):
        pass

    def decision(self, X: NDArray[Self.dtype], coef: NDArray[Self.dtype], intercept: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        return linear_decision(X, coef, intercept)
