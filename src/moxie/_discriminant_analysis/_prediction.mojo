# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Row-wise score post-processing shared by LDA and QDA.

Implements `sklearn.utils.extmath.softmax`, `scipy.special.expit` usage in
LinearDiscriminantAnalysis.predict_proba, the log-sum-exp normalization of
DiscriminantAnalysisPredictionMixin and ClassifierMixin.score (accuracy).
"""
from std.math import exp, log, isfinite, isnan
from numojo import NDArray, Shape, zeros
from ._work import Work, get, put, matrix, vector
from ._labels import ClassLabel


def row_argmax(scores: Work) raises -> List[Int]:
    """Index of the first maximum in each row (like `np.argmax(axis=1)`)."""
    var n = scores.shape[0]
    var k = scores.shape[1]
    var out = List[Int](capacity=n)
    for i in range(n):
        var best = 0
        var best_value = get(scores, i, 0)
        for j in range(1, k):
            var value = get(scores, i, j)
            # NaN never wins, matching NumPy only for finite rows (NaN rows are
            # rejected upstream because inputs must be finite).
            if value > best_value:
                best = j
                best_value = value
        out.append(best)
    return out^


def softmax_rows(scores: Work) raises -> Work:
    """Numerically stable row softmax: exp(s - max) / sum(exp(s - max))."""
    var n = scores.shape[0]
    var k = scores.shape[1]
    var out = matrix(n, k)
    for i in range(n):
        var largest = get(scores, i, 0)
        for j in range(1, k):
            largest = max(largest, get(scores, i, j))
        var total: Float64 = 0
        for j in range(k):
            var e = exp(get(scores, i, j) - largest)
            put(out, i, j, e)
            total += e
        for j in range(k):
            put(out, i, j, get(out, i, j) / total)
    return out^


def log_softmax_rows(scores: Work) raises -> Work:
    """Row log-posterior: (s - max) - log(sum(exp(s - max)))."""
    var n = scores.shape[0]
    var k = scores.shape[1]
    var out = matrix(n, k)
    for i in range(n):
        var largest = get(scores, i, 0)
        for j in range(1, k):
            largest = max(largest, get(scores, i, j))
        var total: Float64 = 0
        for j in range(k):
            total += exp(get(scores, i, j) - largest)
        var log_total = log(total)
        for j in range(k):
            put(out, i, j, (get(scores, i, j) - largest) - log_total)
    return out^


def expit(x: Float64) -> Float64:
    """Logistic sigmoid 1 / (1 + exp(-x)), evaluated without overflow."""
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    var e = exp(x)
    return e / (1.0 + e)


def binary_probabilities(decision: Work) raises -> Work:
    """Stack [1 - expit(d), expit(d)] for a (n,) binary decision vector."""
    var n = decision.shape[0]
    var out = matrix(n, 2)
    for i in range(n):
        var p = expit(decision.unsafe_get(i))
        put(out, i, 0, 1.0 - p)
        put(out, i, 1, p)
    return out^


def exp_array[dtype: DType](log_proba: Work) raises -> NDArray[dtype]:
    """Elementwise exp of a work matrix, cast to the estimator dtype."""
    var out = zeros[dtype](Shape(log_proba.shape[0], log_proba.shape[1]))
    for i in range(log_proba.size):
        out.unsafe_set(i, SIMD[dtype, 1](exp(log_proba.unsafe_get(i))))
    return out^


def log_of_probabilities[dtype: DType](proba: Work) raises -> NDArray[dtype]:
    """Return log(proba) with exact zeros replaced by the dtype's smallest normal (sklearn LDA).

    Round to the public probability dtype before testing for zero, including
    probabilities that underflow when converted from the Float64 work array.
    """
    var smallest: Float64
    comptime if dtype == DType.float32:
        smallest = 1.1754943508222875e-38
    else:
        smallest = 2.2250738585072014e-308
    var out = zeros[dtype](Shape(proba.shape[0], proba.shape[1]))
    for i in range(proba.size):
        var value = Float64(SIMD[dtype, 1](proba.unsafe_get(i)))
        if value == 0.0:
            value += smallest
        out.unsafe_set(i, SIMD[dtype, 1](log(value)))
    return out^


def select_labels[L: ClassLabel](classes: List[L], indices: List[Int]) -> List[L]:
    """Return classes_.take(indices)."""
    var out = List[L](capacity=len(indices))
    for i in range(len(indices)):
        out.append(classes[indices[i]].copy())
    return out^


def accuracy[
    L: ClassLabel, dtype: DType
](
    predicted: List[L], y: List[L], sample_weight: Optional[NDArray[dtype]], estimator: String
) raises -> Float64:
    """ClassifierMixin.score: (weighted) fraction of exact label matches."""
    var n = len(y)
    if len(predicted) != n:
        raise Error(
            String("ValueError: Found input variables with inconsistent numbers of samples: [")
            + String(len(predicted), ", ", n, "] in ", estimator, ".score")
        )
    if not sample_weight:
        var hits = 0
        for i in range(n):
            if predicted[i] == y[i]:
                hits += 1
        return Float64(hits) / Float64(n)
    ref w = sample_weight.value()
    if w.ndim != 1 or w.shape[0] != n:
        raise Error(
            String("ValueError: sample_weight.shape == (")
            + String(w.shape[0] if w.ndim >= 1 else 0)
            + String(",), expected (", n, ",) in ", estimator, ".score")
        )
    var stride = Int(w.strides[0])
    var total: Float64 = 0
    var matched: Float64 = 0
    var any_nonzero = False
    for i in range(n):
        var weight = Float64(w.unsafe_get(i * stride))
        if isnan(weight):
            raise Error("ValueError: Input sample_weight contains NaN.")
        if not isfinite(weight):
            raise Error("ValueError: Input sample_weight contains infinity or a value too large.")
        any_nonzero = any_nonzero or weight != 0
        total += weight
        if predicted[i] == y[i]:
            matched += weight
    if not any_nonzero:
        raise Error("ValueError: Sample weights must contain at least one non-zero number.")
    if total == 0:
        raise Error("ZeroDivisionError: Weights sum to zero, can't be normalized")
    return matched / total
