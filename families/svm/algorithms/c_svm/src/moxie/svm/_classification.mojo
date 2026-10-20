# Adapted from scikit-learn; BSD-3-Clause, see SCIKIT_LEARN_LICENSE.
"""Shared kernel estimator configuration and class presentation."""
import numojo as nm
from std.math import abs, isfinite
from moxie.svm._contracts import Dense, Indices, KernelModel, KernelSpec, TrainingSnapshot, validate_matrix
from moxie.svm._kernels import resolve_gamma

struct Gamma(Copyable, Movable):
    var mode: String
    var value: Float64
    var numeric: Bool

    @implicit
    def __init__(out self, mode: String):
        self.mode = mode
        self.value = 0
        self.numeric = False

    @implicit
    def __init__(out self, value: Float64):
        self.mode = "value"
        self.value = value
        self.numeric = True

    def resolve(self, X: Dense, kernel: String) raises -> Float64:
        if not self.numeric and self.mode != "scale" and self.mode != "auto":
            raise Error("ValueError: gamma must be scale, auto, or nonnegative")
        if self.numeric and (not isfinite(self.value) or self.value < 0):
            raise Error("ValueError: gamma must be finite and nonnegative")
        if kernel == "precomputed" or kernel == "callable":
            return 0
        if self.numeric:
            return self.value
        return resolve_gamma(X, self.mode)

struct Probability(Copyable, Movable):
    var mode: String
    var boolean: Bool

    @implicit
    def __init__(out self, mode: String):
        self.mode = mode
        self.boolean = False

    @implicit
    def __init__(out self, enabled: Bool):
        self.mode = "true" if enabled else "false"
        self.boolean = True

    def enabled(self) raises -> Bool:
        if not self.boolean and self.mode != "deprecated":
            raise Error("ValueError: probability must be Bool or deprecated")
        return self.mode == "true"


def class_scores(raw: Dense, classes: Int, confidence: Bool) raises -> Dense:
    var votes = nm.zeros[DType.float64](nm.Shape(raw.shape[0], classes))
    var sums = nm.zeros[DType.float64](nm.Shape(raw.shape[0], classes))
    for row in range(raw.shape[0]):
        var pair = 0
        for first in range(classes):
            for second in range(first + 1, classes):
                var value = raw.unsafe_get(row * Int(raw.strides[0]) + pair * Int(raw.strides[1]))
                # LIBSVM predicts the second class on an exact zero; sklearn's
                # OVR decision transform assigns that tie to the first class.
                var winner = first if (value >= 0 if confidence else value > 0) else second
                var index = row * classes + winner
                votes.unsafe_set(index, votes.unsafe_get(index) + 1)
                sums.unsafe_set(row * classes + first, sums.unsafe_get(row * classes + first) + value)
                sums.unsafe_set(row * classes + second, sums.unsafe_get(row * classes + second) - value)
                pair += 1
    if confidence:
        for i in range(votes.size):
            var value = sums.unsafe_get(i)
            votes.unsafe_set(i, votes.unsafe_get(i) + value / (3 * (abs(value) + 1)))
    return votes^


def support_counts(model: KernelModel) raises -> nm.NDArray[DType.int32]:
    var result = nm.zeros[DType.int32](nm.Shape(model.n_support.shape[0]))
    for i in range(result.size):
        var value = model.n_support.unsafe_get(i)
        if value < 0 or value > 2147483647:
            raise Error("ValueError: support count exceeds int32")
        result.unsafe_set(i, Int32(value))
    return result^


def linear_coefficients(model: KernelModel, kernel: KernelSpec, classes: Int) raises -> Dense:
    if kernel.kind != "linear":
        raise Error("AttributeError: coef_ is available only for linear kernels")
    var supports = model.support.shape[0]
    if classes <= 2:
        var coefficients = model.dual_coef.copy()
        if classes == 2:
            coefficients = coefficients * Float64(-1)
        return coefficients @ model.support_vectors
    var pairs = classes * (classes - 1) // 2
    var coefficients = nm.zeros[DType.float64](nm.Shape(pairs, supports))
    var pair = 0
    var first_start = 0
    for first in range(classes):
        var first_end = first_start + Int(model.n_support.unsafe_get(first))
        var second_start = first_end
        for second in range(first + 1, classes):
            var second_end = second_start + Int(model.n_support.unsafe_get(second))
            for s in range(first_start, first_end):
                coefficients.unsafe_set(pair * supports + s, model.dual_coef.unsafe_get((second - 1) * supports + s))
            for s in range(second_start, second_end):
                coefficients.unsafe_set(pair * supports + s, model.dual_coef.unsafe_get(first * supports + s))
            second_start = second_end
            pair += 1
        first_start = first_end
    return coefficients @ model.support_vectors


def public_support(model: KernelModel, weights: Dense) raises -> nm.NDArray[DType.int32]:
    """Pinned sklearn sv_ind counts only rows retained by positive weights."""
    var mapping = nm.zeros[DType.int64](nm.Shape(weights.shape[0]))
    var retained = 0
    for row in range(weights.shape[0]):
        mapping.unsafe_set(row, Int64(retained))
        if weights.unsafe_get(row * Int(weights.strides[0])) > 0:
            retained += 1
    var result = model.support.copy()
    for i in range(result.shape[0]):
        result.unsafe_set(i, mapping.unsafe_get(Int(model.support.unsafe_get(i))))
    return checked_int32(result)


def checked_int32(values: Indices) raises -> nm.NDArray[DType.int32]:
    var result = nm.zeros[DType.int32](nm.Shape(values.shape[0]))
    for i in range(result.size):
        var value = values.unsafe_get(i * Int(values.strides[0]))
        if value < -2147483648 or value > 2147483647:
            raise Error("ValueError: index metadata exceeds int32")
        result.unsafe_set(i, Int32(value))
    return result^
