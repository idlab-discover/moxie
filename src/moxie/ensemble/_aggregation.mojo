# scikit-learn developers, BSD-3-Clause; see LICENSES/scikit-learn.txt.
# Summation order follows NumPy's pairwise reduction; see LICENSES/NumPy.txt.
"""Averaging of member outputs, out-of-bag accumulation and default OOB scores.

Ports sklearn `_forest._accumulate_prediction`, `BaseForest._compute_oob_predictions`,
`BaseForest.feature_importances_`, and the `accuracy_score`/`r2_score` defaults the
forests use for `oob_score_`, keeping NumPy's accumulation and reduction order.
"""
import numojo as nm
from moxie.ensemble._records import Values, Indices, ClassAlignment, OOBAccumulator
from moxie.ensemble._engine_api import EnsembleAggregator
from moxie.ensemble._plumbing import validate_alignment, warn
from moxie.ensemble._sampling import numpy_sum


def _columns(x: Values) raises -> Int:
    if x.ndim == 1:
        return 1
    if x.ndim == 2:
        return x.shape[1]
    raise Error("ValueError: expected a vector or matrix")


def _get(x: Values, row: Int, column: Int) raises -> Float64:
    var offset = row * Int(x.strides[0])
    if x.ndim == 2:
        offset += column * Int(x.strides[1])
    return x.unsafe_get(offset)


def _add(mut sums: Values, row: Int, column: Int, value: Float64) raises:
    var offset = row * Int(sums.strides[0]) + column * Int(sums.strides[1])
    sums.unsafe_set(offset, sums.unsafe_get(offset) + value)


def identity_alignment(n_classes: Int) raises -> ClassAlignment:
    """Members trained on the ensemble's own class codes share its columns."""
    var columns = nm.zeros[DType.int64](nm.Shape(n_classes))
    for c in range(n_classes):
        columns.unsafe_set(c, Int64(c))
    return ClassAlignment(columns^, n_classes)


@fieldwise_init
struct ForestAggregator(EnsembleAggregator):
    """Sums member outputs in member order into caller-owned Float64 matrices."""

    def add_regression(self, predictions: Values, weight: Float64, mut sums: Values) raises:
        if sums.ndim != 2 or predictions.shape[0] != sums.shape[0] or _columns(predictions) != sums.shape[1]:
            raise Error("ValueError: prediction shape does not match the accumulator")
        for i in range(sums.shape[0]):
            for k in range(sums.shape[1]):
                _add(sums, i, k, weight * _get(predictions, i, k))

    def add_probability(self, probabilities: Values, alignment: ClassAlignment, weight: Float64, mut sums: Values) raises:
        if probabilities.ndim != 2 or sums.ndim != 2 or probabilities.shape[0] != sums.shape[0] or alignment.n_classes != sums.shape[1]:
            raise Error("ValueError: probability shape does not match the accumulator")
        validate_alignment(alignment, probabilities.shape[1])
        for i in range(sums.shape[0]):
            for c in range(probabilities.shape[1]):
                _add(sums, i, Int(alignment.columns.unsafe_get(c * Int(alignment.columns.strides[0]))), weight * _get(probabilities, i, c))

    def add_oob(self, rows: Indices, predictions: Values, mut state: OOBAccumulator) raises:
        if predictions.shape[0] != rows.size or _columns(predictions) != state.sums.shape[1]:
            raise Error("ValueError: OOB prediction shape does not match its rows")
        for r in range(rows.size):
            var row = Int(rows.unsafe_get(r * Int(rows.strides[0])))
            if row < 0 or row >= state.sums.shape[0]:
                raise Error("ValueError: OOB row outside the training set")
            for k in range(state.sums.shape[1]):
                _add(state.sums, row, k, _get(predictions, r, k))
            state.counts.unsafe_set(row, state.counts.unsafe_get(row) + 1)

    def normalize(self, sums: Values, denominators: Values) raises -> Values:
        """Divide each row by its denominator; the caller applies its zero policy first."""
        if sums.ndim != 2 or denominators.ndim != 1 or denominators.size != sums.shape[0]:
            raise Error("ValueError: one denominator per row is required")
        var result = nm.zeros[DType.float64](nm.Shape(sums.shape[0], sums.shape[1]))
        for i in range(sums.shape[0]):
            var d = denominators.unsafe_get(i * Int(denominators.strides[0]))
            for k in range(sums.shape[1]):
                result.unsafe_set(i * sums.shape[1] + k, _get(sums, i, k) / d)
        return result^


def new_oob(n_samples: Int, width: Int) raises -> OOBAccumulator:
    return OOBAccumulator(nm.zeros[DType.float64](nm.Shape(n_samples, width)), nm.zeros[DType.int64](nm.Shape(n_samples)))


def forest_oob_denominators(counts: Indices, mut warnings: List[String]) raises -> Values:
    """Forest policy: rows without OOB predictions keep zero sums and divide by one, with a warning."""
    var result = nm.zeros[DType.float64](nm.Shape(counts.size))
    var missing = False
    for i in range(counts.size):
        var count = counts.unsafe_get(i * Int(counts.strides[0]))
        missing = missing or count == 0
        result.unsafe_set(i, Float64(count) if count > 0 else Float64(1))
    if missing:
        warn(warnings, "UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.")
    return result^


def constant_denominators(n_samples: Int, value: Int) raises -> Values:
    return nm.full[DType.float64](nm.Shape(n_samples), Float64(value))


def mean_feature_importances(importances: List[Values], n_features: Int) raises -> Values:
    """`np.mean(all, axis=0) / np.sum(mean)` over members with splits; zeros if none split."""
    var result = nm.zeros[DType.float64](nm.Shape(n_features))
    if len(importances) == 0:
        return result^
    for m in range(len(importances)):  # axis-0 reduction accumulates members in order
        if importances[m].size != n_features:
            raise Error("ValueError: member importances have the wrong length")
        for j in range(n_features):
            var value = importances[m].unsafe_get(j * Int(importances[m].strides[0]))
            result.unsafe_set(j, value if m == 0 else result.unsafe_get(j) + value)
    for j in range(n_features):
        result.unsafe_set(j, result.unsafe_get(j) / Float64(len(importances)))
    var total = numpy_sum(result)
    for j in range(n_features):
        result.unsafe_set(j, result.unsafe_get(j) / total)
    return result^


def argmax_rows(x: Values) raises -> Indices:
    """`np.argmax(x, axis=1)`: first maximal column per row."""
    if x.ndim != 2:
        raise Error("ValueError: argmax expects a matrix")
    var result = nm.zeros[DType.int64](nm.Shape(x.shape[0]))
    for i in range(x.shape[0]):
        var best = 0
        for c in range(1, x.shape[1]):
            if _get(x, i, c) > _get(x, i, best):
                best = c
        result.unsafe_set(i, Int64(best))
    return result^


def oob_accuracy(codes: Values, decisions: List[Values]) raises -> Float64:
    """`accuracy_score(y, argmax(oob_decision_function_, axis=1))`; subset accuracy over outputs."""
    var n = codes.shape[0]
    if _columns(codes) != len(decisions) or n == 0:
        raise Error("ValueError: OOB decisions need one matrix per output")
    var predicted = List[Indices]()
    for k in range(len(decisions)):
        predicted.append(argmax_rows(decisions[k]))
    var correct = 0
    for i in range(n):
        var hit = True
        for k in range(len(decisions)):
            hit = hit and Float64(predicted[k].unsafe_get(i)) == _get(codes, i, k)
        if hit:
            correct += 1
    return Float64(correct) / Float64(n)


def _reduce_rows(values: Values, column: Int) raises -> Float64:
    # NumPy reduces a single column pairwise; across several columns axis-0 sums run row by row.
    var n = values.shape[0]
    if values.shape[1] == 1:
        var column_values = nm.zeros[DType.float64](nm.Shape(n))
        for i in range(n):
            column_values.unsafe_set(i, _get(values, i, column))
        return numpy_sum(column_values)
    var total = Float64(0)
    for i in range(n):
        total += _get(values, i, column)
    return total


def r2_uniform(y: Values, prediction: Values, mut warnings: List[String]) raises -> Float64:
    """`r2_score(y, prediction)` with uniform averaging and force_finite=True."""
    var n = y.shape[0]
    var outputs = _columns(y)
    if prediction.shape[0] != n or _columns(prediction) != outputs:
        raise Error("ValueError: r2 target and prediction shapes differ")
    if n < 2:
        warn(warnings, "UndefinedMetricWarning: R^2 score is not well-defined with less than two samples.")
        return Float64("nan")
    var squared = nm.zeros[DType.float64](nm.Shape(n, outputs))
    for i in range(n):
        for k in range(outputs):
            var error = _get(y, i, k) - _get(prediction, i, k)
            squared.unsafe_set(i * outputs + k, error * error)
    var y2 = nm.zeros[DType.float64](nm.Shape(n, outputs))
    for i in range(n):
        for k in range(outputs):
            y2.unsafe_set(i * outputs + k, _get(y, i, k))
    var scores = nm.zeros[DType.float64](nm.Shape(outputs))
    for k in range(outputs):
        var mean = _reduce_rows(y2, k) / Float64(n)
        var centered = nm.zeros[DType.float64](nm.Shape(n, outputs))
        for i in range(n):
            var d = _get(y2, i, k) - mean
            centered.unsafe_set(i * outputs + k, d * d)
        var numerator = _reduce_rows(squared, k)
        var denominator = _reduce_rows(centered, k)
        var score = Float64(1)
        if numerator != 0 and denominator != 0:
            score = 1 - numerator / denominator
        elif numerator != 0:
            score = 0
        scores.unsafe_set(k, score)
    return numpy_sum(scores) / Float64(outputs)
