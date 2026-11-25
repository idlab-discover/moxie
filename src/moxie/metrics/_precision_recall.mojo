# Classification semantics from scikit-learn; BSD-3-Clause, see LICENSES/scikit-learn.txt.
"""Precision, recall and F-scores on dense classification targets."""
from std.builtin.sort import sort
from std.math import isfinite, isinf, isnan, nan
from std.sys import stderr

from moxie._labels import StringTarget, type_of_target
from moxie.metrics._classification import _Weights, _check_rows
from moxie.metrics._confusion import _Label, _numeric_labels, _validate_weights
from numojo import Item, NDArray


@fieldwise_init
struct ClassificationScores(Movable):
    """Per-label scores and support in the requested label order."""

    var precision: List[Float64]
    var recall: List[Float64]
    var fscore: List[Float64]
    var support: List[Float64]


@fieldwise_init
struct AveragedClassificationScores(Copyable, Movable):
    """Scalar scores after applying the requested average."""

    var precision: Float64
    var recall: Float64
    var fscore: Float64


struct _Counts(ImplicitlyCopyable, Movable):
    var true_positive: Float64
    var predicted: Float64
    var actual: Float64

    def __init__(
        out self, true_positive: Float64 = 0, predicted: Float64 = 0, actual: Float64 = 0
    ):
        self.true_positive = true_positive
        self.predicted = predicted
        self.actual = actual

    def add(mut self, actual: Bool, predicted: Bool, weight: Float64):
        if actual:
            self.actual += weight
        if predicted:
            self.predicted += weight
        if actual and predicted:
            self.true_positive += weight

    def merge(mut self, other: Self):
        self.true_positive += other.true_positive
        self.predicted += other.predicted
        self.actual += other.actual


def _weight(weights: _Weights, row: Int) raises -> Float64:
    return weights.value()[Item(row)] if weights else 1.0


def _single_label_counts[Label: _Label](
    y_true: List[Label],
    y_pred: List[Label],
    labels: Optional[List[Label]],
    pos_label: Optional[Label],
    average: String,
    sample_weight: _Weights,
) raises -> List[_Counts]:
    _check_rows(len(y_true), len(y_pred))
    if average == "samples":
        raise Error("ValueError: samples averaging requires multilabel targets")
    _validate_weights(sample_weight, len(y_true))
    var counts = Dict[Label, _Counts]()
    for row in range(len(y_true)):
        ref actual = y_true[row]
        ref predicted = y_pred[row]
        var weight = _weight(sample_weight, row)
        if actual == predicted:
            var tally = counts.get(actual, _Counts())
            tally.add(True, True, weight)
            counts[actual.copy()] = tally
        else:
            var actual_tally = counts.get(actual, _Counts())
            actual_tally.add(True, False, weight)
            counts[actual.copy()] = actual_tally
            var predicted_tally = counts.get(predicted, _Counts())
            predicted_tally.add(False, True, weight)
            counts[predicted.copy()] = predicted_tally

    var result = List[_Counts]()
    if average == "binary":
        if len(counts) > 2:
            raise Error("ValueError: binary averaging requires binary targets")
        var present = pos_label and pos_label.value() in counts
        if not present and len(counts) == 2:
            raise Error("ValueError: pos_label must be one of the observed labels")
        result.append(counts.get(pos_label.value(), _Counts()) if present else _Counts())
    elif labels:
        for label in labels.value():
            result.append(counts.get(label, _Counts()))
    else:
        var order = List[Label](capacity=len(counts))
        for label in counts.keys():
            order.append(label.copy())
        sort(Span(order))
        for label in order:
            result.append(counts.get(label, _Counts()))
    return result^


def _multilabel_counts[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    labels: Optional[List[Scalar[dtype]]],
    average: String,
    sample_weight: _Weights,
) raises -> List[_Counts]:
    if average == "binary":
        raise Error("ValueError: binary averaging requires binary targets")
    var rows = y_true.shape[0]
    var columns = y_true.shape[1]
    _check_rows(rows, y_pred.shape[0])
    if y_pred.shape[1] != columns:
        raise Error("ValueError: multilabel targets must have the same number of columns")
    _validate_weights(sample_weight, rows)
    var selected = List[Int](capacity=columns if not labels else len(labels.value()))
    if labels:
        for label in labels.value():
            var index = Float64(label)
            if (
                not isfinite(index)
                or index < 0
                or index >= Float64(columns)
                or index != Float64(Int(index))
            ):
                raise Error("ValueError: multilabel labels must be valid column indices")
            selected.append(Int(index))
    else:
        for column in range(columns):
            selected.append(column)

    var samplewise = average == "samples"
    var result = List[_Counts](length=rows if samplewise else len(selected), fill=_Counts())
    for row in range(rows):
        var weight = _weight(sample_weight, row)
        if samplewise:
            var tally = _Counts()
            for column in selected:
                tally.add(y_true[Item(row, column)] != 0, y_pred[Item(row, column)] != 0, 1.0)
            result[row] = _Counts(
                tally.true_positive * weight, tally.predicted * weight, tally.actual * weight
            )
        else:
            for i in range(len(selected)):
                var column = selected[i]
                result[i].add(
                    y_true[Item(row, column)] != 0, y_pred[Item(row, column)] != 0, weight
                )
    return result^


def _numeric_counts[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    labels: Optional[List[Scalar[dtype]]],
    pos_label: Scalar[dtype],
    average: String,
    sample_weight: _Weights,
) raises -> List[_Counts]:
    if (y_true.ndim == 2 and y_true.shape[1] > 1) or (
        y_pred.ndim == 2 and y_pred.shape[1] > 1
    ):
        if (
            type_of_target(y_true) != "multilabel-indicator"
            or type_of_target(y_pred) != "multilabel-indicator"
        ):
            raise Error("ValueError: expected matching multilabel indicators")
        return _multilabel_counts(y_true, y_pred, labels, average, sample_weight)
    var order = Optional[List[Scalar[dtype]]]()
    if labels:
        var canonical = List[Scalar[dtype]](capacity=len(labels.value()))
        for label in labels.value():
            canonical.append(0 if label == 0 else label)
        order = canonical^
    var positive = Optional[Scalar[dtype]](0 if pos_label == 0 else pos_label)
    return _single_label_counts(
        _numeric_labels(y_true), _numeric_labels(y_pred), order, positive, average, sample_weight
    )


def _string_counts(
    y_true: StringTarget,
    y_pred: StringTarget,
    labels: Optional[List[String]],
    pos_label: Optional[String],
    average: String,
    sample_weight: _Weights,
) raises -> List[_Counts]:
    if y_true.columns() != 1 or y_pred.columns() != 1:
        raise Error("ValueError: string targets must have one label per sample")
    return _single_label_counts(
        y_true.values(), y_pred.values(), labels, pos_label, average, sample_weight
    )


def _list_counts[Label: _Label](
    y_true: List[Label],
    y_pred: List[Label],
    labels: Optional[List[Label]],
    pos_label: Optional[Label],
    average: String,
    sample_weight: _Weights,
) raises -> List[_Counts]:
    comptime assert Label == Int or Label == String, "List targets require Int or String labels"
    comptime if Label == Int:
        if not pos_label:
            var default_positive = Optional[Label](rebind[Label](Int(1)).copy())
            return _single_label_counts(
                y_true, y_pred, labels, default_positive, average, sample_weight
            )
    return _single_label_counts(y_true, y_pred, labels, pos_label, average, sample_weight)


@fieldwise_init
struct _Division:
    var fallback: Float64
    var warn: Bool
    var undefined: Bool

    def divide(mut self, numerator: Float64, denominator: Float64) -> Float64:
        if denominator == 0:
            self.undefined = True
            return self.fallback
        return numerator / denominator

    def report(self, metric: String):
        if self.warn and self.undefined:
            print(
                "UndefinedMetricWarning:", metric,
                "is ill-defined and being set to 0.0", file=stderr,
            )


struct _Ratios:
    var beta: Float64
    var beta_squared: Float64
    var precision: _Division
    var recall: _Division
    var fscore: _Division

    def __init__(out self, beta: Float64, zero_division: Optional[Float64]) raises:
        if isnan(beta) or beta < 0:
            raise Error("ValueError: beta must be nonnegative")
        var fallback = zero_division.value() if zero_division else 0.0
        if fallback != 0 and fallback != 1 and not isnan(fallback):
            raise Error("ValueError: zero_division must be 0, 1, or NaN")
        self.beta = beta
        self.beta_squared = beta * beta
        if isinf(self.beta_squared) and not isinf(beta):
            raise Error("OverflowError: beta squared exceeds Float64 range")
        self.precision = _Division(fallback, not zero_division, False)
        self.recall = _Division(fallback, not zero_division, False)
        self.fscore = _Division(fallback, not zero_division, False)

    def evaluate(mut self, counts: _Counts) -> AveragedClassificationScores:
        var precision = self.precision.divide(counts.true_positive, counts.predicted)
        var recall = self.recall.divide(counts.true_positive, counts.actual)
        var fscore: Float64
        if self.beta == 0:
            fscore = precision
        elif isinf(self.beta_squared):
            fscore = recall
        else:
            fscore = self.fscore.divide(
                (1.0 + self.beta_squared) * counts.true_positive,
                self.beta_squared * counts.actual + counts.predicted,
            )
        return AveragedClassificationScores(precision, recall, fscore)

    def report(self, metric: String):
        if metric == "" or metric == "Precision":
            self.precision.report("Precision")
        if metric == "" or metric == "Recall":
            self.recall.report("Recall")
        if metric == "" or metric == "F-score":
            self.fscore.report("F-score")


struct _Mean:
    var sum: Float64
    var count: Int
    var weighted_sum: Float64
    var total_weight: Float64

    def __init__(out self):
        self.sum = 0
        self.count = 0
        self.weighted_sum = 0
        self.total_weight = 0

    def add(mut self, value: Float64, weight: Float64):
        if not isnan(value):
            self.sum += value
            self.count += 1
            self.weighted_sum += weight * value
            self.total_weight += weight

    def value(self) -> Float64:
        if self.count == 0:
            return nan[DType.float64]()
        # The reference falls back to the unweighted mean if weights cancel.
        if self.total_weight == 0:
            return self.sum / Float64(self.count)
        return self.weighted_sum / self.total_weight


def _per_label_scores(
    counts: List[_Counts], beta: Float64, zero_division: Optional[Float64]
) raises -> ClassificationScores:
    var ratios = _Ratios(beta, zero_division)
    var result = ClassificationScores(
        List[Float64](capacity=len(counts)), List[Float64](capacity=len(counts)),
        List[Float64](capacity=len(counts)), List[Float64](capacity=len(counts)),
    )
    for tally in counts:
        var scores = ratios.evaluate(tally)
        result.precision.append(scores.precision)
        result.recall.append(scores.recall)
        result.fscore.append(scores.fscore)
        result.support.append(tally.actual)
    ratios.report("")
    return result^


def _averaged_scores(
    counts: List[_Counts],
    beta: Float64,
    average: String,
    sample_weight: _Weights,
    zero_division: Optional[Float64],
    metric: String = "",
) raises -> AveragedClassificationScores:
    var ratios = _Ratios(beta, zero_division)
    if average == "binary" or average == "micro":
        var total = _Counts()
        for tally in counts:
            total.merge(tally)
        var result = ratios.evaluate(total)
        ratios.report(metric)
        return result^
    if average != "macro" and average != "weighted" and average != "samples":
        raise Error("ValueError: average must be binary, micro, macro, weighted, or samples")

    var precision = _Mean()
    var recall = _Mean()
    var fscore = _Mean()
    for i in range(len(counts)):
        var tally = counts[i]
        var scores = ratios.evaluate(tally)
        var weight = tally.actual if average == "weighted" else 1.0
        if average == "samples":
            weight = _weight(sample_weight, i)
        precision.add(scores.precision, weight)
        recall.add(scores.recall, weight)
        fscore.add(scores.fscore, weight)
    ratios.report(metric)
    return AveragedClassificationScores(precision.value(), recall.value(), fscore.value())


def precision_recall_fscore_support[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    *,
    beta: Float64 = 1.0,
    labels: Optional[List[Scalar[dtype]]] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> ClassificationScores:
    """Return precision, recall, F-beta and support for each selected label."""
    var counts = _numeric_counts(
        y_true, y_pred, labels, 1, "", sample_weight
    )
    return _per_label_scores(counts, beta, zero_division)


def precision_recall_fscore_support[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    *,
    beta: Float64 = 1.0,
    average: String,
    labels: Optional[List[Scalar[dtype]]] = None,
    pos_label: Scalar[dtype] = 1,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> AveragedClassificationScores:
    """Return scalar precision, recall and F-beta for the requested average."""
    var counts = _numeric_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(counts, beta, average, sample_weight, zero_division)


def precision_score[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    *,
    average: String = "binary",
    labels: Optional[List[Scalar[dtype]]] = None,
    pos_label: Scalar[dtype] = 1,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the fraction of predicted positives that are correct."""
    var counts = _numeric_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, 1.0, average, sample_weight, zero_division, "Precision"
    ).precision


def recall_score[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    *,
    average: String = "binary",
    labels: Optional[List[Scalar[dtype]]] = None,
    pos_label: Scalar[dtype] = 1,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the fraction of actual positives that were predicted."""
    var counts = _numeric_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, 1.0, average, sample_weight, zero_division, "Recall"
    ).recall


def f1_score[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    *,
    average: String = "binary",
    labels: Optional[List[Scalar[dtype]]] = None,
    pos_label: Scalar[dtype] = 1,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the F1 score for the requested average."""
    var counts = _numeric_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, 1.0, average, sample_weight, zero_division, "F-score"
    ).fscore


def fbeta_score[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    *,
    beta: Float64,
    average: String = "binary",
    labels: Optional[List[Scalar[dtype]]] = None,
    pos_label: Scalar[dtype] = 1,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the F-beta score for the requested average."""
    var counts = _numeric_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, beta, average, sample_weight, zero_division, "F-score"
    ).fscore


def precision_recall_fscore_support(
    y_true: StringTarget,
    y_pred: StringTarget,
    *,
    beta: Float64 = 1.0,
    labels: Optional[List[String]] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> ClassificationScores:
    """Return precision, recall, F-beta and support for each selected label."""
    var counts = _string_counts(
        y_true, y_pred, labels, None, "", sample_weight
    )
    return _per_label_scores(counts, beta, zero_division)


def precision_recall_fscore_support(
    y_true: StringTarget,
    y_pred: StringTarget,
    *,
    beta: Float64 = 1.0,
    average: String,
    labels: Optional[List[String]] = None,
    pos_label: Optional[String] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> AveragedClassificationScores:
    """Return scalar precision, recall and F-beta for the requested average."""
    var counts = _string_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(counts, beta, average, sample_weight, zero_division)


def precision_score(
    y_true: StringTarget,
    y_pred: StringTarget,
    *,
    average: String = "binary",
    labels: Optional[List[String]] = None,
    pos_label: Optional[String] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the fraction of predicted positives that are correct."""
    var counts = _string_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, 1.0, average, sample_weight, zero_division, "Precision"
    ).precision


def recall_score(
    y_true: StringTarget,
    y_pred: StringTarget,
    *,
    average: String = "binary",
    labels: Optional[List[String]] = None,
    pos_label: Optional[String] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the fraction of actual positives that were predicted."""
    var counts = _string_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, 1.0, average, sample_weight, zero_division, "Recall"
    ).recall


def f1_score(
    y_true: StringTarget,
    y_pred: StringTarget,
    *,
    average: String = "binary",
    labels: Optional[List[String]] = None,
    pos_label: Optional[String] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the F1 score for the requested average."""
    var counts = _string_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, 1.0, average, sample_weight, zero_division, "F-score"
    ).fscore


def fbeta_score(
    y_true: StringTarget,
    y_pred: StringTarget,
    *,
    beta: Float64,
    average: String = "binary",
    labels: Optional[List[String]] = None,
    pos_label: Optional[String] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the F-beta score for the requested average."""
    var counts = _string_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, beta, average, sample_weight, zero_division, "F-score"
    ).fscore


def precision_recall_fscore_support[Label: _Label](
    y_true: List[Label],
    y_pred: List[Label],
    *,
    beta: Float64 = 1.0,
    labels: Optional[List[Label]] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> ClassificationScores:
    """Return precision, recall, F-beta and support for each selected label."""
    var counts = _list_counts(
        y_true, y_pred, labels, None, "", sample_weight
    )
    return _per_label_scores(counts, beta, zero_division)


def precision_recall_fscore_support[Label: _Label](
    y_true: List[Label],
    y_pred: List[Label],
    *,
    beta: Float64 = 1.0,
    average: String,
    labels: Optional[List[Label]] = None,
    pos_label: Optional[Label] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> AveragedClassificationScores:
    """Return scalar precision, recall and F-beta for the requested average."""
    var counts = _list_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(counts, beta, average, sample_weight, zero_division)


def precision_score[Label: _Label](
    y_true: List[Label],
    y_pred: List[Label],
    *,
    average: String = "binary",
    labels: Optional[List[Label]] = None,
    pos_label: Optional[Label] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the fraction of predicted positives that are correct."""
    var counts = _list_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, 1.0, average, sample_weight, zero_division, "Precision"
    ).precision


def recall_score[Label: _Label](
    y_true: List[Label],
    y_pred: List[Label],
    *,
    average: String = "binary",
    labels: Optional[List[Label]] = None,
    pos_label: Optional[Label] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the fraction of actual positives that were predicted."""
    var counts = _list_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, 1.0, average, sample_weight, zero_division, "Recall"
    ).recall


def f1_score[Label: _Label](
    y_true: List[Label],
    y_pred: List[Label],
    *,
    average: String = "binary",
    labels: Optional[List[Label]] = None,
    pos_label: Optional[Label] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the F1 score for the requested average."""
    var counts = _list_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, 1.0, average, sample_weight, zero_division, "F-score"
    ).fscore


def fbeta_score[Label: _Label](
    y_true: List[Label],
    y_pred: List[Label],
    *,
    beta: Float64,
    average: String = "binary",
    labels: Optional[List[Label]] = None,
    pos_label: Optional[Label] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> Float64:
    """Return the F-beta score for the requested average."""
    var counts = _list_counts(
        y_true, y_pred, labels, pos_label, average, sample_weight
    )
    return _averaged_scores(
        counts, beta, average, sample_weight, zero_division, "F-score"
    ).fscore
