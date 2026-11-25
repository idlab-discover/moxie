# Classification semantics from scikit-learn; BSD-3-Clause, see LICENSES/scikit-learn.txt.
"""Confusion matrices and class-balanced scoring on dense targets."""
from std.builtin.sort import sort
from std.math import isfinite, isinf, isnan

from moxie._labels import StringTarget, type_of_target
from moxie.metrics._classification import _Weights, _check_rows
from numojo import Item, NDArray, Shape, zeros

comptime _Label = Hashable & Comparable & Copyable & Deinitable


@fieldwise_init
struct _LabelPairs(Movable):
    var truth: List[Int]
    var prediction: List[Int]
    var classes: Int


def _index_pairs[Label: _Label](
    y_true: List[Label], y_pred: List[Label], labels: Optional[List[Label]]
) raises -> _LabelPairs:
    var rows = len(y_true)
    _check_rows(rows, len(y_pred))
    var index = Dict[Label, Int]()
    var classes: Int
    if labels:
        ref order = labels.value()
        classes = len(order)
        if classes == 0:
            raise Error("ValueError: labels must contain at least one label")
        for i in range(classes):
            index[order[i].copy()] = i
    else:
        var order = List[Label]()
        for row in range(rows):
            if index.get(y_true[row], -1) == -1:
                index[y_true[row].copy()] = 0
                order.append(y_true[row].copy())
            if index.get(y_pred[row], -1) == -1:
                index[y_pred[row].copy()] = 0
                order.append(y_pred[row].copy())
        sort(Span(order))
        classes = len(order)
        for i in range(classes):
            index[order[i].copy()] = i

    var truth = List[Int](capacity=rows)
    var prediction = List[Int](capacity=rows)
    var found_truth = False
    for row in range(rows):
        var actual = index.get(y_true[row], -1)
        found_truth = found_truth or actual >= 0
        truth.append(actual)
        prediction.append(index.get(y_pred[row], -1))
    if not found_truth:
        raise Error("ValueError: at least one specified label must be in y_true")
    return _LabelPairs(truth^, prediction^, classes)


def _numeric_labels[dtype: DType](target: NDArray[dtype]) raises -> List[Scalar[dtype]]:
    var kind = type_of_target(target)
    if kind != "binary" and kind != "multiclass":
        raise Error("ValueError: expected binary or multiclass targets")
    var result = List[Scalar[dtype]](capacity=target.shape[0])
    for row in range(target.shape[0]):
        var value = target[Item(row)] if target.ndim == 1 else target[Item(row, 0)]
        # Floating-point Dict keys distinguish the bit patterns of signed zeros.
        result.append(0 if value == 0 else value)
    return result^


def _label_pairs[dtype: DType](
    y_true: NDArray[dtype], y_pred: NDArray[dtype],
    labels: Optional[List[Scalar[dtype]]],
) raises -> _LabelPairs:
    var order = Optional[List[Scalar[dtype]]]()
    if labels:
        var canonical = List[Scalar[dtype]](capacity=len(labels.value()))
        for value in labels.value():
            canonical.append(0 if value == 0 else value)
        order = canonical^
    return _index_pairs(_numeric_labels(y_true), _numeric_labels(y_pred), order)


def _label_pairs(
    y_true: StringTarget, y_pred: StringTarget, labels: Optional[List[String]]
) raises -> _LabelPairs:
    if y_true.columns() != 1 or y_pred.columns() != 1:
        raise Error("ValueError: string targets must have one label per sample")
    return _index_pairs(y_true.values(), y_pred.values(), labels)


def _label_pairs[Label: _Label](
    y_true: List[Label], y_pred: List[Label], labels: Optional[List[Label]]
) raises -> _LabelPairs:
    comptime assert Label == Int or Label == String, "List targets require Int or String labels"
    return _index_pairs(y_true, y_pred, labels)


def _validate_weights(sample_weight: _Weights, rows: Int) raises:
    if not sample_weight:
        return
    ref weights = sample_weight.value()
    if weights.ndim != 1 or weights.shape[0] != rows:
        raise Error("ValueError: sample_weight must be a vector with one value per sample")
    var nonzero = False
    for row in range(rows):
        var weight = weights[Item(row)]
        if not isfinite(weight):
            raise Error("ValueError: sample_weight must be finite")
        nonzero = nonzero or weight != 0
    if not nonzero:
        raise Error("ValueError: sample_weight must contain a nonzero value")


def _accumulate[dtype: DType](
    pairs: _LabelPairs, sample_weight: _Weights
) raises -> NDArray[dtype]:
    var classes = pairs.classes
    if classes > Int.MAX // 8 // classes:
        raise Error("ValueError: confusion matrix dimensions overflow")
    _validate_weights(sample_weight, len(pairs.truth))
    var result = zeros[dtype](Shape(classes, classes))
    for row in range(len(pairs.truth)):
        var actual = pairs.truth[row]
        var predicted = pairs.prediction[row]
        if actual < 0 or predicted < 0:
            continue
        var cell = Item(actual, predicted)
        var weight = Scalar[dtype](1)
        comptime if dtype == DType.float64:
            if sample_weight:
                weight = sample_weight.value()[Item(row)].cast[dtype]()
        result[cell] = result[cell] + weight
    return result^


def _finite_or_limit(value: Float64) -> Float64:
    """Convert NaN to zero and saturate infinite confusion counts."""
    if isnan(value):
        return 0
    if isinf(value):
        comptime largest = Float64(1.7976931348623157e308)
        return -largest if value < 0 else largest
    return value


def _weighted_matrix(
    pairs: _LabelPairs,
    sample_weight: _Weights,
    normalize: Optional[String] = None,
) raises -> NDArray[DType.float64]:
    var result = _accumulate[DType.float64](pairs, sample_weight)
    if normalize:
        _normalize_matrix(result, normalize.value())
    else:
        for i in range(pairs.classes):
            for j in range(pairs.classes):
                var cell = Item(i, j)
                result[cell] = _finite_or_limit(result[cell])
    return result^


def _normalize_matrix(mut result: NDArray[DType.float64], normalize: String) raises:
    if normalize != "true" and normalize != "pred" and normalize != "all":
        raise Error("ValueError: normalize must be 'true', 'pred', or 'all'")
    var classes = result.shape[0]
    var by_true = normalize == "true"
    var by_pred = normalize == "pred"
    var groups = classes if by_true or by_pred else 1
    var totals = List[Float64](length=groups, fill=0)
    for i in range(classes):
        for j in range(classes):
            var group = i if by_true else (j if by_pred else 0)
            totals[group] += result[Item(i, j)]
    for i in range(classes):
        for j in range(classes):
            var group = i if by_true else (j if by_pred else 0)
            var cell = Item(i, j)
            result[cell] = _finite_or_limit(result[cell] / totals[group])


def _balanced_accuracy(
    pairs: _LabelPairs, sample_weight: _Weights, adjusted: Bool
) raises -> Float64:
    _validate_weights(sample_weight, len(pairs.truth))
    # Keep only observed cells and sum each row in class order as in the matrix.
    var cells = List[Dict[Int, Float64]](capacity=pairs.classes)
    for _ in range(pairs.classes):
        cells.append(Dict[Int, Float64]())
    for row in range(len(pairs.truth)):
        ref counts = cells[pairs.truth[row]]
        var predicted = pairs.prediction[row]
        var weight: Float64 = 1
        if sample_weight:
            weight = sample_weight.value()[Item(row)]
        counts[predicted] = counts.get(predicted, 0) + weight

    var score: Float64 = 0
    var classes = 0
    for actual in range(pairs.classes):
        ref counts = cells[actual]
        var columns = List[Int](capacity=len(counts))
        for key in counts.keys():
            columns.append(key)
        sort(Span(columns))
        var support: Float64 = 0
        for predicted in columns:
            support += _finite_or_limit(counts.get(predicted, 0))
        var recall = _finite_or_limit(counts.get(actual, 0)) / support
        if not isnan(recall):
            score += recall
            classes += 1
    if adjusted and classes == 0:
        raise Error("ZeroDivisionError: no class recalls for chance adjustment")
    score /= Float64(classes)
    if adjusted:
        var chance = 1.0 / Float64(classes)
        score = (score - chance) / (1.0 - chance)
    return score


def confusion_matrix[dtype: DType](
    y_true: NDArray[dtype], y_pred: NDArray[dtype], *,
    labels: Optional[List[Scalar[dtype]]] = None,
) raises -> NDArray[DType.int64]:
    """Count samples by true label (rows) and predicted label (columns)."""
    return _accumulate[DType.int64](_label_pairs(y_true, y_pred, labels), None)


def confusion_matrix[dtype: DType](
    y_true: NDArray[dtype], y_pred: NDArray[dtype], *,
    sample_weight: _Weights,
    labels: Optional[List[Scalar[dtype]]] = None,
) raises -> NDArray[DType.float64]:
    """Sum sample weights by true and predicted label."""
    return _weighted_matrix(_label_pairs(y_true, y_pred, labels), sample_weight)


def confusion_matrix[dtype: DType](
    y_true: NDArray[dtype], y_pred: NDArray[dtype], *,
    normalize: String,
    labels: Optional[List[Scalar[dtype]]] = None,
    sample_weight: _Weights = None,
) raises -> NDArray[DType.float64]:
    """Normalize by true label ('true'), predicted label ('pred'), or total ('all')."""
    return _weighted_matrix(_label_pairs(y_true, y_pred, labels), sample_weight, normalize)


def balanced_accuracy_score[dtype: DType](
    y_true: NDArray[dtype], y_pred: NDArray[dtype], *,
    sample_weight: _Weights = None,
    adjusted: Bool = False,
) raises -> Float64:
    """Mean class recall, optionally adjusted for chance."""
    return _balanced_accuracy(_label_pairs(y_true, y_pred, None), sample_weight, adjusted)


def confusion_matrix(
    y_true: StringTarget, y_pred: StringTarget, *,
    labels: Optional[List[String]] = None,
) raises -> NDArray[DType.int64]:
    """Count samples by true label (rows) and predicted label (columns)."""
    return _accumulate[DType.int64](_label_pairs(y_true, y_pred, labels), None)


def confusion_matrix(
    y_true: StringTarget, y_pred: StringTarget, *,
    sample_weight: _Weights,
    labels: Optional[List[String]] = None,
) raises -> NDArray[DType.float64]:
    """Sum sample weights by true and predicted label."""
    return _weighted_matrix(_label_pairs(y_true, y_pred, labels), sample_weight)


def confusion_matrix(
    y_true: StringTarget, y_pred: StringTarget, *,
    normalize: String,
    labels: Optional[List[String]] = None,
    sample_weight: _Weights = None,
) raises -> NDArray[DType.float64]:
    """Normalize by true label ('true'), predicted label ('pred'), or total ('all')."""
    return _weighted_matrix(_label_pairs(y_true, y_pred, labels), sample_weight, normalize)


def balanced_accuracy_score(
    y_true: StringTarget, y_pred: StringTarget, *,
    sample_weight: _Weights = None,
    adjusted: Bool = False,
) raises -> Float64:
    """Mean class recall, optionally adjusted for chance."""
    return _balanced_accuracy(_label_pairs(y_true, y_pred, None), sample_weight, adjusted)


def confusion_matrix[Label: _Label](
    y_true: List[Label], y_pred: List[Label], *,
    labels: Optional[List[Label]] = None,
) raises -> NDArray[DType.int64]:
    """Count samples by true label (rows) and predicted label (columns)."""
    return _accumulate[DType.int64](_label_pairs(y_true, y_pred, labels), None)


def confusion_matrix[Label: _Label](
    y_true: List[Label], y_pred: List[Label], *,
    sample_weight: _Weights,
    labels: Optional[List[Label]] = None,
) raises -> NDArray[DType.float64]:
    """Sum sample weights by true and predicted label."""
    return _weighted_matrix(_label_pairs(y_true, y_pred, labels), sample_weight)


def confusion_matrix[Label: _Label](
    y_true: List[Label], y_pred: List[Label], *,
    normalize: String,
    labels: Optional[List[Label]] = None,
    sample_weight: _Weights = None,
) raises -> NDArray[DType.float64]:
    """Normalize by true label ('true'), predicted label ('pred'), or total ('all')."""
    return _weighted_matrix(_label_pairs(y_true, y_pred, labels), sample_weight, normalize)


def balanced_accuracy_score[Label: _Label](
    y_true: List[Label], y_pred: List[Label], *,
    sample_weight: _Weights = None,
    adjusted: Bool = False,
) raises -> Float64:
    """Mean class recall, optionally adjusted for chance."""
    return _balanced_accuracy(_label_pairs(y_true, y_pred, None), sample_weight, adjusted)
