# Classification semantics from scikit-learn; BSD-3-Clause, see LICENSES/scikit-learn.txt.
"""Agreement metrics on dense classification targets."""
from std.math import isfinite

from moxie._labels import StringTarget, type_of_target
from numojo import Item, NDArray

comptime _Weights = Optional[NDArray[DType.float64]]


@fieldwise_init
struct _Agreement:
    var total: Float64
    var exact: Float64
    var differences: Float64
    var columns: Int

    def accuracy(self, normalize: Bool) raises -> Float64:
        if not normalize:
            return self.exact
        if self.total == 0:
            raise Error("ZeroDivisionError: sample weights sum to zero")
        return self.exact / self.total

    def zero_one(self, normalize: Bool) raises -> Float64:
        if normalize:
            return 1.0 - self.accuracy(True)
        return self.total - self.exact

    def hamming(self) raises -> Float64:
        if self.columns == 1 and self.total == 0:
            raise Error("ZeroDivisionError: sample weights sum to zero")
        # The reference's multilabel path uses IEEE division for a zero total.
        return self.differences / (self.total * Float64(self.columns))


def _check_rows(rows: Int, predicted_rows: Int) raises:
    if rows != predicted_rows:
        raise Error("ValueError: inconsistent numbers of samples")
    if rows == 0:
        raise Error("ValueError: classification metrics require at least one sample")


def _accumulate[F: def(Int) raises -> Int](
    mismatches: F, rows: Int, columns: Int, sample_weight: _Weights
) raises -> _Agreement:
    var total: Float64 = 0
    var exact: Float64 = 0
    var differences: Float64 = 0
    if sample_weight:
        ref weights = sample_weight.value()
        if weights.ndim != 1 or weights.shape[0] != rows:
            raise Error(
                "ValueError: sample_weight must be a vector with one value per sample"
            )
        var nonzero = False
        for row in range(rows):
            var weight = weights[Item(row)]
            if not isfinite(weight):
                raise Error("ValueError: sample_weight must be finite")
            nonzero = nonzero or weight != 0
            var count = mismatches(row)
            total += weight
            if count == 0:
                exact += weight
            differences += weight * Float64(count)
        if not nonzero:
            raise Error("ValueError: sample_weight must contain a nonzero value")
    else:
        total = Float64(rows)
        for row in range(rows):
            var count = mismatches(row)
            if count == 0:
                exact += 1
            differences += Float64(count)
    return _Agreement(total, exact, differences, columns)


def _agreement[dtype: DType](
    y_true: NDArray[dtype], y_pred: NDArray[dtype], sample_weight: _Weights
) raises -> _Agreement:
    var true_kind = type_of_target(y_true)
    var pred_kind = type_of_target(y_pred)
    var columns = 1
    if true_kind == "multilabel-indicator" and pred_kind == "multilabel-indicator":
        columns = y_true.shape[1]
        if y_pred.shape[1] != columns:
            raise Error(
                "ValueError: multilabel targets must have the same number of columns"
            )
    elif not (
        (true_kind == "binary" or true_kind == "multiclass")
        and (pred_kind == "binary" or pred_kind == "multiclass")
    ):
        raise Error(
            "ValueError: expected binary, multiclass, or matching multilabel targets"
        )
    var rows = y_true.shape[0]
    _check_rows(rows, y_pred.shape[0])

    def mismatches(row: Int) raises {imm y_true, imm y_pred, columns} -> Int:
        if columns == 1:
            var truth = y_true[Item(row)] if y_true.ndim == 1 else y_true[Item(row, 0)]
            var predicted = y_pred[Item(row)] if y_pred.ndim == 1 else y_pred[Item(row, 0)]
            return Int(truth != predicted)
        var count = 0
        for column in range(columns):
            count += Int(y_true[Item(row, column)] != y_pred[Item(row, column)])
        return count

    return _accumulate(mismatches, rows, columns, sample_weight)


def _agreement(
    y_true: StringTarget, y_pred: StringTarget, sample_weight: _Weights
) raises -> _Agreement:
    if y_true.columns() != 1 or y_pred.columns() != 1:
        raise Error(
            "ValueError: string classification targets must have one label per sample"
        )
    _check_rows(y_true.rows(), y_pred.rows())

    def mismatches(row: Int) raises {imm y_true, imm y_pred} -> Int:
        return Int(y_true.get(row) != y_pred.get(row))

    return _accumulate(mismatches, y_true.rows(), 1, sample_weight)


def _agreement[Label: Equatable & Copyable](
    y_true: List[Label], y_pred: List[Label], sample_weight: _Weights
) raises -> _Agreement:
    comptime assert Label == Int or Label == String, (
        "List targets require Int or String labels"
    )
    _check_rows(len(y_true), len(y_pred))

    def mismatches(row: Int) raises {imm y_true, imm y_pred} -> Int:
        return Int(y_true[row] != y_pred[row])

    return _accumulate(mismatches, len(y_true), 1, sample_weight)


def accuracy_score[dtype: DType](
    y_true: NDArray[dtype], y_pred: NDArray[dtype], *,
    normalize: Bool = True,
    sample_weight: _Weights = None,
) raises -> Float64:
    """Fraction or weighted count of rows whose labels match exactly."""
    return _agreement(y_true, y_pred, sample_weight).accuracy(normalize)


def accuracy_score(
    y_true: StringTarget, y_pred: StringTarget, *,
    normalize: Bool = True,
    sample_weight: _Weights = None,
) raises -> Float64:
    """Fraction or weighted count of rows whose labels match exactly."""
    return _agreement(y_true, y_pred, sample_weight).accuracy(normalize)


def accuracy_score[Label: Equatable & Copyable](
    y_true: List[Label], y_pred: List[Label], *,
    normalize: Bool = True,
    sample_weight: _Weights = None,
) raises -> Float64:
    """Fraction or weighted count of rows whose labels match exactly."""
    return _agreement(y_true, y_pred, sample_weight).accuracy(normalize)


def zero_one_loss[dtype: DType](
    y_true: NDArray[dtype], y_pred: NDArray[dtype], *,
    normalize: Bool = True,
    sample_weight: _Weights = None,
) raises -> Float64:
    """Fraction or weighted count of rows with any differing label."""
    return _agreement(y_true, y_pred, sample_weight).zero_one(normalize)


def zero_one_loss(
    y_true: StringTarget, y_pred: StringTarget, *,
    normalize: Bool = True,
    sample_weight: _Weights = None,
) raises -> Float64:
    """Fraction or weighted count of rows with any differing label."""
    return _agreement(y_true, y_pred, sample_weight).zero_one(normalize)


def zero_one_loss[Label: Equatable & Copyable](
    y_true: List[Label], y_pred: List[Label], *,
    normalize: Bool = True,
    sample_weight: _Weights = None,
) raises -> Float64:
    """Fraction or weighted count of rows with any differing label."""
    return _agreement(y_true, y_pred, sample_weight).zero_one(normalize)


def hamming_loss[dtype: DType](
    y_true: NDArray[dtype], y_pred: NDArray[dtype], *,
    sample_weight: _Weights = None,
) raises -> Float64:
    """Fraction of differing entries, weighted by sample."""
    return _agreement(y_true, y_pred, sample_weight).hamming()


def hamming_loss(
    y_true: StringTarget, y_pred: StringTarget, *,
    sample_weight: _Weights = None,
) raises -> Float64:
    """Fraction of differing entries, weighted by sample."""
    return _agreement(y_true, y_pred, sample_weight).hamming()


def hamming_loss[Label: Equatable & Copyable](
    y_true: List[Label], y_pred: List[Label], *,
    sample_weight: _Weights = None,
) raises -> Float64:
    """Fraction of differing entries, weighted by sample."""
    return _agreement(y_true, y_pred, sample_weight).hamming()
