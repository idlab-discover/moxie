# Regression metric semantics from scikit-learn; BSD-3-Clause, see LICENSES/scikit-learn.txt.
"""Regression metrics on borrowed, strided dense targets."""
from std.builtin.sort import sort
from std.math import isfinite, nan, sqrt
from std.memory import bitcast
from std.sys import stderr

from moxie.metrics._classification import _Weights
from moxie.metrics._confusion import _validate_weights
from numojo import NDArray, Shape

comptime _Output = NDArray[DType.float64]
comptime _EPSILON = Float64(2.220446049250313e-16)


@fieldwise_init
struct _RegressionResult(Movable):
    var average: Float64
    var per_output: Optional[_Output]


def _columns[dtype: DType](target: NDArray[dtype]) raises -> Int:
    comptime assert (
        dtype == DType.float32 or dtype == DType.float64 or dtype.is_integral()
    ), "regression targets must have Float32, Float64 or integral dtype"
    if target.ndim == 1:
        return 1
    if target.ndim != 2 or target.shape[1] == 0:
        raise Error("ValueError: expected a vector or a matrix with nonempty output columns")
    return target.shape[1]


def _entry[dtype: DType](target: NDArray[dtype], row: Int, column: Int) -> Float64:
    # Shape validation and the caller's loop bounds establish valid logical indices.
    var offset = row * target.strides.unsafe_get(0)
    if target.ndim == 2:
        offset += column * target.strides.unsafe_get(1)
    return Float64(target.unsafe_get(offset))


def _check_targets[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype]
) raises -> Int:
    var columns = _columns(y_true)
    if columns != _columns(y_pred):
        raise Error("ValueError: y_true and y_pred have different numbers of outputs")
    if y_true.shape[0] != y_pred.shape[0]:
        raise Error("ValueError: inconsistent numbers of samples")
    if y_true.shape[0] == 0:
        raise Error("ValueError: regression metrics require at least one sample")
    return columns


def _weight_at(weights: _Weights, row: Int) -> Float64:
    if weights:
        ref values = weights.value()
        return values.unsafe_get(row * values.strides.unsafe_get(0))
    return 1.0


def _check_finite_pair(truth: Float64, prediction: Float64) raises:
    if not (isfinite(truth) and isfinite(prediction)):
        raise Error("ValueError: y_true and y_pred must be finite")


@fieldwise_init
struct _WeightedResidual(Comparable, Copyable, Movable):
    var error: Float64
    var row: Int
    var weight: Float64
    var cumulative: Float64

    def __eq__(self, other: Self) -> Bool:
        return self.error == other.error and self.row == other.row

    def __lt__(self, other: Self) -> Bool:
        return self.error < other.error or (self.error == other.error and self.row < other.row)


def _cdf_search(values: List[_WeightedResidual], rank: Float64, upper: Bool = False) -> Int:
    var low = 0
    var high = len(values)
    while low < high:
        var middle = low + (high - low) // 2
        var cumulative = values[middle].cumulative
        if cumulative < rank or (upper and cumulative == rank):
            low = middle + 1
        else:
            high = middle
    return low


def _median_error[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype],
    column: Int, weights: _Weights,
) raises -> Float64:
    var rows = y_true.shape[0]
    if not weights:
        var errors = List[Float64](capacity=rows)
        for row in range(rows):
            var truth = _entry(y_true, row, column)
            var prediction = _entry(y_pred, row, column)
            _check_finite_pair(truth, prediction)
            errors.append(abs(truth - prediction))
        sort(Span(errors))
        if rows % 2 != 0:
            return errors[rows // 2]
        return (errors[rows // 2 - 1] + errors[rows // 2]) / 2.0

    var weighted = List[_WeightedResidual](capacity=rows)
    for row in range(rows):
        var truth = _entry(y_true, row, column)
        var prediction = _entry(y_pred, row, column)
        _check_finite_pair(truth, prediction)
        weighted.append(
            _WeightedResidual(abs(truth - prediction), row, _weight_at(weights, row), 0.0)
        )
    sort(Span(weighted))
    var total: Float64 = 0.0
    for ref residual in weighted:
        total += residual.weight
        residual.cumulative = total

    # sklearn uses a left search and skips leading zero weights at rank zero.
    var rank = total / 2.0
    if rank == 0.0:
        rank = bitcast[DType.float64](UInt64(1))
    var index = min(_cdf_search(weighted, rank), rows - 1)
    var cumulative = weighted[index].cumulative
    if cumulative - rank > _EPSILON:
        return weighted[index].error

    var next = min(index + 1, rows - 1)
    if weighted[next].weight == 0.0:
        var after = _cdf_search(weighted, cumulative, upper=True)
        next = after if after < rows else index
    return (weighted[index].error + weighted[next].error) / 2.0


def _ratio_score(numerator: Float64, denominator: Float64, force_finite: Bool) -> Float64:
    if force_finite:
        if numerator == 0.0:
            return 1.0
        if denominator == 0.0:
            return 0.0
    return 1.0 - numerator / denominator


def _column_score[
    actual_dtype: DType, predicted_dtype: DType, //, metric: StaticString,
](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], column: Int,
    weights: _Weights, mass: Float64, force_finite: Bool,
) raises -> Tuple[Float64, Float64]:
    comptime if metric == "median":
        return (_median_error(y_true, y_pred, column, weights), 0.0)

    var rows = y_true.shape[0]
    var loss: Float64 = 0.0
    var truth_sum: Float64 = 0.0
    var residual_sum: Float64 = 0.0
    for row in range(rows):
        var truth = _entry(y_true, row, column)
        var prediction = _entry(y_pred, row, column)
        _check_finite_pair(truth, prediction)
        var residual = truth - prediction
        var weight = _weight_at(weights, row)
        comptime if metric == "max":
            loss = max(loss, abs(residual))
        elif metric == "mae":
            loss += weight * abs(residual)
        elif metric != "explained_variance":
            loss += weight * (residual * residual)
        comptime if metric == "r2" or metric == "explained_variance":
            truth_sum += weight * truth
        comptime if metric == "explained_variance":
            residual_sum += weight * residual

    comptime if metric == "max":
        return (loss, 0.0)
    elif metric == "r2" or metric == "explained_variance":
        var truth_mean = truth_sum / mass
        var residual_mean = residual_sum / mass
        var variance: Float64 = 0.0
        for row in range(rows):
            var truth = _entry(y_true, row, column)
            var centered = truth - truth_mean
            var weight = _weight_at(weights, row)
            variance += weight * (centered * centered)
            comptime if metric == "explained_variance":
                var residual = truth - _entry(y_pred, row, column) - residual_mean
                loss += weight * (residual * residual)
        comptime if metric == "explained_variance":
            loss /= mass
            variance /= mass
        return (_ratio_score(loss, variance, force_finite), variance)
    elif metric == "rmse":
        return (sqrt(loss / mass), 0.0)
    else:
        return (loss / mass, 0.0)


def _regression[
    actual_dtype: DType, predicted_dtype: DType, //, metric: StaticString, raw: Bool = False,
](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype],
    sample_weight: _Weights, output_weight: _Weights = None,
    multioutput: String = "uniform_average", force_finite: Bool = True,
) raises -> _RegressionResult:
    var columns = _check_targets(y_true, y_pred)
    var rows = y_true.shape[0]
    _validate_weights(sample_weight, rows)
    if output_weight:
        if columns == 1:
            raise Error("ValueError: custom weights require multiple outputs")
        ref weights = output_weight.value()
        if weights.ndim != 1 or weights.shape[0] != columns:
            raise Error("ValueError: output_weight must have one value per output")
        for column in range(columns):
            if not isfinite(_weight_at(output_weight, column)):
                raise Error("ValueError: output_weight must be finite")
    if multioutput != "uniform_average" and multioutput != "variance_weighted":
        raise Error("ValueError: multioutput must be uniform_average or variance_weighted")
    if multioutput == "variance_weighted" and output_weight:
        raise Error("ValueError: output_weight cannot be combined with variance_weighted")

    comptime if metric == "r2":
        if rows < 2:
            for column in range(columns):
                _check_finite_pair(_entry(y_true, 0, column), _entry(y_pred, 0, column))
            print(
                "UndefinedMetricWarning: R^2 score is not well-defined with less than two samples.",
                file=stderr,
            )
            comptime if raw:
                var undefined = _Output(Shape(columns))
                for column in range(columns):
                    undefined.unsafe_set(column, nan[DType.float64]())
                return _RegressionResult(0.0, undefined^)
            else:
                return _RegressionResult(nan[DType.float64](), None)

    var mass = Float64(rows)
    comptime if metric != "median":
        if sample_weight:
            mass = 0.0
            for row in range(rows):
                mass += _weight_at(sample_weight, row)
            if mass == 0.0:
                raise Error("ZeroDivisionError: sample weights sum to zero")

    var result = Optional[_Output]()
    comptime if raw:
        result = _Output(Shape(columns))
    var sum_scores: Float64 = 0.0
    var sum_weighted: Float64 = 0.0
    var sum_weights: Float64 = 0.0
    var nonzero_variance = False
    for column in range(columns):
        var score, variance = _column_score[metric](
            y_true, y_pred, column, sample_weight, mass, force_finite,
        )
        comptime if raw:
            result.value().unsafe_set(column, score)
        else:
            sum_scores += score
            if output_weight or multioutput == "variance_weighted":
                var weight = _weight_at(output_weight, column) if output_weight else variance
                sum_weighted += weight * score
                sum_weights += weight
                nonzero_variance = nonzero_variance or variance != 0.0
    comptime if raw:
        return _RegressionResult(0.0, result^)
    else:
        if output_weight or (multioutput == "variance_weighted" and nonzero_variance):
            if sum_weights == 0.0:
                raise Error("ZeroDivisionError: output weights sum to zero")
            return _RegressionResult(sum_weighted / sum_weights, None)
        return _RegressionResult(sum_scores / Float64(columns), None)


def mean_absolute_error[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
    output_weight: _Weights = None,
) raises -> Float64:
    """Mean absolute residual."""
    return _regression["mae"](
        y_true, y_pred, sample_weight, output_weight=output_weight,
    ).average


def mean_absolute_error_raw[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
) raises -> _Output:
    """Return one value per output."""
    var result = _regression["mae", raw=True](
        y_true, y_pred, sample_weight,
    )
    return result.per_output.take()


def median_absolute_error[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
    output_weight: _Weights = None,
) raises -> Float64:
    """Median absolute residual, with averaged inverse CDF for weighted samples."""
    return _regression["median"](
        y_true, y_pred, sample_weight, output_weight=output_weight,
    ).average


def median_absolute_error_raw[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
) raises -> _Output:
    """Return one value per output."""
    var result = _regression["median", raw=True](
        y_true, y_pred, sample_weight,
    )
    return result.per_output.take()


def mean_squared_error[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
    output_weight: _Weights = None,
) raises -> Float64:
    """Mean squared residual."""
    return _regression["mse"](
        y_true, y_pred, sample_weight, output_weight=output_weight,
    ).average


def mean_squared_error_raw[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
) raises -> _Output:
    """Return one value per output."""
    var result = _regression["mse", raw=True](
        y_true, y_pred, sample_weight,
    )
    return result.per_output.take()


def root_mean_squared_error[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
    output_weight: _Weights = None,
) raises -> Float64:
    """Root mean squared residual, taking the root per output before averaging."""
    return _regression["rmse"](
        y_true, y_pred, sample_weight, output_weight=output_weight,
    ).average


def root_mean_squared_error_raw[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
) raises -> _Output:
    """Return one value per output."""
    var result = _regression["rmse", raw=True](
        y_true, y_pred, sample_weight,
    )
    return result.per_output.take()


def r2_score[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
    output_weight: _Weights = None,
    multioutput: String = "uniform_average",
    force_finite: Bool = True,
) raises -> Float64:
    """Coefficient of determination."""
    return _regression["r2"](
        y_true, y_pred, sample_weight, output_weight=output_weight,
        multioutput=multioutput, force_finite=force_finite,
    ).average


def r2_score_raw[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
    force_finite: Bool = True,
) raises -> _Output:
    """Return one value per output."""
    var result = _regression["r2", raw=True](
        y_true, y_pred, sample_weight, force_finite=force_finite,
    )
    return result.per_output.take()


def explained_variance_score[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
    output_weight: _Weights = None,
    multioutput: String = "uniform_average",
    force_finite: Bool = True,
) raises -> Float64:
    """Fraction of population variance explained."""
    return _regression["explained_variance"](
        y_true, y_pred, sample_weight, output_weight=output_weight,
        multioutput=multioutput, force_finite=force_finite,
    ).average


def explained_variance_score_raw[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype], *,
    sample_weight: _Weights = None,
    force_finite: Bool = True,
) raises -> _Output:
    """Return one value per output."""
    var result = _regression["explained_variance", raw=True](
        y_true, y_pred, sample_weight, force_finite=force_finite,
    )
    return result.per_output.take()


def max_error[actual_dtype: DType, predicted_dtype: DType](
    y_true: NDArray[actual_dtype], y_pred: NDArray[predicted_dtype]
) raises -> Float64:
    """Maximum absolute residual for a single output."""
    if _check_targets(y_true, y_pred) != 1:
        raise Error("ValueError: max_error requires a single output")
    return _column_score["max"](y_true, y_pred, 0, None, 1.0, True)[0]
