"""Exercise dense regression metrics and emit results for the pinned oracle."""
from std.math import inf, nan
from std.testing import assert_equal, assert_raises

from moxie.metrics import explained_variance_score, explained_variance_score_raw
from moxie.metrics import max_error, mean_absolute_error, mean_absolute_error_raw
from moxie.metrics import mean_squared_error, mean_squared_error_raw
from moxie.metrics import median_absolute_error, median_absolute_error_raw
from moxie.metrics import r2_score, r2_score_raw, root_mean_squared_error
from moxie.metrics import root_mean_squared_error_raw
from numojo import NDArray, Shape
from numojo.core.layout.ndstrides import NDArrayStrides

comptime Dense = NDArray[DType.float64]
comptime Weights = Optional[Dense]


def array[dtype: DType = DType.float64](
    values: List[Scalar[dtype]], shape: Shape
) raises -> NDArray[dtype]:
    var result = NDArray[dtype](shape)
    assert_equal(result.size, len(values))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def raw_line(name: String, scores: Dense, columns: Int) raises:
    assert_equal(scores.ndim, 1)
    assert_equal(scores.shape[0], columns)
    var line = name.copy()
    for column in range(columns):
        line += String(" ", scores.unsafe_get(column))
    print(line)


def scores[actual_dtype: DType, predicted_dtype: DType](
    name: String, truth: NDArray[actual_dtype], pred: NDArray[predicted_dtype],
    weights: Weights = None, outputs: Weights = None, force_finite: Bool = True,
) raises:
    var columns = truth.shape[1] if truth.ndim == 2 else 1
    print(
        String(name, "_aggregate"),
        mean_absolute_error(truth, pred, sample_weight=weights, output_weight=outputs),
        median_absolute_error(truth, pred, sample_weight=weights, output_weight=outputs),
        mean_squared_error(truth, pred, sample_weight=weights, output_weight=outputs),
        root_mean_squared_error(truth, pred, sample_weight=weights, output_weight=outputs),
        r2_score(truth, pred, sample_weight=weights, output_weight=outputs, force_finite=force_finite),
        explained_variance_score(
            truth, pred, sample_weight=weights, output_weight=outputs, force_finite=force_finite,
        ),
    )
    raw_line(String(name, "_mae"), mean_absolute_error_raw(truth, pred, sample_weight=weights), columns)
    raw_line(String(name, "_median"), median_absolute_error_raw(truth, pred, sample_weight=weights), columns)
    raw_line(String(name, "_mse"), mean_squared_error_raw(truth, pred, sample_weight=weights), columns)
    raw_line(String(name, "_rmse"), root_mean_squared_error_raw(truth, pred, sample_weight=weights), columns)
    raw_line(String(name, "_r2"), r2_score_raw(truth, pred, sample_weight=weights, force_finite=force_finite), columns)
    raw_line(
        String(name, "_ev"),
        explained_variance_score_raw(truth, pred, sample_weight=weights, force_finite=force_finite),
        columns,
    )
    if columns == 1:
        print(String(name, "_max"), max_error(truth, pred))
    else:
        print(
            String(name, "_variance"),
            r2_score(truth, pred, sample_weight=weights, multioutput="variance_weighted", force_finite=force_finite),
            explained_variance_score(
                truth, pred, sample_weight=weights, multioutput="variance_weighted", force_finite=force_finite,
            ),
        )


def invalid(vec: Dense, matrix: Dense) raises:
    var wrong_rows = array([0.0, 1.0], Shape(2))
    var bad_rank = array([0.0, 1.0, 2.0, 3.0], Shape(4, 1, 1))
    var bad_value = array([0.0, 1.0, 2.0, nan[DType.float64]()], Shape(4))
    var bad_weight = array([0.0, 1.0, 2.0, inf[DType.float64]()], Shape(4))
    var column_weight = array([0.0, 1.0, 2.0, 3.0], Shape(4, 1))
    var zero_weight = array([0.0, 0.0, 0.0, 0.0], Shape(4))
    var cancel_weight = array([1.0, -1.0, 0.0, 0.0], Shape(4))
    var cancel_output = array([1.0, -1.0], Shape(2))
    var empty = Dense(Shape(0))
    var no_columns = Dense(Shape(4, 0))

    with assert_raises(contains="ValueError"):
        _ = mean_absolute_error(vec, wrong_rows)
    with assert_raises(contains="ValueError"):
        _ = mean_squared_error(vec, bad_rank)
    with assert_raises(contains="ValueError"):
        _ = r2_score(vec, matrix)
    with assert_raises(contains="ValueError"):
        _ = explained_variance_score(vec, bad_value)
    with assert_raises(contains="ValueError"):
        _ = max_error(matrix, matrix)
    with assert_raises(contains="ValueError"):
        _ = median_absolute_error(empty, empty)
    with assert_raises(contains="ValueError"):
        _ = mean_absolute_error(no_columns, no_columns)
    with assert_raises(contains="ValueError"):
        _ = mean_absolute_error(vec, vec, sample_weight=wrong_rows.view())
    with assert_raises(contains="ValueError"):
        _ = mean_absolute_error(vec, vec, sample_weight=bad_weight^)
    with assert_raises(contains="ValueError"):
        _ = mean_absolute_error(vec, vec, sample_weight=column_weight^)
    with assert_raises(contains="ValueError"):
        _ = median_absolute_error(vec, vec, sample_weight=zero_weight^)
    with assert_raises(contains="ZeroDivisionError"):
        _ = mean_absolute_error(vec, vec, sample_weight=cancel_weight^)
    with assert_raises(contains="ValueError"):
        _ = mean_absolute_error(vec, vec, output_weight=wrong_rows.view())
    with assert_raises(contains="ValueError"):
        _ = mean_absolute_error(matrix, matrix, output_weight=vec.copy())
    with assert_raises(contains="ZeroDivisionError"):
        _ = mean_squared_error(matrix, matrix, output_weight=cancel_output^)
    with assert_raises(contains="ValueError"):
        _ = r2_score(matrix, matrix, multioutput="invalid")
    with assert_raises(contains="ValueError"):
        _ = r2_score(
            matrix, matrix, multioutput="variance_weighted", output_weight=array([1.0, 1.0], Shape(2)),
        )


def main() raises:
    var vector = array([3.0, -0.5, 2.0, 7.0], Shape(4))
    var pred = array([2.5, 0.0, 2.0, 8.0], Shape(4, 1))
    scores("vector", vector, pred)

    var truth = array([1.0, 8.0, 2.0, 8.0, 4.0, 8.0, 6.0, 8.0], Shape(4, 2))
    var predicted = array([1.0, 7.0, 3.0, 5.0, 2.0, 7.0, 8.0, 4.0], Shape(4, 2))
    var weights = array([1.0, 0.0, 1.0, 2.0], Shape(4))
    var outputs = array([0.25, 0.75], Shape(2))
    scores("multi", truth, predicted, weights.view(), outputs.view())
    var view = truth.view_with_layout(Shape(4, 2), NDArrayStrides(-2, -1), 7)
    var reversed_weights = weights.view_with_layout(Shape(4), NDArrayStrides(-1), 3)
    var reversed_outputs = outputs.view_with_layout(Shape(2), NDArrayStrides(-1), 1)
    scores("strided", view, predicted, reversed_weights^, reversed_outputs^)
    var signed = array([-1.0, 2.0, 0.5, 3.0], Shape(4))
    scores("signed", truth, predicted, signed^)

    var constant = array([1.0, 1.0, 1.0], Shape(3))
    var shifted = array([2.0, 2.0, 2.0], Shape(3))
    scores("constant_perfect", constant, constant, force_finite=False)
    scores("constant_shifted", constant, shifted, force_finite=False)
    scores("constant_multi", truth, predicted, force_finite=False)
    var flat = array([1.0, 4.0, 1.0, 4.0, 1.0, 4.0], Shape(3, 2))
    var flat_pred = array([1.0, 3.0, 1.0, 4.0, 1.0, 5.0], Shape(3, 2))
    scores("all_constant", flat, flat_pred)
    scores("singleton", array([1.0, 2.0], Shape(1, 2)), array([2.0, 2.0], Shape(1, 2)))

    var f32_truth = array[DType.float32]([0.2, 1.7, 2.1], Shape(3))
    var f32_pred = array[DType.float32]([0.1, 1.2, 2.6], Shape(3))
    scores("float32", f32_truth, f32_pred)
    var int_truth = array[DType.int64]([0, 1, 2], Shape(3))
    scores("mixed", int_truth, f32_pred)
    scores("integral", int_truth, array[DType.int64]([2, -1, 4], Shape(3)))

    # At the median boundary, skip the following zero-weight residual.
    var zeros = array([0.0, 0.0, 0.0, 0.0], Shape(4))
    var errors = array([1.0, 2.0, 3.0, 4.0], Shape(4))
    var gapped = array([1.0, 0.0, 1.0, 0.0], Shape(4))
    var leading = array([0.0, 1.0, 0.0, 1.0], Shape(4))
    var zero_total = array([0.0, 2.0, 1.0, -3.0], Shape(4))
    print("median_zero_total", median_absolute_error(zeros, errors, sample_weight=zero_total^))
    print("median_gap", median_absolute_error(zeros, errors, sample_weight=gapped^))
    print("median_leading", median_absolute_error(zeros, errors, sample_weight=leading^))
    invalid(vector, truth)
