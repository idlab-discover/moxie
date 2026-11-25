"""Ownership, layout, and scheduling helpers for dense multioutput estimators."""
from std.sys import num_logical_cores

from max.algorithm import parallelize
import numojo as nm

from moxie.base import BaseEstimator, clone


def validate_targets[dtype: DType](y: nm.NDArray[dtype], rows: Int) raises:
    if y.ndim != 2:
        raise Error("ValueError: multioutput targets must have rank 2")
    if rows <= 0 or y.shape[0] != rows or y.shape[1] <= 0:
        raise Error("ValueError: nonempty targets must match sample count")


def clone_outputs[E: BaseEstimator & Deinitable](estimator: E, count: Int) raises -> List[E]:
    if count <= 0:
        raise Error("ValueError: expected at least one output")
    var result = List[E]()
    for _ in range(count):
        result.append(clone(estimator))
    return result^


def output_workers(n_jobs: Optional[Int], outputs: Int) raises -> Int:
    """Resolve joblib-style worker counts against native logical cores."""
    var jobs = n_jobs.or_else(1)
    if jobs == 0:
        raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
    if jobs < 0:
        jobs = max(1, num_logical_cores() + 1 + jobs)
    return min(outputs, jobs)


def fit_outputs[E: Movable, F: def(mut E, Int) raises -> None](
    mut models: List[E], fit_one: F, workers: Int,
) raises:
    """Fit disjoint models and report the first error in column order."""
    if workers <= 1:
        for column in range(len(models)):
            fit_one(models[column], column)
        return

    var errors = List[Optional[Error]](length=len(models), fill=None)
    var model_ptr = models.unsafe_ptr()
    var error_ptr = errors.unsafe_ptr()

    # MAX joins before either list can move. Each task has exclusive access
    # to its own model and error slot, and shared inputs are borrowed.
    def work(column: Int) {model_ptr, error_ptr, fit_one}:
        try:
            fit_one(model_ptr[unsafe_offset=column], column)
        except error:
            error_ptr[unsafe_offset=column] = error^

    parallelize(work, len(models), workers)
    for column in range(len(errors)):
        if errors[column]:
            raise errors[column].take()


def partial_fit_outputs[
    E: BaseEstimator & Deinitable, F: def(mut E, Int) raises -> None
](
    mut models: List[E], estimator: E, first_fit: Bool, outputs: Int,
    fit_one: F, workers: Int,
) raises:
    """Publish first-batch models only after every component has fitted."""
    if first_fit:
        var fresh = clone_outputs(estimator, outputs)
        fit_outputs(fresh, fit_one, workers)
        models = fresh^
    else:
        fit_outputs(models, fit_one, workers)


struct ChainOrder(Copyable, Movable):
    """Private owned permutation; position maps to original output column."""
    var _order: List[Int]
    var _inverse: List[Int]

    def __init__(out self, order: List[Int], outputs: Int) raises:
        if outputs <= 0 or len(order) != outputs:
            raise Error("ValueError: invalid order length")
        self._order = order.copy()
        self._inverse = List[Int]()
        for _ in range(outputs):
            self._inverse.append(-1)
        for j in range(outputs):
            var column = order[j]
            if column < 0 or column >= outputs:
                raise Error("ValueError: invalid order index")
            if self._inverse[column] != -1:
                raise Error("ValueError: duplicate order index")
            self._inverse[column] = j

    def column(self, position: Int) raises -> Int:
        if position < 0 or position >= len(self._order):
            raise Error("ValueError: invalid chain position")
        return self._order[position]

    def position(self, column: Int) raises -> Int:
        if column < 0 or column >= len(self._inverse):
            raise Error("ValueError: invalid output column")
        return self._inverse[column]


def target_column[dtype: DType](y: nm.NDArray[dtype], column: Int) raises -> nm.NDArray[dtype]:
    if y.ndim != 2:
        raise Error("ValueError: expected matrix")
    if column < 0 or column >= y.shape[1]:
        raise Error("ValueError: invalid output column")
    var result = nm.zeros[dtype](nm.Shape(y.shape[0]))
    for i in range(y.shape[0]):
        result.unsafe_set(i, y.unsafe_get(i * Int(y.strides[0]) + column * Int(y.strides[1])))
    return result^


def stack_predictions[dtype: DType](columns: List[nm.NDArray[dtype]]) raises -> nm.NDArray[dtype]:
    if len(columns) == 0:
        raise Error("ValueError: expected output columns")
    if columns[0].ndim != 1:
        raise Error("ValueError: predictions must be vectors")
    var rows = columns[0].shape[0]
    for j in range(len(columns)):
        if columns[j].ndim != 1:
            raise Error("ValueError: predictions must be vectors")
        if columns[j].shape[0] != rows:
            raise Error("ValueError: prediction sample counts differ")
    var result = nm.zeros[dtype](nm.Shape(rows, len(columns)))
    for i in range(rows):
        for j in range(len(columns)):
            result.unsafe_set(i * len(columns) + j, columns[j].unsafe_get(i * Int(columns[j].strides[0])))
    return result^


def augment_features[dtype: DType, input_dtype: DType](
    x: nm.NDArray[input_dtype], responses: nm.NDArray[dtype], prefix: Int,
) raises -> nm.NDArray[dtype]:
    if x.ndim != 2 or responses.ndim != 2:
        raise Error("ValueError: expected feature and response matrices")
    if x.shape[0] != responses.shape[0] or prefix < 0 or prefix > responses.shape[1]:
        raise Error("ValueError: invalid response prefix")
    var width = x.shape[1] + prefix
    var result = nm.zeros[dtype](nm.Shape(x.shape[0], width))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            var value = x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))
            result.unsafe_set(i * width + j, value.cast[dtype]())
        for j in range(prefix):
            result.unsafe_set(i * width + x.shape[1] + j, responses.unsafe_get(i * Int(responses.strides[0]) + j * Int(responses.strides[1])))
    return result^


trait ChainResponses(Movable):
    """Explicit out-of-fold response provider supplied by the caller.

    Return an owned vector in original sample order, with no training-fold
    leakage. The chain owner resolves the response method before dispatch.
    """
    comptime Estimator: BaseEstimator
    comptime Input: Movable
    comptime Target: Movable
    comptime Response: Movable

    def cross_val_response(self, estimator: Self.Estimator, X: Self.Input,
                          y: Self.Target, method: String) raises -> Self.Response:
        ...
