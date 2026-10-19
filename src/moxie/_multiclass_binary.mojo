# Adapted from scikit-learn multiclass.py; BSD-3-Clause, see LICENSES/multiclass-scikit-learn.txt.
"""Binary-component helpers shared by the multiclass meta-estimators."""
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie._labels import StringTarget
from max.algorithm import parallelize
from std.sys import num_logical_cores
from moxie.base import DecisionFunction, Predictor, ProbabilityPredictor, RegressorMixin

comptime BinaryTarget = NDArray[DType.int64]
"""Component targets: 0/1 for OvR/OvO, or -1/+1 for margin-based output codes."""


def n_workers(n_jobs: Optional[Int], n_tasks: Int) raises -> Int:
    """joblib's effective `Parallel(n_jobs)` worker count, capped at `n_tasks`.

    None means 1; negative values count back from the logical core count
    (`-1` is every core) and never drop below 1; 0 raises, as in joblib.
    """
    var jobs = n_jobs.value() if n_jobs else 1
    if jobs == 0:
        raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
    if jobs < 0:
        jobs = max(1, num_logical_cores() + 1 + jobs)
    return min(max(1, n_tasks), jobs)


def fit_components[
    T: Copyable & Deinitable, F: def(Int) raises -> T
](fit_one: F, n_tasks: Int, workers: Int) raises -> List[T]:
    """Return `[fit_one(0), ..., fit_one(n_tasks - 1)]`, on `workers` MAX CPU threads.

    Tasks must be independent. One worker runs them in order on the calling
    thread and stops at the first error. Several workers run every task, then
    re-raise the lowest-index error, so the reported failure matches the
    sequential order.
    """
    var fitted = List[T]()
    if workers <= 1:
        for t in range(n_tasks):
            fitted.append(fit_one(t))
        return fitted^
    var slots = List[Optional[T]](length=n_tasks, fill=None)
    var errors = List[Optional[String]](length=n_tasks, fill=None)
    var slot_ptr = slots.unsafe_ptr()
    var error_ptr = errors.unsafe_ptr()

    # parallelize aborts on a raising task, so each task records its own error.
    def work(t: Int) {slot_ptr, error_ptr, fit_one}:
        try:
            slot_ptr[unsafe_offset=t] = fit_one(t)
        except error:
            error_ptr[unsafe_offset=t] = String(error)

    parallelize(work, n_tasks, workers)
    for t in range(n_tasks):
        if errors[t]:
            raise Error(errors[t].value())
        fitted.append(slots[t].take())
    return fitted^


def float_vector[dtype: DType, rank_one: Bool = False](values: NDArray[dtype], n_samples: Int) raises -> NDArray[DType.float64]:
    """Ravel decision values, or validate a rank-one regressor prediction."""
    comptime if rank_one:
        if values.ndim != 1:
            raise Error("ValueError: regressor predict must have shape (n_samples,)")
    if values.size != n_samples or values.ndim > 2 or (values.ndim == 2 and values.shape[1] != 1):
        raise Error(
            "ValueError: binary decision_function must have shape (n_samples,)"
        )
    var result = nm.zeros[DType.float64](Shape(n_samples))
    for i in range(n_samples):
        if values.ndim == 1:
            result[Item(i)] = Float64(values[Item(i)])
        else:
            result[Item(i)] = Float64(values[Item(i, 0)])
    return result^


def positive_column[dtype: DType](values: NDArray[dtype], n_samples: Int) raises -> NDArray[DType.float64]:
    """Return predict_proba(X)[:, 1] as Float64."""
    if values.ndim != 2 or values.shape[0] != n_samples or values.shape[1] < 2:
        raise Error(
            "ValueError: binary predict_proba must have shape (n_samples, 2)"
        )
    var result = nm.zeros[DType.float64](Shape(n_samples))
    for i in range(n_samples):
        result[Item(i)] = Float64(values[Item(i, 1)])
    return result^


def positive_probability[E: Predictor](estimator: E, X: E.Input, n_samples: Int) raises -> NDArray[DType.float64]:
    """P(positive class) of a fitted binary component."""
    comptime if conforms_to(E, ProbabilityPredictor):
        comptime if E.Probabilities == NDArray[DType.float64]:
            return positive_column(rebind_var[NDArray[DType.float64]](estimator.predict_proba(X)), n_samples)
        elif E.Probabilities == NDArray[DType.float32]:
            return positive_column(rebind_var[NDArray[DType.float32]](estimator.predict_proba(X)), n_samples)
        else:
            comptime assert False, "component probabilities must be NDArray[float32/float64]"
    else:
        comptime assert False, "component needs decision_function or predict_proba"


def predict_binary[E: Predictor, rank_one: Bool = False](estimator: E, X: E.Input, n_samples: Int) raises -> NDArray[DType.float64]:
    """Scikit-learn `_predict_binary`: regressors predict, classifiers prefer decisions."""
    comptime if conforms_to(E, RegressorMixin):
        comptime if E.Prediction == NDArray[DType.float64]:
            return float_vector[DType.float64, rank_one](rebind_var[NDArray[DType.float64]](estimator.predict(X)), n_samples)
        elif E.Prediction == NDArray[DType.float32]:
            return float_vector[DType.float32, rank_one](rebind_var[NDArray[DType.float32]](estimator.predict(X)), n_samples)
        else:
            comptime assert False, "regressor predictions must be NDArray[float32/float64]"
    else:
        return binary_confidence(estimator, X, n_samples)


def binary_confidence[E: Predictor](estimator: E, X: E.Input, n_samples: Int) raises -> NDArray[DType.float64]:
    """Classifier confidence, also used for explicit decision_function requests."""
    comptime if conforms_to(E, DecisionFunction):
        comptime if E.Decision == NDArray[DType.float64]:
            return float_vector(rebind_var[NDArray[DType.float64]](estimator.decision_function(X)), n_samples)
        elif E.Decision == NDArray[DType.float32]:
            return float_vector(rebind_var[NDArray[DType.float32]](estimator.decision_function(X)), n_samples)
        else:
            comptime assert False, "component decision values must be NDArray[float32/float64]"
    else:
        return positive_probability(estimator, X, n_samples)


def binary_threshold[E: Predictor]() -> Float64:
    """Scikit-learn `_threshold_for_binary_predict` for a fitted classifier component."""
    comptime if conforms_to(E, DecisionFunction):
        return 0.0
    return 0.5


def mean_accuracy[label: DType](
    truth: NDArray[label],
    predicted: NDArray[label],
    sample_weight: Optional[NDArray[DType.float64]],
) raises -> Float64:
    """ClassifierMixin.score: accuracy_score, weighted by `sample_weight` when given."""
    if truth.size != predicted.size:
        raise Error(
            String("ValueError: Found input variables with inconsistent numbers of samples: [",
                   truth.size, ", ", predicted.size, "]")
        )
    var correct: Float64 = 0.0
    var total: Float64 = 0.0
    for i in range(truth.size):
        var weight: Float64 = 1.0
        if sample_weight:
            ref weights = sample_weight.value()
            if weights.ndim != 1 or weights.size != truth.size:
                raise Error(
                    "ValueError: sample_weight must have shape (n_samples,)"
                )
            weight = weights[Item(i)]
        total += weight
        if truth[Item(i)] == predicted[Item(i)]:
            correct += weight
    if total == 0.0:
        raise Error("ValueError: Weights sum to zero, can't be normalized")
    return correct / total


def mean_accuracy(
    truth: StringTarget, predicted: StringTarget,
    sample_weight: Optional[NDArray[DType.float64]],
) raises -> Float64:
    if truth.rows() != predicted.rows():
        raise Error("ValueError: Found input variables with inconsistent numbers of samples")
    var matches = nm.zeros[DType.int64](Shape(truth.rows()))
    var expected = nm.full[DType.int64](Shape(truth.rows()), 1)
    for i in range(truth.rows()):
        if truth.get(i) == predicted.get(i):
            matches[Item(i)] = 1
    return mean_accuracy(expected, matches, sample_weight)
