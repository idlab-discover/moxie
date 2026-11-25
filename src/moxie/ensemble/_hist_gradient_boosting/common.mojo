# scikit-learn developers, BSD-3-Clause.
# See families/ensemble/LICENSES/scikit-learn.txt.
"""Fit utilities for dense histogram boosting estimators."""

from std.math import isfinite
from std.utils import Variant

from numojo import NDArray, Shape

from moxie._random import check_random_state

from .feature_sampling import _FeatureSampler


comptime _Values = NDArray[DType.float64]
comptime _EarlyStopping = Variant[Bool, String]
comptime _ValidationSize = Variant[Int, Float64]
comptime _Interactions = Variant[String, List[List[Int]]]


def _uses_early_stopping(
    early_stopping: _EarlyStopping, n_samples: Int
) raises -> Bool:
    if early_stopping.isa[Bool]():
        return early_stopping[Bool]
    if early_stopping[String] != "auto":
        raise Error("ValueError: early_stopping must be true, false, or auto")
    return n_samples > 10_000


def _check_settings(
    max_iter: Int,
    max_depth: Optional[Int],
    max_leaf_nodes: Optional[Int],
    learning_rate: Float64,
    max_features: Float64,
    max_bins: Int,
    n_iter_no_change: Int,
    tol: Float64,
    verbose: Int,
    validation_fraction: Optional[_ValidationSize],
    categorical_features: Optional[String],
) raises:
    if (
        max_iter < 1
        or max_depth.or_else(1) < 1
        or max_leaf_nodes.or_else(2) < 2
    ):
        raise Error("ValueError: invalid iteration, depth, or leaf limit")
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise Error("ValueError: learning_rate must be positive and finite")
    if not (0.0 < max_features <= 1.0) or not (2 <= max_bins <= 255):
        raise Error(
            "ValueError: max_features must be in (0, 1] and max_bins in [2,"
            " 255]"
        )
    if n_iter_no_change < 1 or not isfinite(tol) or tol < 0 or verbose < 0:
        raise Error(
            "ValueError: invalid early stopping tolerance, patience, or"
            " verbosity"
        )
    if validation_fraction:
        ref size = validation_fraction.value()
        if size.isa[Int]():
            if size[Int] < 1:
                raise Error("ValueError: validation row count must be positive")
        elif not (0.0 < size[Float64] < 1.0):
            raise Error("ValueError: validation fraction must be in (0, 1)")
    if categorical_features and categorical_features.value() != "from_dtype":
        raise Error(
            "ValueError: categorical_features must be from_dtype or None on"
            " numeric arrays"
        )


def _interaction_groups(
    interaction_cst: Optional[_Interactions], n_features: Int
) raises -> List[List[Int]]:
    var groups = List[List[Int]]()
    if not interaction_cst:
        return groups^
    ref constraints = interaction_cst.value()
    if constraints.isa[String]():
        if constraints[String] == "no_interactions":
            for i in range(n_features):
                groups.append([i])
        elif constraints[String] == "pairwise":
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    groups.append([i, j])
        else:
            raise Error("ValueError: unknown interaction constraint")
        return groups^
    groups = constraints[List[List[Int]]].copy()
    var listed = List[Bool](length=n_features, fill=False)
    for group in groups:
        for feature in group:
            if not (0 <= feature < n_features):
                raise Error(
                    "ValueError: interaction feature index is out of range"
                )
            listed[feature] = True
    var remaining = List[Int]()
    for i in range(n_features):
        if not listed[i]:
            remaining.append(i)
    if remaining:
        groups.append(remaining^)
    return groups^


def _take_rows[
    dtype: DType
](
    data: NDArray[dtype], indices: NDArray[DType.int64], start: Int, stop: Int
) raises -> NDArray[dtype]:
    var count = stop - start
    if data.ndim == 1:
        var vector = NDArray[dtype](Shape(count))
        for i in range(count):
            var row = Int(indices.unsafe_get(start + i))
            vector.unsafe_set(i, data.unsafe_get(row * data.strides[0]))
        return vector^

    var columns = data.shape[1]
    var matrix = NDArray[dtype](Shape(count, columns))
    for i in range(count):
        var row = Int(indices.unsafe_get(start + i))
        for j in range(columns):
            matrix.unsafe_set(
                i * columns + j,
                data.unsafe_get(row * data.strides[0] + j * data.strides[1]),
            )
    return matrix^


def _should_stop(scores: List[Float64], patience: Int, tol: Float64) -> Bool:
    if len(scores) <= patience:
        return False
    var start = len(scores) - patience
    var threshold = scores[start - 1] + tol
    for i in range(start, len(scores)):
        if scores[i] > threshold:
            return False
    return True


def _history(scores: List[Float64]) raises -> _Values:
    var result = _Values(Shape(len(scores)))
    for i in range(len(scores)):
        result.unsafe_set(i, scores[i])
    return result^


@fieldwise_init
struct _FitRandomState(Copyable, Movable):
    var split_seed: Int
    var feature_sampler: _FeatureSampler

    @staticmethod
    def start(random_state: Optional[Int]) raises -> Self:
        with check_random_state(random_state) as rng:
            var split_seed = rng[].randint(4294967295)
            var feature_seed = UInt32(rng[].randint(4294967295))
            return Self(split_seed, _FeatureSampler(feature_seed))


def _resume_history(scores: _Values) raises -> List[Float64]:
    if scores.ndim != 1:
        raise Error("ValueError: warm start requires rank-one score histories")
    var stride = scores.strides[0]
    var result = List[Float64](capacity=scores.size)
    for i in range(scores.size):
        result.append(scores.unsafe_get(i * stride))
    return result^


def _check_warm_start(
    max_iter: Int, n_iter: Int, n_features: Int, fitted_features: Int
) raises:
    if max_iter < n_iter:
        raise Error(
            t"ValueError: max_iter={max_iter} must be at least the fitted"
            t" n_iter_={n_iter} when warm_start is true"
        )
    if n_features != fitted_features:
        raise Error("ValueError: warm start requires the fitted feature count")
