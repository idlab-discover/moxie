# scikit-learn developers, BSD-3-Clause; see families/ensemble/LICENSES/scikit-learn.txt.
# Quantile interpolation follows NumPy; see families/ensemble/LICENSES/NumPy.txt.
"""Fit byte-sized bins for dense histogram gradient boosting."""
from std.builtin.sort import sort
from std.collections import Array
from std.math import floor, isfinite, isnan, nan

from numojo import NDArray, Shape
from moxie._random import check_random_state
from moxie.ensemble._sampling import check_sample_weight, normalized_weights

comptime _ALMOST_INF = Float64(1e300)
comptime _EPSILON = Float64(2.220446049250313e-16)
comptime _Weights = NDArray[DType.float64]
comptime _RowIndices = NDArray[DType.int64]
comptime _Categories = List[Optional[List[Float64]]]
comptime _CategoryBits = Array[UInt32, 8]


@fieldwise_init
struct _Observation(Comparable, Copyable, Movable):
    var value: Float64
    var weight: Float64

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __lt__(self, other: Self) -> Bool:
        return self.value < other.value


def _unweighted_quantile(values: List[_Observation], fraction: Float64) -> Float64:
    # NumPy's averaged_inverted_cdf interpolates from the upper order
    # statistic. Preserve its rounding and infinity behavior.
    var index = Float64(len(values)) * fraction - 1.0
    var below = Int(floor(index))
    var last = len(values) - 1
    var a = values[min(max(below, 0), last)].value
    var b = values[min(max(below + 1, 0), last)].value
    var averaged = index >= 0 and index < Float64(last) and index == Float64(below)
    return b - (b - a) * (0.5 if averaged else 0.0)


def _weighted_quantile(values: List[_Observation], fraction: Float64) -> Float64:
    var last = len(values) - 1
    var rank = fraction * values[last].weight
    var low = 0
    var high = last
    while low < high:
        var middle = low + (high - low) // 2
        if values[middle].weight < rank:
            low = middle + 1
        else:
            high = middle
    var value = values[low].value
    if values[low].weight - rank > _EPSILON:
        return value
    return (value + values[min(low + 1, last)].value) / 2.0


def _thresholds(var values: List[_Observation], max_bins: Int, weighted: Bool) raises -> List[Float64]:
    var thresholds = List[Float64](capacity=max_bins - 1)
    if not values:
        return thresholds^

    sort(Span(values))
    var previous = values[0].value
    var distinct = 1
    for i in range(1, len(values)):
        var value = values[i].value
        if value != previous:
            distinct += 1
            if distinct > max_bins:
                break
            thresholds.append((previous + value) / 2.0)
            previous = value

    if distinct > max_bins:
        thresholds.clear()
        if weighted:
            var cumulative = Float64(0)
            for i in range(len(values)):
                cumulative += values[i].weight
                values[i].weight = cumulative
            if not isfinite(cumulative):
                raise Error("ValueError: cumulative sample_weight must be finite")
        var step = 100.0 / Float64(max_bins)
        for i in range(1, max_bins):
            var fraction = (Float64(i) * step) / 100.0
            thresholds.append(
                _weighted_quantile(values, fraction) if weighted else _unweighted_quantile(values, fraction)
            )

    # NumPy unique sorts, keeping one NaN at the end. Deduplicate before
    # clipping, since distinct thresholds may clip to the same finite value.
    var ordered = List[Float64](capacity=len(thresholds))
    var has_nan = False
    for value in thresholds:
        if isnan(value):
            has_nan = True
        else:
            ordered.append(value)
    sort(Span(ordered))
    var unique = List[Float64](capacity=len(thresholds))
    for value in ordered:
        if not unique or value != unique[len(unique) - 1]:
            unique.append(value)
    if has_nan:
        unique.append(nan[DType.float64]())
    for i in range(len(unique)):
        if unique[i] > _ALMOST_INF:
            unique[i] = _ALMOST_INF
    return unique^


def _numeric_bins[dtype: DType](
    X: NDArray[dtype],
    feature: Int,
    max_bins: Int,
    weights: Optional[_Weights],
    subset: Optional[_RowIndices],
) raises -> List[Float64]:
    var count = subset.value().size if subset else X.shape[0]
    var values = List[_Observation](capacity=count)
    var row_stride = X.strides[0]
    var col_offset = feature * X.strides[1]
    for i in range(count):
        var row = Int(subset.value().unsafe_get(i)) if subset else i
        var value = Float64(X.unsafe_get(row * row_stride + col_offset))
        if isnan(value):
            continue
        var weight = weights.value().unsafe_get(row * weights.value().strides[0]) if weights else Float64(1)
        if weight != 0:
            values.append(_Observation(value, weight))
    return _thresholds(values^, max_bins, Bool(weights))


def _bin(value: Float64, thresholds: List[Float64], categorical: Bool, missing: UInt8) -> UInt8:
    if isnan(value) or (categorical and value < 0):
        return missing
    var low = 0
    var high = len(thresholds)
    while low < high:
        var middle = low + (high - low) // 2
        if thresholds[middle] < value:
            low = middle + 1
        else:
            high = middle
    return UInt8(low)


struct _BinMapper(Copyable, Movable):
    """Private transformer with one category domain per categorical column."""

    var n_bins: Int
    var subsample: Optional[Int]
    var random_state: Optional[Int]
    var bin_thresholds_: List[List[Float64]]
    var n_bins_non_missing_: List[UInt32]
    var is_categorical_: List[Bool]
    var missing_values_bin_idx_: UInt8

    def __init__(
        out self,
        n_bins: Int = 256,
        subsample: Optional[Int] = 200_000,
        random_state: Optional[Int] = None,
    ):
        self.n_bins = n_bins
        self.subsample = subsample
        self.random_state = random_state
        self.bin_thresholds_ = List[List[Float64]]()
        self.n_bins_non_missing_ = List[UInt32]()
        self.is_categorical_ = List[Bool]()
        self.missing_values_bin_idx_ = 0

    def fit[dtype: DType, origin: MutOrigin](
        ref[origin] self,
        X: NDArray[dtype],
        *,
        known_categories: Optional[_Categories] = None,
        sample_weight: Optional[_Weights] = None,
    ) raises -> ref[self] Self:
        comptime assert dtype in (DType.float32, DType.float64), "binning requires Float32 or Float64"
        self._check_matrix(X)
        if self.n_bins < 3 or self.n_bins > 256:
            raise Error("ValueError: n_bins must be between 3 and 256")
        if self.subsample and self.subsample.value() < 1:
            raise Error("ValueError: subsample must be positive")
        var n_rows = X.shape[0]
        var n_features = X.shape[1]
        var max_bins = self.n_bins - 1
        if known_categories and len(known_categories.value()) != n_features:
            raise Error("ValueError: known_categories must have one entry per feature")
        if sample_weight:
            ref weights = sample_weight.value()
            check_sample_weight(weights, n_rows)
            for i in range(n_rows):
                if weights.unsafe_get(i * weights.strides[0]) < 0:
                    raise Error("ValueError: sample_weight must be nonnegative")

        var categorical = List[Bool](length=n_features, fill=False)
        var thresholds = List[List[Float64]](length=n_features, fill=List[Float64]())
        for feature in range(n_features):
            if not known_categories or not known_categories.value()[feature]:
                continue
            ref categories = known_categories.value()[feature].value()
            if len(categories) > max_bins:
                raise Error("ValueError: categorical cardinality exceeds max_bins")
            var previous = Float64(-1)
            for value in categories:
                if not (value > previous and value <= 254 and floor(value) == value):
                    raise Error("ValueError: categories must be sorted distinct integers in [0, 254]")
                previous = value
            categorical[feature] = True
            thresholds[feature] = categories.copy()

        var subset: Optional[_RowIndices] = None
        var count = self.subsample.or_else(n_rows)
        with check_random_state(self.random_state) as rng:
            if count < n_rows:
                if sample_weight:
                    var probabilities = normalized_weights(sample_weight.value(), n_rows)
                    subset = rng[].choice(n_rows, probabilities, size=count)
                else:
                    subset = rng[].choice(n_rows, size=count)

        var counts = List[UInt32](capacity=n_features)
        var no_weights: Optional[_Weights] = None
        for feature in range(n_features):
            if categorical[feature]:
                counts.append(UInt32(len(thresholds[feature])))
                continue
            if subset:
                thresholds[feature] = _numeric_bins(X, feature, max_bins, no_weights, subset)
            else:
                thresholds[feature] = _numeric_bins(X, feature, max_bins, sample_weight, subset)
            counts.append(UInt32(len(thresholds[feature]) + 1))

        self.bin_thresholds_ = thresholds^
        self.n_bins_non_missing_ = counts^
        self.is_categorical_ = categorical^
        self.missing_values_bin_idx_ = UInt8(max_bins)
        return self

    def transform[dtype: DType](self, X: NDArray[dtype]) raises -> NDArray[DType.uint8]:
        comptime assert dtype in (DType.float32, DType.float64), "binning requires Float32 or Float64"
        self._check_matrix(X)
        self._require_fitted()
        if X.shape[1] != len(self.bin_thresholds_):
            raise Error("ValueError: transform requires the fitted feature count")

        var binned = NDArray[DType.uint8](Shape(X.shape[0], X.shape[1]), order="F")
        for feature in range(X.shape[1]):
            var offset = feature * X.strides[1]
            var out_offset = feature * binned.strides[1]
            for row in range(X.shape[0]):
                var value = Float64(X.unsafe_get(offset + row * X.strides[0]))
                binned.unsafe_set(
                    out_offset + row,
                    _bin(value, self.bin_thresholds_[feature], self.is_categorical_[feature], self.missing_values_bin_idx_),
                )
        return binned^

    def make_known_categories_bitsets(self) raises -> Tuple[List[_CategoryBits], List[UInt32]]:
        self._require_fitted()
        var bits = List[_CategoryBits]()
        var feature_map = List[UInt32](length=len(self.is_categorical_), fill=0)
        for feature in range(len(self.is_categorical_)):
            if not self.is_categorical_[feature]:
                continue
            feature_map[feature] = UInt32(len(bits))
            var words = _CategoryBits(fill=0)
            for value in self.bin_thresholds_[feature]:
                var category = Int(value)
                words[category // 32] |= UInt32(1) << UInt32(category % 32)
            bits.append(words^)
        return (bits^, feature_map^)

    def _require_fitted(self) raises:
        if not self.bin_thresholds_:
            raise Error("NotFittedError: _BinMapper must be fitted")

    @staticmethod
    def _check_matrix[dtype: DType](X: NDArray[dtype]) raises:
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise Error("ValueError: X must be a nonempty rank-2 matrix")
