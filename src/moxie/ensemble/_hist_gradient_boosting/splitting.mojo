# scikit-learn developers, BSD-3-Clause.
# See families/ensemble/LICENSES/scikit-learn.txt.
"""Select and apply splits on dense histogram gradient boosting bins."""

from std.builtin.sort import sort
from std.math import inf, isfinite, isnan

from numojo import NDArray

from .binning import _CategoryBits
from .histogram import _HistogramBin, _Histograms


comptime _MIN_CATEGORY_SUPPORT = 10.0


struct _ValueBounds(ImplicitlyCopyable, Movable):
    var lower: Float64
    var upper: Float64

    def __init__(
        out self,
        lower: Float64 = -inf[DType.float64](),
        upper: Float64 = inf[DType.float64](),
    ):
        self.lower = lower
        self.upper = upper


struct _SplitOptions(ImplicitlyCopyable, Movable):
    var l2_regularization: Float64
    var min_hessian_to_split: Float64
    var min_samples_leaf: Int
    var min_gain_to_split: Float64

    def __init__(
        out self,
        *,
        l2_regularization: Float64 = 0.0,
        min_hessian_to_split: Float64 = 1e-3,
        min_samples_leaf: Int = 20,
        min_gain_to_split: Float64 = 0.0,
    ):
        self.l2_regularization = l2_regularization
        self.min_hessian_to_split = min_hessian_to_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split

    def validate(self) raises:
        if (
            self.min_samples_leaf < 1
            or not isfinite(self.l2_regularization)
            or self.l2_regularization < 0
            or not isfinite(self.min_hessian_to_split)
            or self.min_hessian_to_split < 0
            or not isfinite(self.min_gain_to_split)
            or self.min_gain_to_split < 0
        ):
            raise Error("ValueError: invalid split options or value bounds")


@fieldwise_init
struct _Split(Copyable, Movable):
    var gain: Float64
    var feature: Int
    var bin_idx: UInt8
    var missing_go_to_left: Bool
    var left_categories: Optional[_CategoryBits]
    var left: _HistogramBin
    var right: _HistogramBin
    var value_left: Float64
    var value_right: Float64

    def goes_left(self, code: UInt8, missing_bin: UInt8) -> Bool:
        if code == missing_bin:
            return self.missing_go_to_left
        if self.left_categories:
            return _contains_category(self.left_categories.value(), code)
        return code <= self.bin_idx


def _contains_category(categories: _CategoryBits, code: UInt8) -> Bool:
    var index = Int(code)
    return (categories[index // 32] & (UInt32(1) << UInt32(index % 32))) != 0


def compute_node_value(
    stats: _HistogramBin, bounds: _ValueBounds, l2_regularization: Float64
) -> Float64:
    var denominator = stats.sum_hessians + l2_regularization + 1e-15
    var value = -stats.sum_gradients / denominator
    if value < bounds.lower:
        return bounds.lower
    if value > bounds.upper:
        return bounds.upper
    return value


@fieldwise_init
struct _Category(Comparable, Copyable, Movable):
    var code: Int
    var score: Float64

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code and self.score == other.score

    def __lt__(self, other: Self) -> Bool:
        if self.score == other.score:
            return self.code > other.code
        return self.score < other.score


@fieldwise_init
struct _SplitSearch(Movable):
    var total: _HistogramBin
    var parent_loss: Float64
    var options: _SplitOptions
    var bounds: _ValueBounds
    var unit_hessian: Bool
    var best_gain: Float64
    var best: Optional[_Split]

    def hessian(self, bin: _HistogramBin) -> Float64:
        return Float64(bin.count) if self.unit_hessian else bin.sum_hessians

    def accumulate(self, mut left: _HistogramBin, bin: _HistogramBin):
        left.count += bin.count
        left.sum_gradients += bin.sum_gradients
        left.sum_hessians += self.hessian(bin)

    def complement(self, left: _HistogramBin) -> _HistogramBin:
        return _HistogramBin(
            self.total.sum_gradients - left.sum_gradients,
            self.total.sum_hessians - left.sum_hessians,
            self.total.count - left.count,
        )

    def valid_child(self, stats: _HistogramBin) -> Bool:
        return (
            stats.count >= self.options.min_samples_leaf
            and stats.sum_hessians >= self.options.min_hessian_to_split
        )

    def consider(
        mut self,
        feature: Int,
        bin_idx: UInt8,
        missing_go_to_left: Bool,
        var categories: Optional[_CategoryBits],
        monotonic: Int,
        left: _HistogramBin,
        right: _HistogramBin,
    ):
        var l2_regularization = self.options.l2_regularization
        var value_left = compute_node_value(left, self.bounds, l2_regularization)
        var value_right = compute_node_value(right, self.bounds, l2_regularization)
        if (monotonic > 0 and value_left > value_right) or (
            monotonic < 0 and value_left < value_right
        ):
            return
        var gain = self.parent_loss - left.sum_gradients * value_left
        gain -= right.sum_gradients * value_right
        if gain > self.best_gain:
            self.best_gain = gain
            self.best = _Split(
                gain,
                feature,
                bin_idx,
                missing_go_to_left,
                categories^,
                left,
                right,
                value_left,
                value_right,
            )

    def scan_numeric(
        mut self,
        bins: List[_HistogramBin],
        feature: Int,
        n_non_missing: Int,
        missing_bin: Int,
        has_missing: Bool,
        monotonic: Int,
    ):
        for direction in range(2 if has_missing else 1):
            var missing_left = direction == 1
            var left = _HistogramBin(0.0, 0.0, 0)
            var stop = n_non_missing - 1
            if missing_left:
                self.accumulate(left, bins[missing_bin])
            elif has_missing:
                # This includes the candidate that isolates all missing rows.
                stop += 1
            for code in range(stop):
                self.accumulate(left, bins[code])
                if not self.valid_child(left):
                    continue
                var right = self.complement(left)
                if not self.valid_child(right):
                    break
                self.consider(
                    feature, UInt8(code), missing_left, None, monotonic, left, right
                )

    def scan_categorical(
        mut self,
        bins: List[_HistogramBin],
        feature: Int,
        n_non_missing: Int,
        missing_bin: Int,
        has_missing: Bool,
    ):
        var support_factor = Float64(self.total.count) / self.total.sum_hessians
        var candidates = List[_Category](capacity=n_non_missing + Int(has_missing))
        for i in range(n_non_missing + Int(has_missing)):
            var code = i if i < n_non_missing else missing_bin
            var bin = bins[code]
            var hessian = self.hessian(bin)
            if hessian * support_factor >= _MIN_CATEGORY_SUPPORT:
                var score = bin.sum_gradients / (hessian + _MIN_CATEGORY_SUPPORT)
                candidates.append(_Category(code, score))
        if len(candidates) < 2:
            return
        sort(candidates)

        # The right child always receives all low-support categories.
        var midpoint = (len(candidates) + 1) // 2
        for direction in range(2):
            var left = _HistogramBin(0.0, 0.0, 0)
            var bits = _CategoryBits(fill=0)
            for i in range(midpoint - direction):
                var index = i if direction == 0 else len(candidates) - 1 - i
                var code = candidates[index].code
                bits[code // 32] |= UInt32(1) << UInt32(code % 32)
                self.accumulate(left, bins[code])
                if not self.valid_child(left):
                    continue
                var right = self.complement(left)
                if not self.valid_child(right):
                    break
                self.consider(
                    feature,
                    0,
                    _contains_category(bits, UInt8(missing_bin)),
                    bits.copy(),
                    0,
                    left,
                    right,
                )


def find_node_split[
    features_origin: ImmOrigin, //
](
    histograms: _Histograms,
    total: _HistogramBin,
    value: Float64,
    n_bins_non_missing: List[UInt32],
    has_missing_values: List[Bool],
    is_categorical: List[Bool],
    monotonic_cst: List[Int],
    *,
    allowed_features: Span[Int, features_origin],
    bounds: _ValueBounds = _ValueBounds(),
    options: _SplitOptions = _SplitOptions(),
) raises -> Optional[_Split]:
    """Scan the supplied features in order, retaining the first strict winner."""
    var n_features = len(histograms._columns)
    if (
        len(n_bins_non_missing) != n_features
        or len(has_missing_values) != n_features
        or len(is_categorical) != n_features
        or len(monotonic_cst) != n_features
    ):
        raise Error("ValueError: split metadata must match histogram features")
    options.validate()
    if isnan(bounds.lower) or isnan(bounds.upper) or bounds.lower > bounds.upper:
        raise Error("ValueError: invalid split options or value bounds")

    var missing_bin = histograms._n_bins - 1
    var search = _SplitSearch(
        total,
        total.sum_gradients * value,
        options,
        bounds,
        histograms._constant_hessian,
        options.min_gain_to_split,
        None,
    )
    for feature in allowed_features:
        ref bins = histograms.feature(feature)
        if not bins:
            raise Error("ValueError: requested split feature is absent from histograms")
        var n_non_missing = Int(n_bins_non_missing[feature])
        var monotonic = monotonic_cst[feature]
        var has_missing = has_missing_values[feature]
        if n_non_missing > missing_bin or monotonic < -1 or monotonic > 1:
            raise Error("ValueError: invalid feature bin count or monotonic constraint")
        if is_categorical[feature]:
            if monotonic != 0:
                raise Error("ValueError: categorical features cannot be monotonic")
            if total.sum_hessians > 0:
                search.scan_categorical(
                    bins, feature, n_non_missing, missing_bin, has_missing
                )
        else:
            search.scan_numeric(
                bins, feature, n_non_missing, missing_bin, has_missing, monotonic
            )
    return search.best.copy()


def partition_rows[
    rows_origin: MutOrigin, scratch_origin: MutOrigin, //
](
    X_binned: NDArray[DType.uint8],
    split: _Split,
    missing_bin: UInt8,
    rows: Span[Int, rows_origin],
    scratch: Span[Int, scratch_origin],
) raises -> Int:
    """Stably partition a node's row IDs using separate caller-owned scratch."""
    if (
        X_binned.ndim != 2
        or split.feature < 0
        or split.feature >= X_binned.shape[1]
        or len(scratch) < len(rows)
        or split.left.count <= 0
        or split.left.count >= len(rows)
        or split.right.count != len(rows) - split.left.count
    ):
        raise Error(
            "ValueError: invalid partition shape, scratch size, or split counts"
        )

    var left = 0
    var right = split.left.count
    var feature_offset = split.feature * X_binned.strides[1]
    for row in rows:
        if row < 0 or row >= X_binned.shape[0]:
            raise Error("ValueError: partition row index is out of range")
        var code = X_binned.unsafe_get(feature_offset + row * X_binned.strides[0])
        if split.goes_left(code, missing_bin):
            if left == split.left.count:
                raise Error("ValueError: left partition count exceeds split count")
            scratch[left] = row
            left += 1
        else:
            if right == len(rows):
                raise Error("ValueError: right partition count exceeds split count")
            scratch[right] = row
            right += 1
    for i in range(len(rows)):
        rows[i] = scratch[i]
    return left
