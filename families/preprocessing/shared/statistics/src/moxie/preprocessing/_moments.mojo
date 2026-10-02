"""Version 1 bounded-memory statistics on borrowed NuMojo arrays.

Corrected two-pass/Youngs-Cramer equations adapted from scikit-learn
4358542a (BSD-3-Clause; see LICENSES/sklearn.txt). No public estimators.
Masked count reduction follows NumPy's pairwise tree (LICENSES/numpy.txt).
"""
from std.os import getenv
from std.collections import InlineArray
from std.math import isnan, isfinite, abs, min, max, sqrt
from max.algorithm import sum
from numojo import NDArray, Shape, zeros
from moxie.preprocessing._validation import validate_dense

comptime MOMENTS_API_VERSION = 1
comptime DEFAULT_TILE_ROWS = 256
comptime TILE_SCRATCH_BYTES = 1600


@always_inline
def _count_value[dtype: DType, wdtype: DType](
    x: NDArray[dtype], weights: NDArray[wdtype], row: Int, feature: Int,
    mode: Int, scalar: Float64,
) raises -> Float64:
    var value = x.unsafe_get(row * Int(x.strides[0]) + feature * Int(x.strides[1]))
    var weight = scalar if mode == 2 else Float64(weights.unsafe_get(row * Int(weights.strides[0])))
    # Retain nonfinite scalar * zero; missing finite weights contribute zero.
    return weight * (Float64(0) if isnan(value) else Float64(1))


def _pairwise_count[dtype: DType, wdtype: DType](
    x: NDArray[dtype], weights: NDArray[wdtype], feature: Int,
    mode: Int, scalar: Float64, begin: Int, end: Int,
) raises -> Float64:
    # NumPy loops_utils.h.src: eight lanes, 128-element leaves, balanced splits
    # on multiples of eight. Fixed stacks replace recursion and materialization.
    # Int-sized row counts require fewer than 64 pending right branches.
    var right_begin = InlineArray[Int, 64](fill=0)
    var right_end = InlineArray[Int, 64](fill=0)
    var left_sum = InlineArray[Float64, 64](fill=0)
    var depth = 0
    var start = begin
    var stop = end
    while True:
        var size = stop - start
        if size > 128:
            var half = size // 2
            half -= half % 8
            right_begin[depth] = start + half
            right_end[depth] = stop
            depth += 1
            stop = start + half
            continue
        var result = Float64(-0.0)
        if size < 8:
            for row in range(start, stop):
                result += _count_value(x, weights, row, feature, mode, scalar)
        else:
            var lanes = SIMD[DType.float64, 8](0)
            for lane in range(8):
                lanes[lane] = _count_value(x, weights, start + lane, feature, mode, scalar)
            var tail = stop - size % 8
            for row in range(start + 8, tail, 8):
                for lane in range(8):
                    lanes[lane] += _count_value(x, weights, row + lane, feature, mode, scalar)
            result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
            for row in range(tail, stop):
                result += _count_value(x, weights, row, feature, mode, scalar)
        while depth > 0 and right_begin[depth - 1] == -1:
            depth -= 1
            result = left_sum[depth] + result
        if depth == 0:
            return result
        left_sum[depth - 1] = result
        start = Int(right_begin[depth - 1])
        stop = Int(right_end[depth - 1])
        right_begin[depth - 1] = -1


def _fast_axis_count[dtype: DType, wdtype: DType](
    x: NDArray[dtype], weights: NDArray[wdtype], feature: Int,
    mode: Int, scalar: Float64,
) raises -> Float64:
    var block = x.shape[0]
    # Widening a Float32 reduction uses NumPy's default 8192-element iterator
    # buffer. This is a logical reduction boundary, independent of tile_rows.
    if (mode == 1 and wdtype == DType.float32) or (mode == 2 and dtype == DType.float32):
        block = 8192
    var result = Float64(0)
    for start in range(0, x.shape[0], block):
        result += _pairwise_count(x, weights, feature, mode, scalar, start, start + min(block, x.shape[0] - start))
    return result

def _pairwise_rows[
    generate: def(Int) capturing -> Float64,
](begin: Int, end: Int) raises -> Float64:
    """NumPy's eight-lane pairwise tree over generated row values (Float64)."""
    var right_begin = InlineArray[Int, 64](fill=0)
    var right_end = InlineArray[Int, 64](fill=0)
    var left_sum = InlineArray[Float64, 64](fill=0)
    var depth = 0
    var start = begin
    var stop = end
    while True:
        var size = stop - start
        if size > 128:
            var half = size // 2
            half -= half % 8
            right_begin[depth] = start + half
            right_end[depth] = stop
            depth += 1
            stop = start + half
            continue
        var result = Float64(-0.0)
        if size < 8:
            for row in range(start, stop):
                result += generate(row)
        else:
            var lanes = SIMD[DType.float64, 8](0)
            for lane in range(8):
                lanes[lane] = generate(start + lane)
            var tail = stop - size % 8
            for row in range(start + 8, tail, 8):
                for lane in range(8):
                    lanes[lane] += generate(row + lane)
            result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
            for row in range(tail, stop):
                result += generate(row)
        while depth > 0 and right_begin[depth - 1] == -1:
            depth -= 1
            result = left_sum[depth] + result
        if depth == 0:
            return result
        left_sum[depth - 1] = result
        start = Int(right_begin[depth - 1])
        stop = Int(right_end[depth - 1])
        right_begin[depth - 1] = -1


def _blocked_pairwise[
    generate: def(Int) capturing -> Float64,
](size: Int, block: Int) raises -> Float64:
    # Casting reductions pairwise-sum each NumPy iterator buffer, then add serially.
    var result = Float64(0)
    for start in range(0, size, block):
        result += _pairwise_rows[generate](start, start + min(block, size - start))
    return result


def _column_sum[dtype: DType, kind: Int, origin: Origin](
    column_ptr: Pointer[SIMD[dtype, 1], origin], stride: Int, size: Int,
    center: Float64, block: Int,
) raises -> Float64:
    """kind 0: raw values, 1: centered values, 2: squared residuals; NaN adds 0."""
    def generate(i: Int) capturing -> Float64:
        var value = Float64(column_ptr.unsafe_load(i * stride))
        if isnan(value):
            return Float64(0)
        comptime if kind == 0:
            return value
        elif kind == 1:
            return value - center
        else:
            var delta = value - center
            return delta * delta
    return _blocked_pairwise[generate](size, block)


def _allocate_vector(size: Int) raises -> NDArray[DType.float64]:
    # Single allocation site: memory acceptance pairs these native events with
    # Callgrind's actual NuMojo allocator calls and isolated process peak RSS.
    if getenv("MOXIE_STATISTICS_ALLOC_TRACE") == "1":
        print("statistics_allocation", size * 8)
    return zeros[DType.float64](Shape(size))


struct MomentsStateV1(Copyable, Movable):
    var count: NDArray[DType.float64]
    var mean: NDArray[DType.float64]
    var variance: NDArray[DType.float64]
    var new_sum: NDArray[DType.float64]
    var track_variance: Bool
    var weighted: Bool

    def __init__(out self, features: Int, *, track_variance: Bool = True) raises:
        if features < 1:
            raise Error("ValueError: features must be positive")
        self.count = _allocate_vector(features)
        self.mean = _allocate_vector(features)
        self.variance = _allocate_vector(features)
        self.new_sum = _allocate_vector(features)
        self.track_variance = track_variance
        self.weighted = False

    def scalar_count(self) raises -> Bool:
        if isnan(self.count.unsafe_get(0)):
            return False
        for j in range(1, self.count.shape[0]):
            if self.count.unsafe_get(j) != self.count.unsafe_get(0):
                return False
        return True

    def scalar_count_value(self) raises -> Float64:
        """Borrow-free scalar projection; unequal feature counts raise."""
        if not self.scalar_count():
            raise Error("ValueError: counts require a feature vector")
        return self.count.unsafe_get(0)

    def integer_count_at(self, feature: Int) raises -> Int64:
        """Native unweighted projection without an extra integer array."""
        if self.weighted:
            raise Error("ValueError: weighted counts retain floating representation")
        if feature < 0 or feature >= self.count.shape[0]:
            raise Error("ValueError: feature out of bounds")
        var value = self.count.unsafe_get(feature)
        if not isfinite(value) or value < 0 or value >= Float64(9223372036854775808):
            raise Error("ValueError: count cannot be represented as Int64")
        var result = Int64(value)
        if Float64(result) != value:
            raise Error("ValueError: unweighted count must be integral")
        return result

    def constant_at(self, feature: Int) raises -> Bool:
        if not self.track_variance:
            raise Error("ValueError: variance was disabled")
        if feature < 0 or feature >= self.count.shape[0]:
            raise Error("ValueError: feature out of bounds")
        var n = self.count.unsafe_get(feature)
        var m = self.mean.unsafe_get(feature)
        var v = self.variance.unsafe_get(feature)
        var eps = Float64(2.220446049250313e-16)
        var term = n * m * eps
        return v <= n * eps * v + term * term


def _check_state(state: MomentsStateV1, features: Int) raises:
    if state.count.ndim != 1 or state.mean.ndim != 1 or state.variance.ndim != 1 or state.new_sum.ndim != 1:
        raise Error("ValueError: state must contain feature vectors")
    if state.count.shape[0] != features or state.mean.shape[0] != features or state.variance.shape[0] != features or state.new_sum.shape[0] != features:
        raise Error("ValueError: state feature mismatch")
    # The state owns compact vectors; callers may seed elements, not replace layout.
    if state.count.strides[0] != 1 or state.mean.strides[0] != 1 or state.variance.strides[0] != 1 or state.new_sum.strides[0] != 1:
        raise Error("ValueError: state vectors must be contiguous")


def _update[dtype: DType, wdtype: DType](
    x: NDArray[dtype], weights: NDArray[wdtype], mut state: MomentsStateV1,
    mode: Int, scalar: Float64, tile_rows: Int,
) raises:
    comptime assert wdtype == DType.float32 or wdtype == DType.float64
    validate_dense(x, allow_nan=True)
    _check_state(state, x.shape[1])
    if tile_rows < 1:
        raise Error("ValueError: tile_rows must be positive")
    if mode == 1:
        if weights.ndim != 1 or weights.shape[0] != x.shape[0]:
            raise Error("ValueError: weights must have one entry per sample")
        var nonzero = False
        for i in range(x.shape[0]):
            var w = Float64(weights.unsafe_get(i * Int(weights.strides[0])))
            if not isfinite(w):
                raise Error("ValueError: weights must be finite")
            nonzero = nonzero or w != 0
        if not nonzero:
            raise Error("ValueError: weights cannot all be zero")
    elif mode == 2:
        if scalar == 0:
            raise Error("ValueError: scalar weight must be nonzero")
    # Validation completes before any state writes. Traversal tiles do not create
    # updates: the raw whole-batch sum/count survive across all tiles and passes.
    for j in range(x.shape[1]):
        var total = Float64(0)
        var count = Float64(0)
        for start in range(0, x.shape[0], tile_rows):
            for i in range(start, start + min(tile_rows, x.shape[0] - start)):
                var value = Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
                if not isnan(value):
                    var w = scalar if mode == 2 else Float64(1)
                    if mode == 1:
                        w = Float64(weights.unsafe_get(i * Int(weights.strides[0])))
                    total += value * w
                    count += w
                elif mode == 2 and not isfinite(scalar):
                    # Reference broadcasts scalar weights without its vector
                    # finite check: nonfinite * masked zero must still propagate.
                    total += scalar * 0
                    count += scalar * 0
        # Unweighted reference sums are NumPy nansum reductions: fast (column-
        # contiguous or single-feature) axes use the pairwise tree, Float32
        # widening adds 8192-element buffer boundaries; row-major is serial.
        var fast_axis = x.shape[1] == 1 or abs(Int(x.strides[0])) < abs(Int(x.strides[1]))
        var column_ptr = x.unsafe_ptr().unsafe_offset(j * Int(x.strides[1]))
        var row_stride = Int(x.strides[0])
        if mode == 0 and fast_axis:
            total = _column_sum[dtype, 0](column_ptr, row_stride, x.shape[0], 0, 8192 if dtype == DType.float32 else x.shape[0])
        # The reference's materialized weighted missingness mask preserves axis
        # order: a singleton/column-major feature uses a fast-axis pairwise sum;
        # row-major multi-feature arrays use the serial slow-axis reduction.
        if mode != 0 and (x.shape[1] == 1 or abs(Int(x.strides[0])) < abs(Int(x.strides[1]))):
            count = _fast_axis_count(x, weights, j, mode, scalar)
        var old_count = state.count.unsafe_get(j)
        var old_sum = state.mean.unsafe_get(j) * old_count
        var updated_count = old_count + count
        var updated_mean = (old_sum + total) / updated_count
        if state.track_variance:
            var center = total / count
            var correction = Float64(0)
            var squares = Float64(0)
            for start in range(0, x.shape[0], tile_rows):
                for i in range(start, start + min(tile_rows, x.shape[0] - start)):
                    var value = Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
                    if not isnan(value):
                        var w = scalar if mode == 2 else Float64(1)
                        if mode == 1:
                            w = Float64(weights.unsafe_get(i * Int(weights.strides[0])))
                        var delta = value - center
                        correction += w * delta
                        squares += w * (delta * delta)
                    elif mode == 2 and not isfinite(scalar):
                        correction += scalar * 0
                        squares += scalar * 0
            if mode == 0 and fast_axis:
                # temp = X - T is Float64 for both dtypes: no casting buffer.
                correction = _column_sum[dtype, 1](column_ptr, row_stride, x.shape[0], center, x.shape[0])
                squares = _column_sum[dtype, 2](column_ptr, row_stride, x.shape[0], center, x.shape[0])
            var new_unnormalized = squares - correction * correction / count
            var ratio = old_count / count
            var delta_sum = old_sum / ratio - total
            var combined = state.variance.unsafe_get(j) * old_count + new_unnormalized + ratio / updated_count * (delta_sum * delta_sum)
            if old_count == 0:
                combined = new_unnormalized
            state.variance.unsafe_set(j, combined / updated_count)
        state.count.unsafe_set(j, updated_count)
        state.mean.unsafe_set(j, updated_mean)
        state.new_sum.unsafe_set(j, total)
    state.weighted = state.weighted or mode != 0


def update_moments[dtype: DType](x: NDArray[dtype], mut state: MomentsStateV1, *, tile_rows: Int = DEFAULT_TILE_ROWS) raises:
    _update(x, x, state, 0, 1, tile_rows)


def update_moments_weighted[dtype: DType, wdtype: DType](x: NDArray[dtype], weights: NDArray[wdtype], mut state: MomentsStateV1, *, tile_rows: Int = DEFAULT_TILE_ROWS) raises:
    _update(x, weights, state, 1, 1, tile_rows)


def update_moments_scalar[dtype: DType](x: NDArray[dtype], sample_weight: Float64, mut state: MomentsStateV1, *, tile_rows: Int = DEFAULT_TILE_ROWS) raises:
    _update(x, x, state, 2, Float64(sample_weight.cast[dtype]()), tile_rows)


struct ExtremaStateV1(Copyable, Movable):
    var minimum: NDArray[DType.float64]
    var maximum: NDArray[DType.float64]
    var max_abs: NDArray[DType.float64]
    var initialized: Bool

    def __init__(out self, features: Int) raises:
        if features < 1:
            raise Error("ValueError: features must be positive")
        self.minimum = _allocate_vector(features)
        self.maximum = _allocate_vector(features)
        self.max_abs = _allocate_vector(features)
        self.initialized = False


def update_extrema[dtype: DType](x: NDArray[dtype], mut state: ExtremaStateV1) raises:
    validate_dense(x, allow_nan=True)
    if state.minimum.ndim != 1 or state.maximum.ndim != 1 or state.max_abs.ndim != 1:
        raise Error("ValueError: extrema must be vectors")
    if state.minimum.shape[0] != x.shape[1] or state.maximum.shape[0] != x.shape[1] or state.max_abs.shape[0] != x.shape[1]:
        raise Error("ValueError: extrema feature mismatch")
    for j in range(x.shape[1]):
        var lo = Float64("nan")
        var hi = Float64("nan")
        for i in range(x.shape[0]):
            var v = Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
            if not isnan(v):
                lo = v if isnan(lo) else min(lo, v)
                hi = v if isnan(hi) else max(hi, v)
        var magnitude = max(abs(lo), abs(hi))
        if state.initialized:
            var old_lo = state.minimum.unsafe_get(j)
            var old_hi = state.maximum.unsafe_get(j)
            var old_abs = state.max_abs.unsafe_get(j)
            lo = Float64("nan") if isnan(lo) or isnan(old_lo) else min(lo, old_lo)
            hi = Float64("nan") if isnan(hi) or isnan(old_hi) else max(hi, old_hi)
            magnitude = Float64("nan") if isnan(magnitude) or isnan(old_abs) else max(magnitude, old_abs)
        state.minimum.unsafe_set(j, lo)
        state.maximum.unsafe_set(j, hi)
        state.max_abs.unsafe_set(j, magnitude)
    state.initialized = True


struct RobustQuantilesV1(Copyable, Movable):
    var lower: NDArray[DType.float64]
    var median: NDArray[DType.float64]
    var upper: NDArray[DType.float64]

    def __init__(out self, features: Int) raises:
        self.lower = _allocate_vector(features)
        self.median = _allocate_vector(features)
        self.upper = _allocate_vector(features)


def _sift(mut column: NDArray[DType.float64], start: Int, stop: Int):
    var root = start
    while root < stop // 2:
        var child = 2 * root + 1
        if child + 1 < stop and column.unsafe_get(child) < column.unsafe_get(child + 1):
            child += 1
        if column.unsafe_get(root) >= column.unsafe_get(child):
            return
        var tmp = column.unsafe_get(root)
        column.unsafe_set(root, column.unsafe_get(child))
        column.unsafe_set(child, tmp)
        root = child


def _sort(mut column: NDArray[DType.float64], size: Int):
    for k in range(size // 2):
        _sift(column, size // 2 - k - 1, size)
    for k in range(size - 1):
        var end = size - k - 1
        var tmp = column.unsafe_get(0)
        column.unsafe_set(0, column.unsafe_get(end))
        column.unsafe_set(end, tmp)
        _sift(column, 0, end)


def _percentile[dtype: DType](column: NDArray[DType.float64], size: Int, q: Float64) raises -> Float64:
    if size == 0:
        return Float64("nan")
    var position = Float64(size - 1) * (q / 100)
    var lower = Int(position)
    var upper = min(lower + 1, size - 1)
    var fraction = position - Float64(lower)
    var a = column.unsafe_get(lower)
    var b = column.unsafe_get(upper)
    # NumPy's linear quantile subtracts in input dtype then interpolates in f64.
    var diff = Float64(b.cast[dtype]() - a.cast[dtype]())
    return b - diff * (1 - fraction) if fraction >= 0.5 else a + diff * fraction


def robust_quantiles[dtype: DType](
    x: NDArray[dtype], *, q_min: Float64 = 25, q_max: Float64 = 75,
    column_budget_bytes: Int = 67108864,
) raises -> RobustQuantilesV1:
    validate_dense(x, allow_nan=True)
    if not (q_min >= 0 and q_min <= q_max and q_max <= 100):
        raise Error("ValueError: invalid quantile range")
    if column_budget_bytes < 0 or x.shape[0] > column_budget_bytes // 8:
        raise Error("ValueError: exact quantiles require one Float64 column workspace")
    var result = RobustQuantilesV1(x.shape[1])
    var column = _allocate_vector(x.shape[0])
    for j in range(x.shape[1]):
        var size = 0
        for i in range(x.shape[0]):
            var value = Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
            if not isnan(value):
                column.unsafe_set(size, value)
                size += 1
        if size > 1:
            _sort(column, size)
        result.lower.unsafe_set(j, _percentile[dtype](column, size, q_min))
        result.upper.unsafe_set(j, _percentile[dtype](column, size, q_max))
        var median = Float64("nan")
        if size > 0:
            if size % 2:
                median = column.unsafe_get(size // 2)
            else:
                var a = column.unsafe_get(size // 2 - 1).cast[dtype]()
                var b = column.unsafe_get(size // 2).cast[dtype]()
                median = Float64((a + b) / 2)
        result.median.unsafe_set(j, median)
    return result^


# Functional scale has a distinct input-precision, uncorrected variance contract.
comptime INPUT_PRECISION_API_VERSION = 1


@fieldwise_init
struct InputPrecisionStatsV1[dtype: DType](Copyable, Movable):
    var count: Int
    var mean: SIMD[Self.dtype, 1]
    var stddev: SIMD[Self.dtype, 1]


def _precision_axis[dtype: DType](x: NDArray[dtype], axis: Int, index: Int) raises -> Int:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim != 2:
        raise Error("ValueError: input-precision statistics require rank two")
    var normalized = axis + 2 if axis < 0 else axis
    if normalized < 0 or normalized > 1:
        raise Error("ValueError: axis must be 0 or 1 (or -2 or -1)")
    if index < 0 or index >= x.shape[1 - normalized]:
        raise Error("ValueError: reduction index out of bounds")
    return normalized


def _precision_sum[
    dtype: DType,
    generate: def[d: DType, width: Int](Int) capturing -> SIMD[d, width],
](size: Int, pairwise: Bool) raises -> SIMD[dtype, 1]:
    # MAX floating sums differ in reduction order and flush subnormal tails.
    # Retain its integer-count generator; use the NumPy order for float values.
    if size < 8 or not pairwise:
        var total = SIMD[dtype, 1](0)
        for i in range(size):
            total += generate[dtype, 1](i)
        return total
    var right_begin = InlineArray[Int, 64](fill=0)
    var right_end = InlineArray[Int, 64](fill=0)
    var left_sum = InlineArray[SIMD[dtype, 1], 64](fill=0)
    var depth = 0
    var start = 0
    var stop = size
    while True:
        var length = stop - start
        if length > 128:
            var half = length // 2
            half -= half % 8
            right_begin[depth] = start + half
            right_end[depth] = stop
            depth += 1
            stop = start + half
            continue
        var result = SIMD[dtype, 1](-0.0)
        if length < 8:
            for i in range(start, stop):
                result += generate[dtype, 1](i)
        else:
            var lanes = generate[dtype, 8](start)
            var tail = stop - length % 8
            for i in range(start + 8, tail, 8):
                lanes += generate[dtype, 8](i)
            result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
            for i in range(tail, stop):
                result += generate[dtype, 1](i)
        while depth > 0 and right_begin[depth - 1] == -1:
            depth -= 1
            result = left_sum[depth] + result
        if depth == 0:
            return SIMD[dtype, 1](0) + result
        left_sum[depth - 1] = result
        start = right_begin[depth - 1]
        stop = right_end[depth - 1]
        right_begin[depth - 1] = -1


def _input_precision[dtype: DType, with_std: Bool](
    x: NDArray[dtype], axis: Int, index: Int,
) raises -> InputPrecisionStatsV1[dtype]:
    var ax = _precision_axis(x, axis, index)
    var size = x.shape[ax]
    var stride = Int(x.strides[ax])
    var input_ptr = x.unsafe_ptr().unsafe_offset(index * Int(x.strides[1 - ax]))
    # NumPy replaces NaNs in a K-order copy even when none occur. Signed and
    # gapped source strides affect loads, not the resulting reduction order.
    var fast = x.shape[1 - ax] == 1 or abs(stride) < abs(Int(x.strides[1 - ax]))
    if abs(stride) == abs(Int(x.strides[1 - ax])):
        fast = ax == 1 or x.shape[1 - ax] == 1

    var count = _precision_count(input_ptr, stride, size)
    var total = _precision_reduce[dtype, False](input_ptr, stride, size, fast, SIMD[dtype, 1](0))
    var average = (Float64(total) / Float64(count)).cast[dtype]()
    var deviation = Float64("nan").cast[dtype]()
    comptime if with_std:
        var squares = _precision_reduce[dtype, True](input_ptr, stride, size, fast, average)
        deviation = sqrt((Float64(squares) / Float64(count)).cast[dtype]())
    return InputPrecisionStatsV1[dtype](count, average, deviation)


def _precision_count[dtype: DType, origin: Origin](
    input_ptr: Pointer[SIMD[dtype, 1], origin], stride: Int, size: Int,
) raises -> Int:
    def valid[d: DType, width: Int](i: Int) capturing -> SIMD[d, width]:
        var result = SIMD[d, width](0)
        for lane in range(width):
            result[lane] = SIMD[d, 1](Int(not isnan(input_ptr.unsafe_load((i + lane) * stride))))
        return result
    return Int(sum[DType.int64, valid](size))


def _precision_reduce[dtype: DType, squared: Bool, origin: Origin](
    input_ptr: Pointer[SIMD[dtype, 1], origin], stride: Int, size: Int,
    pairwise: Bool, average: SIMD[dtype, 1],
) raises -> SIMD[dtype, 1]:
    def values[d: DType, width: Int](i: Int) capturing -> SIMD[d, width]:
        var result = SIMD[d, width](0)
        for lane in range(width):
            var value = input_ptr.unsafe_load((i + lane) * stride).cast[d]()
            var output = value
            comptime if squared:
                var delta = value - average.cast[d]()
                output = delta * delta
            result[lane] = isnan(value).select(SIMD[d, 1](0), output)
        return result
    return _precision_sum[dtype, values](size, pairwise)


def nanmean_at[dtype: DType](x: NDArray[dtype], axis: Int, index: Int) raises -> SIMD[dtype, 1]:
    """Input-precision NaN-aware mean of one row/column, with no owned arrays."""
    return _input_precision[dtype, False](x, axis, index).mean


def nanmean_std_at[dtype: DType](x: NDArray[dtype], axis: Int, index: Int) raises -> InputPrecisionStatsV1[dtype]:
    """Input-precision mean and population stddev; IEEE Inf/NaN propagate."""
    return _input_precision[dtype, True](x, axis, index)
