# Criterion formulas and update protocol adapted from scikit-learn.
# Copyright the scikit-learn developers. BSD-3-Clause; LICENSES/scikit-learn.txt.
"""Native weighted tree criteria on the family-owned dense records."""
import numojo as nm
from std.math import isfinite
from std.sys import llvm_intrinsic
from moxie.tree._records import (TrainingData, CriterionConfig, Indices, Values,
    ChildImpurities, validate_training_layout, validate_range)
from moxie.tree._engine_api import Criterion

def log(x: Float64) -> Float64:
    # Same native compiler primitive validated by the pinned RNG owner.
    return llvm_intrinsic["llvm.log", Float64, has_side_effect=False](x)

# MAE weighted-median machinery, ported with the same operation order from the
# pinned sklearn: _criterion.pyx (MAE, precompute_absolute_errors, compute_ranks),
# _utils.pyx (WeightedFenwickTree) and utils/_sorting.pyx (introsort_3way).

def _swap(mut values: Values, mut indices: Indices, i: Int, j: Int) raises:
    var v = values.unsafe_get(i)
    values.unsafe_set(i, values.unsafe_get(j))
    values.unsafe_set(j, v)
    var k = indices.unsafe_get(i)
    indices.unsafe_set(i, indices.unsafe_get(j))
    indices.unsafe_set(j, k)

def _median3(values: Values, base: Int, n: Int) raises -> Float64:
    var a = values.unsafe_get(base)
    var b = values.unsafe_get(base + n // 2)
    var c = values.unsafe_get(base + n - 1)
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        return a
    elif b < c:
        if a < c:
            return a
        return c
    return b

def _sift_down(mut values: Values, mut indices: Indices, base: Int, start: Int, end: Int) raises:
    var root = start
    while True:
        var child = root * 2 + 1
        var maxind = root
        if child < end and values.unsafe_get(base + maxind) < values.unsafe_get(base + child):
            maxind = child
        if child + 1 < end and values.unsafe_get(base + maxind) < values.unsafe_get(base + child + 1):
            maxind = child + 1
        if maxind == root:
            break
        _swap(values, indices, base + root, base + maxind)
        root = maxind

def _heapsort(mut values: Values, mut indices: Indices, base: Int, n: Int) raises:
    var start = (n - 2) // 2
    while True:
        _sift_down(values, indices, base, start, n)
        if start == 0:
            break
        start -= 1
    var end = n - 1
    while end > 0:
        _swap(values, indices, base, base + end)
        _sift_down(values, indices, base, 0, end)
        end -= 1

def _insertion_sort(mut values: Values, mut indices: Indices, base: Int, n: Int) raises:
    for i in range(1, n):
        var value = values.unsafe_get(base + i)
        var index = indices.unsafe_get(base + i)
        var j = i
        while j > 0 and values.unsafe_get(base + j - 1) > value:
            values.unsafe_set(base + j, values.unsafe_get(base + j - 1))
            indices.unsafe_set(base + j, indices.unsafe_get(base + j - 1))
            j -= 1
        values.unsafe_set(base + j, value)
        indices.unsafe_set(base + j, index)

def _introsort_3way(mut values: Values, mut indices: Indices, base: Int, n: Int, maxd: Int) raises:
    var b = base
    var m = n
    var depth = maxd
    while m > 15:
        if depth <= 0:
            _heapsort(values, indices, b, m)
            return
        depth -= 1
        var pivot = _median3(values, b, m)
        var i = 0
        var l = 0
        var r = m
        while i < r:
            var x = values.unsafe_get(b + i)
            if x < pivot:
                _swap(values, indices, b + i, b + l)
                i += 1
                l += 1
            elif x > pivot:
                r -= 1
                _swap(values, indices, b + i, b + r)
            else:
                i += 1
        _introsort_3way(values, indices, b, l, depth)
        b += r
        m -= r
    _insertion_sort(values, indices, b, m)

def _sort_3way(mut values: Values, mut indices: Indices, n: Int) raises:
    """sklearn simultaneous_sort(use_three_way_partition=True) on [0, n)."""
    if n == 0:
        return
    # 2 * <intp_t>log2(n): the integer floor of log2 for positive n.
    var floor_log2 = 0
    var m = n
    while m > 1:
        m >>= 1
        floor_log2 += 1
    _introsort_3way(values, indices, 0, n, 2 * floor_log2)

@fieldwise_init
struct FenwickHit(Copyable, Movable):
    var index: Int
    var weight: Float64
    var weighted_y: Float64
    var previous: Int

struct WeightedFenwickTree(Copyable, Movable):
    """Prefix sums of weight and weight*y over 1-based ranks."""
    var tree_w: Values
    var tree_wy: Values
    var size: Int
    var max_pow2: Int
    var total_w: Float64
    var total_wy: Float64

    def __init__(out self, capacity: Int) raises:
        self.tree_w = nm.zeros[DType.float64](nm.Shape(capacity + 1))
        self.tree_wy = nm.zeros[DType.float64](nm.Shape(capacity + 1))
        self.size = 0
        self.max_pow2 = 0
        self.total_w = 0
        self.total_wy = 0

    def reset(mut self, size: Int) raises:
        if size + 1 > self.tree_w.size:
            raise Error("ValueError: Fenwick tree capacity exceeded")
        self.size = size
        for j in range(size + 1):
            self.tree_w.unsafe_set(j, 0)
            self.tree_wy.unsafe_set(j, 0)
        self.total_w = 0
        self.total_wy = 0
        var p = 1
        while p <= size:
            p <<= 1
        self.max_pow2 = p >> 1

    def add(mut self, index: Int, y: Float64, weight: Float64) raises:
        var weighted_y = weight * y
        var f = index + 1
        while f <= self.size:
            self.tree_w.unsafe_set(f, self.tree_w.unsafe_get(f) + weight)
            self.tree_wy.unsafe_set(f, self.tree_wy.unsafe_get(f) + weighted_y)
            f += f & -f
        self.total_w += weight
        self.total_wy += weighted_y

    def search(self, target_weight: Float64) raises -> FenwickHit:
        """Smallest rank with cumulative weight > target, as the reference search."""
        var target = target_weight
        var current = 0
        var cumulative_w = Float64(0)
        var cumulative_wy = Float64(0)
        var bit = self.max_pow2
        var equal_target = Float64(0)
        var equal_bit = 0
        while bit != 0:
            var nxt = current + bit
            if nxt <= self.size:
                var node = self.tree_w.unsafe_get(nxt)
                if target == node:
                    equal_target = target
                    equal_bit = bit
                    break
                elif target > node:
                    target -= node
                    current = nxt
                    cumulative_w += node
                    cumulative_wy += self.tree_wy.unsafe_get(nxt)
            bit >>= 1
        if bit == 0:
            return FenwickHit(current, cumulative_w, cumulative_wy, current)
        var previous = current
        while bit != 0:
            var nxt = previous + bit
            if nxt <= self.size:
                var node = self.tree_w.unsafe_get(nxt)
                if target > node:
                    target -= node
                    previous = nxt
            bit >>= 1
        bit = equal_bit
        target = equal_target
        while bit != 0:
            var nxt = current + bit
            if nxt <= self.size:
                var node = self.tree_w.unsafe_get(nxt)
                if target >= node:
                    target -= node
                    current = nxt
                    cumulative_w += node
                    cumulative_wy += self.tree_wy.unsafe_get(nxt)
            bit >>= 1
        return FenwickHit(current, cumulative_w, cumulative_wy, previous)

# Static selection retains one protocol without runtime string dispatch.
# 0: Gini, 1: Entropy, 2: MSE, 3: MAE, 4: Poisson.
struct WeightedCriterion[kind: Int](Criterion):
    var n_outputs: Int
    var n_classes: List[Int]
    var max_n_classes: Int
    var n_samples: Int
    var start: Int
    var end: Int
    var pos: Int
    var weighted_n_samples: Float64
    var weighted_n_node_samples: Float64
    var weighted_n_left: Float64
    var weighted_n_right: Float64
    var sq_sum_total: Float64
    var sum_total: Values
    var sum_left: Values
    var sum_right: Values
    var y: Values
    var weights: Optional[Values]
    var samples: Indices
    var initialized: Bool
    # MAE precomputations indexed by absolute sample position; empty otherwise.
    var left_abs_errors: Values
    var right_abs_errors: Values
    var left_medians: Values
    var right_medians: Values
    var node_medians: Values
    var sorted_y: Values
    var sorted_indices: Indices
    var ranks: Indices
    var prefix_sum_tree: WeightedFenwickTree

    def __init__(out self, config: CriterionConfig) raises:
        comptime assert Self.kind >= 0 and Self.kind <= 4
        if config.n_outputs <= 0 or config.n_samples <= 0 or len(config.n_classes) != config.n_outputs:
            raise Error("ValueError: invalid CriterionConfig dimensions")
        self.n_outputs = config.n_outputs
        self.n_samples = config.n_samples
        self.n_classes = config.n_classes.copy()
        self.max_n_classes = 1
        for c in self.n_classes:
            if c <= 0 or (Self.kind >= 2 and c != 1):
                raise Error("ValueError: invalid class count")
            self.max_n_classes = max(self.max_n_classes, c)
        if self.n_outputs > 9223372036854775807 // self.max_n_classes:
            raise Error("ValueError: statistics shape overflow")
        self.sum_total = nm.zeros[DType.float64](nm.Shape(self.n_outputs, self.max_n_classes))
        self.sum_left = nm.zeros[DType.float64](nm.Shape(self.n_outputs, self.max_n_classes))
        self.sum_right = nm.zeros[DType.float64](nm.Shape(self.n_outputs, self.max_n_classes))
        self.y = nm.zeros[DType.float64](nm.Shape(0, self.n_outputs))
        self.samples = nm.zeros[DType.int64](nm.Shape(0))
        self.weights = None
        self.start = 0
        self.end = 0
        self.pos = 0
        self.weighted_n_samples = 0
        self.weighted_n_node_samples = 0
        self.weighted_n_left = 0
        self.weighted_n_right = 0
        self.sq_sum_total = 0
        self.initialized = False
        var mae_size = self.n_samples if Self.kind == 3 else 0
        self.left_abs_errors = nm.zeros[DType.float64](nm.Shape(mae_size))
        self.right_abs_errors = nm.zeros[DType.float64](nm.Shape(mae_size))
        self.left_medians = nm.zeros[DType.float64](nm.Shape(mae_size))
        self.right_medians = nm.zeros[DType.float64](nm.Shape(mae_size))
        self.node_medians = nm.zeros[DType.float64](nm.Shape(self.n_outputs if Self.kind == 3 else 0))
        self.sorted_y = nm.zeros[DType.float64](nm.Shape(mae_size))
        self.sorted_indices = nm.zeros[DType.int64](nm.Shape(mae_size))
        self.ranks = nm.zeros[DType.int64](nm.Shape(mae_size))
        self.prefix_sum_tree = WeightedFenwickTree(mae_size)

    def _ready(self) raises:
        if not self.initialized:
            raise Error("NotFittedError: criterion has not been initialized")

    def _row(self, p: Int) raises -> Int:
        return Int(self.samples.unsafe_get(p * Int(self.samples.strides[0])))

    def _target(self, i: Int, k: Int) raises -> Float64:
        return self.y.unsafe_get(i * Int(self.y.strides[0]) + k)

    def _weight(self, i: Int) raises -> Float64:
        if self.weights:
            return self.weights.value().unsafe_get(i * Int(self.weights.value().strides[0]))
        return 1

    def initialize(mut self, data: TrainingData, samples: Indices,
                   start: Int, end: Int, weighted_n_samples: Float64) raises:
        self.initialized = False
        validate_training_layout(data)
        if samples.ndim != 1 or data.y.shape[1] != self.n_outputs or data.y.shape[0] > self.n_samples:
            raise Error("ValueError: incompatible criterion input dimensions")
        validate_range(start, start, end, samples.size)
        comptime if Self.kind == 3:
            if samples.size > self.left_abs_errors.size:
                raise Error("ValueError: MAE sample permutation exceeds criterion n_samples")
        if not isfinite(weighted_n_samples):
            raise Error("ValueError: criterion requires finite global weight")
        self.y = data.y.view_with_layout(data.y.shape, data.y.strides, data.y.offset)
        self.samples = samples.view_with_layout(samples.shape, samples.strides, samples.offset)
        self.weights = None
        if data.sample_weight:
            self.weights = data.sample_weight.value().view_with_layout(data.sample_weight.value().shape, data.sample_weight.value().strides, data.sample_weight.value().offset)
        self.start = start
        self.end = end
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0
        self.sq_sum_total = 0
        for j in range(self.sum_total.size):
            self.sum_total.unsafe_set(j, 0)
        for p in range(start, end):
            var i = self._row(p)
            if i < 0 or i >= data.y.shape[0]:
                raise Error("ValueError: sample index out of bounds")
            var w = self._weight(i)
            if not isfinite(w):
                raise Error("ValueError: weights must be finite")
            self.weighted_n_node_samples += w
            for k in range(self.n_outputs):
                var y = self._target(i, k)
                if not isfinite(y):
                    raise Error("ValueError: targets must be finite")
                var j = k * self.max_n_classes
                comptime if Self.kind < 2:
                    if y < 0 or y >= Float64(self.n_classes[k]) or y != Float64(Int(y)):
                        raise Error("ValueError: class labels must be encoded integer indices")
                    j += Int(y)
                    self.sum_total.unsafe_set(j, self.sum_total.unsafe_get(j) + w)
                else:
                    comptime if Self.kind == 4:
                        if y < 0:
                            raise Error("ValueError: Poisson targets must be nonnegative")
                    self.sum_total.unsafe_set(j, self.sum_total.unsafe_get(j) + w * y)
                    self.sq_sum_total += w * y * y
        if not isfinite(self.weighted_n_node_samples):
            raise Error("ValueError: node weight must be finite")
        comptime if Self.kind == 3:
            if start == end or self.weighted_n_node_samples <= 0:
                raise Error("ValueError: absolute_error requires positive node weight; reference median is undefined otherwise")
        self.initialized = True
        self.reset()

    def reset(mut self) raises:
        self._ready()
        self.pos = self.start
        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_node_samples
        for j in range(self.sum_total.size):
            self.sum_left.unsafe_set(j, 0)
            self.sum_right.unsafe_set(j, self.sum_total.unsafe_get(j))
        comptime if Self.kind == 3:
            # Reset may follow an external permutation of samples[start:end].
            for p in range(self.start, self.end):
                self.left_abs_errors.unsafe_set(p, 0)
                self.right_abs_errors.unsafe_set(p, 0)
            var n = self.end - self.start
            for k in range(self.n_outputs):
                for p in range(self.start, self.end):
                    self.sorted_y.unsafe_set(p - self.start, self._target(self._row(p), k))
                for i in range(n):
                    self.sorted_indices.unsafe_set(i, Int64(i))
                _sort_3way(self.sorted_y, self.sorted_indices, n)
                for i in range(n):
                    self.ranks.unsafe_set(self.start + Int(self.sorted_indices.unsafe_get(i)), Int64(i))
                self._precompute_absolute_errors[True](self.start, self.end)
                self._precompute_absolute_errors[False](self.end - 1, self.start - 1)
                # At p == start every sample is in the right child.
                self.node_medians.unsafe_set(k, self.right_medians.unsafe_get(self.start))

    def _precompute_absolute_errors[left: Bool](mut self, start: Int, end: Int) raises:
        """Forward (left) prefix or backward (right) suffix medians and errors."""
        var step = 1
        var n = end - start
        if not start < end:
            n = start - end
            step = -1
        self.prefix_sum_tree.reset(n)
        var p = start
        for _ in range(n):
            var w = self._weight(self._row(p))
            var rank = Int(self.ranks.unsafe_get(p))
            self.prefix_sum_tree.add(rank, self.sorted_y.unsafe_get(rank), w)
            # These children are rejected by the splitter. Searching their
            # nonpositive cumulative weight can run past the reference buffer.
            if self.prefix_sum_tree.total_w <= 0:
                p += step
                continue
            var hit = self.prefix_sum_tree.search(0.5 * self.prefix_sum_tree.total_w)
            if hit.index >= n or hit.previous >= n:
                raise Error("ValueError: MAE weighted median search left the sample buffer")
            var median: Float64
            if hit.index != hit.previous:
                median = (self.sorted_y.unsafe_get(hit.previous) + self.sorted_y.unsafe_get(hit.index)) / 2
            else:
                median = self.sorted_y.unsafe_get(hit.index)
            var w_right = self.prefix_sum_tree.total_w - hit.weight
            var wy_right = self.prefix_sum_tree.total_wy - hit.weighted_y
            var error = (wy_right - median * w_right) + (median * hit.weight - hit.weighted_y)
            comptime if left:
                self.left_medians.unsafe_set(p, median)
                self.left_abs_errors.unsafe_set(p, self.left_abs_errors.unsafe_get(p) + error)
            else:
                self.right_medians.unsafe_set(p, median)
                self.right_abs_errors.unsafe_set(p, self.right_abs_errors.unsafe_get(p) + error)
            p += step

    def reverse_reset(mut self) raises:
        self._ready()
        comptime if Self.kind == 3:
            raise Error("NotImplementedError: MAE reverse_reset is unsupported by the reference")
        self.pos = self.end
        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0
        for j in range(self.sum_total.size):
            self.sum_left.unsafe_set(j, self.sum_total.unsafe_get(j))
            self.sum_right.unsafe_set(j, 0)

    def _move(mut self, p: Int, sign: Float64) raises:
        var i = self._row(p)
        if i < 0 or i >= self.y.shape[0]:
            raise Error("ValueError: sample permutation has an invalid index")
        var w = sign * self._weight(i)
        self.weighted_n_left += w
        for k in range(self.n_outputs):
            var j = k * self.max_n_classes
            comptime if Self.kind < 2:
                j += Int(self._target(i, k))
                self.sum_left.unsafe_set(j, self.sum_left.unsafe_get(j) + w)
            else:
                self.sum_left.unsafe_set(j, self.sum_left.unsafe_get(j) + w * self._target(i, k))

    def update(mut self, new_pos: Int) raises:
        self._ready()
        validate_range(self.start, new_pos, self.end, self.samples.size)
        if new_pos < self.pos:
            raise Error("ValueError: update must move forward; call reset first")
        comptime if Self.kind == 3:
            for p in range(self.pos, new_pos):
                self._move(p, 1)
        else:
            if new_pos - self.pos <= self.end - new_pos:
                for p in range(self.pos, new_pos):
                    self._move(p, 1)
            else:
                self.reverse_reset()
                for p in range(self.end - 1, new_pos - 1, -1):
                    self._move(p, -1)
        self.pos = new_pos
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for j in range(self.sum_total.size):
            self.sum_right.unsafe_set(j, self.sum_total.unsafe_get(j) - self.sum_left.unsafe_get(j))

    # Evaluation order follows the reference exactly: last-bit differences decide
    # tied splits. Its tree `log` is log2(x) = ln(x) / ln(2) (sklearn tree/_utils.pxd).
    def _impurity(self, sums: Values, weight: Float64, begin: Int,
                  end: Int, square_sum: Float64) raises -> Float64:
        var result = Float64(0)
        comptime if Self.kind == 0:
            for k in range(self.n_outputs):
                var v = Float64(0)
                for c in range(self.n_classes[k]):
                    var count = sums.unsafe_get(k * self.max_n_classes + c)
                    v += count * count
                result += 1 - v / (weight * weight)
        elif Self.kind == 1:
            for k in range(self.n_outputs):
                for c in range(self.n_classes[k]):
                    var count = sums.unsafe_get(k * self.max_n_classes + c)
                    if count > 0:
                        var probability = count / weight
                        result -= probability * (log(probability) / log(Float64(2)))
        elif Self.kind == 2:
            result = square_sum / weight
            for k in range(self.n_outputs):
                var mean = sums.unsafe_get(k) / weight
                result -= mean * mean
        else:
            # poisson_loss: sum of w * xlogy(y, y / mean), divided once.
            for k in range(self.n_outputs):
                if sums.unsafe_get(k) <= 2.220446049250313e-15:
                    return Float64("inf")
                var center = sums.unsafe_get(k) / weight
                for p in range(begin, end):
                    var i = self._row(p)
                    var y = self._target(i, k)
                    if y != 0:
                        result += self._weight(i) * (y * log(y / center))
            return result / (weight * Float64(self.n_outputs))
        return result / Float64(self.n_outputs)

    def node_impurity(self) raises -> Float64:
        self._ready()
        comptime if Self.kind == 3:
            # The reference reads index 0, which is this node's start for the
            # root, the only node whose impurity its builders query.
            return self.right_abs_errors.unsafe_get(self.start) / (self.weighted_n_node_samples * Float64(self.n_outputs))
        return self._impurity(self.sum_total, self.weighted_n_node_samples,
                              self.start, self.end, self.sq_sum_total)

    def children_impurity(self) raises -> ChildImpurities:
        self._ready()
        if self.weighted_n_left <= 0 or self.weighted_n_right <= 0:
            raise Error("ValueError: impurity requires positive child weight")
        comptime if Self.kind == 3:
            var left = Float64(0)
            var right = Float64(0)
            if self.pos > self.start:
                left += self.left_abs_errors.unsafe_get(self.pos - 1)
            if self.pos < self.end:
                right += self.right_abs_errors.unsafe_get(self.pos)
            return ChildImpurities(left / (self.weighted_n_left * Float64(self.n_outputs)),
                                   right / (self.weighted_n_right * Float64(self.n_outputs)))
        var sq_left = Float64(0)
        comptime if Self.kind == 2:
            for p in range(self.start, self.pos):
                var i = self._row(p)
                for k in range(self.n_outputs):
                    var y = self._target(i, k)
                    sq_left += self._weight(i) * y * y
        return ChildImpurities(
            self._impurity(self.sum_left, self.weighted_n_left, self.start, self.pos, sq_left),
            self._impurity(self.sum_right, self.weighted_n_right, self.pos, self.end, self.sq_sum_total - sq_left))

    def _destination(self, dest: Values) raises:
        if dest.ndim != 2 or dest.shape[0] != self.n_outputs or dest.shape[1] != self.max_n_classes or dest.strides[1] != 1 or dest.strides[0] != self.max_n_classes:
            raise Error("ValueError: destination must be contiguous (n_outputs,max_n_classes)")

    def node_value(self, mut dest: Values) raises:
        self._ready()
        self._destination(dest)
        for j in range(dest.size):
            dest.unsafe_set(j, self.sum_total.unsafe_get(j) / self.weighted_n_node_samples)
        comptime if Self.kind == 3:
            for k in range(self.n_outputs):
                dest.unsafe_set(k, self.node_medians.unsafe_get(k))

    def _monotonic(self) raises:
        self._ready()
        if self.n_outputs != 1 or (Self.kind < 2 and self.n_classes[0] > 2):
            raise Error("ValueError: monotonicity requires single-output regression or binary classification")

    def clip_node_value(self, mut dest: Values, lower: Float64, upper: Float64) raises:
        self._monotonic()
        self._destination(dest)
        if not lower <= upper:
            raise Error("ValueError: invalid monotonic bounds")
        # Ordered comparisons preserve NaNs, matching the reference clipping.
        if dest.unsafe_get(0) < lower:
            dest.unsafe_set(0, lower)
        elif dest.unsafe_get(0) > upper:
            dest.unsafe_set(0, upper)
        comptime if Self.kind < 2:
            if self.n_classes[0] == 2:
                dest.unsafe_set(1, 1 - dest.unsafe_get(0))

    def _child_value(self, left: Bool) raises -> Float64:
        comptime if Self.kind == 3:
            if (left and self.pos <= self.start) or (not left and self.pos >= self.end):
                raise Error("ValueError: child must be nonempty")
            if left:
                return self.left_medians.unsafe_get(self.pos - 1)
            return self.right_medians.unsafe_get(self.pos)
        var weight = self.weighted_n_left if left else self.weighted_n_right
        if weight <= 0:
            raise Error("ValueError: child weight must be positive")
        if left:
            return self.sum_left.unsafe_get(0) / weight
        return self.sum_right.unsafe_get(0) / weight

    def middle_value(self) raises -> Float64:
        self._monotonic()
        return self._child_value(True) / 2 + self._child_value(False) / 2

    def check_monotonicity(self, constraint: Int8, lower: Float64, upper: Float64) raises -> Bool:
        self._monotonic()
        if constraint < -1 or constraint > 1 or not lower <= upper:
            raise Error("ValueError: invalid monotonic constraint or bounds")
        var left = self._child_value(True)
        var right = self._child_value(False)
        return left >= lower and right >= lower and left <= upper and right <= upper and (left - right) * Float64(constraint) <= 0

    def impurity_improvement(self, parent: Float64, left: Float64, right: Float64) raises -> Float64:
        self._ready()
        return (self.weighted_n_node_samples / self.weighted_n_samples) * (parent -
            self.weighted_n_right / self.weighted_n_node_samples * right -
            self.weighted_n_left / self.weighted_n_node_samples * left)

    def proxy_impurity_improvement(self) raises -> Float64:
        self._ready()
        if self.weighted_n_left <= 0 or self.weighted_n_right <= 0:
            raise Error("ValueError: proxy requires positive child weights")
        comptime if Self.kind == 2 or Self.kind == 4:
            var proxy_left = Float64(0)
            var proxy_right = Float64(0)
            for k in range(self.n_outputs):
                var left = self.sum_left.unsafe_get(k)
                var right = self.sum_right.unsafe_get(k)
                comptime if Self.kind == 2:
                    proxy_left += left * left
                    proxy_right += right * right
                else:
                    if left <= 2.220446049250313e-15 or right <= 2.220446049250313e-15:
                        return -Float64("inf")
                    proxy_left -= left * (log(left / self.weighted_n_left) / log(Float64(2)))
                    proxy_right -= right * (log(right / self.weighted_n_right) / log(Float64(2)))
            comptime if Self.kind == 2:
                return proxy_left / self.weighted_n_left + proxy_right / self.weighted_n_right
            return -proxy_left - proxy_right
        else:
            var children = self.children_impurity()
            return -self.weighted_n_left * children.left - self.weighted_n_right * children.right

comptime Gini = WeightedCriterion[0]
comptime Entropy = WeightedCriterion[1]
comptime MSE = WeightedCriterion[2]
comptime MAE = WeightedCriterion[3]
comptime Poisson = WeightedCriterion[4]
