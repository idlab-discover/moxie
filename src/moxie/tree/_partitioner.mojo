# scikit-learn developers, BSD-3-Clause; see family LICENSES/scikit-learn.txt.
"""Dense numeric and categorical partitioning on the single family workspace representation."""
import numojo as nm
from moxie.tree._records import TrainingData, SplitWorkspace, SplitRecord, Indices, FeatureRange, Values, CategoryBits

def swap(mut values: Indices, i: Int, j: Int):
    var saved = values.unsafe_get(i)
    values.unsafe_set(i, values.unsafe_get(j))
    values.unsafe_set(j, saved)

def feature(data: TrainingData, sample: Int, column: Int) raises -> Float32:
    return data.X.unsafe_get(sample * Int(data.X.strides[0]) + column * Int(data.X.strides[1]))

def sort_feature(data: TrainingData, mut work: SplitWorkspace, column: Int) raises -> Bool:
    # Reference sample order decides weighted summation order, hence split ties:
    # missing compaction and the 3-way introsort follow sklearn _partitioner.pyx.
    var n_missing = 0
    if data.missing_values_in_feature_mask.unsafe_get(column * Int(data.missing_values_in_feature_mask.strides[0])) != 0:
        var i = work.start
        var current_end = work.end - 1
        while i <= current_end:
            var last = feature(data, Int(work.samples.unsafe_get(current_end)), column)
            if last != last:
                n_missing += 1
                current_end -= 1
                continue
            var x = feature(data, Int(work.samples.unsafe_get(i)), column)
            if x != x:
                swap(work.samples, i, current_end)
                n_missing += 1
                current_end -= 1
            work.feature_values.unsafe_set(i, feature(data, Int(work.samples.unsafe_get(i)), column))
            i += 1
    else:
        for i in range(work.start, work.end):
            work.feature_values.unsafe_set(i, feature(data, Int(work.samples.unsafe_get(i)), column))
    work.n_missing = n_missing
    var end = work.end - n_missing
    if end == work.start:
        return True
    introsort3(work.feature_values, work.samples, work.start, end, introsort_depth(end - work.start))
    return n_missing == 0 and work.feature_values.unsafe_get(end - 1) <= work.feature_values.unsafe_get(work.start) + Float32(1e-7)

def partition(data: TrainingData, mut work: SplitWorkspace, split: SplitRecord) raises -> Int:
    var p = work.start
    var end = work.end
    while p < end:
        var x = feature(data, Int(work.samples.unsafe_get(p)), split.feature)
        var left = split.missing_go_to_left != 0 if x != x else Float64(x) <= split.threshold
        if x == x and split.split_kind != 0:
            var code = Int(x)
            if split.split_kind == 1:
                left = (split.left_cat_bitset[code // 32] & (UInt32(1) << UInt32(code % 32))) != 0
            else:
                var mixed = split.left_cat_bitset[0] ^ UInt32(code)
                mixed ^= mixed >> 16
                mixed *= UInt32(0x7feb352d)
                mixed ^= mixed >> 15
                mixed *= UInt32(0x846ca68b)
                mixed ^= mixed >> 16
                left = (mixed & 1) != 0
        if left:
            p += 1
        else:
            end -= 1
            swap(work.samples, p, end)
    return p

def find_range(data: TrainingData, mut work: SplitWorkspace, column: Int) raises -> FeatureRange:
    # Random search must preserve sample order: sorting changes weighted sums.
    var low = Float32(Float64("inf"))
    var high = -low
    work.n_missing = 0
    for i in range(work.start, work.end):
        var x = feature(data, Int(work.samples.unsafe_get(i)), column)
        if x != x:
            work.n_missing += 1
        else:
            low = min(low, x)
            high = max(high, x)
    return FeatureRange(low, high)

from moxie.tree._engine_api import Partitioner
from moxie.tree._records import SplitPosition, validate_range

struct DensePartitioner(Partitioner):
    var sorted_categories: Indices
    var counts: Indices
    var categorical: Bool
    def __init__(out self) raises:
        self.categorical = False
        self.sorted_categories = nm.zeros[DType.int64](nm.Shape(0))
        self.counts = nm.zeros[DType.int64](nm.Shape(0))
    def init_node_split(mut self, mut work: SplitWorkspace, start: Int, end: Int) raises:
        validate_range(start, start, end, work.samples.size)
        work.start = start
        work.end = end
        work.n_missing = 0
    def sort_samples_and_feature_values(mut self, data: TrainingData, mut work: SplitWorkspace, feature: Int) raises -> Bool:
        self.categorical = data.n_categories.unsafe_get(feature * Int(data.n_categories.strides[0])) >= 0
        if self.categorical:
            self.sort_categories(data, work, feature)
            var end = work.end - work.n_missing
            return end == work.start or (work.n_missing == 0 and work.feature_values.unsafe_get(work.start) == work.feature_values.unsafe_get(end - 1))
        return sort_feature(data, work, feature)
    def shift_missing_to_the_left(mut self, mut work: SplitWorkspace) raises:
        var samples = work.samples.copy()
        var values = work.feature_values.copy()
        var finite = work.end - work.start - work.n_missing
        for i in range(work.end - work.start):
            var source = work.start + (i + finite) % (work.end - work.start)
            work.samples.unsafe_set(work.start + i, samples.unsafe_get(source))
            work.feature_values.unsafe_set(work.start + i, values.unsafe_get(source))
    def find_min_max(mut self, data: TrainingData, mut work: SplitWorkspace, feature: Int) raises -> FeatureRange:
        return find_range(data, work, feature)
    def next_p(mut self, work: SplitWorkspace, position: SplitPosition, missing_left: Bool) raises -> SplitPosition:
        var end = work.end if missing_left else work.end - work.n_missing
        var p = position.current
        if p == end and not missing_left:
            return SplitPosition(work.end, work.end)
        if missing_left and p == work.start:
            p += work.n_missing
        p += 1
        while p < end and (work.feature_values.unsafe_get(p) == work.feature_values.unsafe_get(p - 1) if self.categorical else work.feature_values.unsafe_get(p) <= work.feature_values.unsafe_get(p - 1) + Float32(1e-7)):
            p += 1
        return SplitPosition(p - 1, p)
    def partition_samples(mut self, data: TrainingData, mut work: SplitWorkspace, split: SplitRecord) raises -> Int:
        return partition(data, work, split)
    def partition_samples_final(mut self, data: TrainingData, mut work: SplitWorkspace, split: SplitRecord) raises:
        var pos = self.partition_samples(data, work, split)
        if pos != split.pos:
            raise Error("ValueError: split position does not match threshold partition")
    def cat_position_to_split_bitset(self, work: SplitWorkspace, position: Int, missing_left: Bool, mut split: SplitRecord) raises:
        split.left_cat_bitset = CategoryBits(0)
        var needed = position - work.start - (work.n_missing if missing_left else 0)
        if needed <= 0:
            return
        var count = 0
        for r in range(self.sorted_categories.size):
            var c = Int(self.sorted_categories.unsafe_get(r))
            split.left_cat_bitset[c // 32] |= UInt32(1) << UInt32(c % 32)
            count += Int(self.counts.unsafe_get(c))
            if count >= needed:
                break
    def sort_categories(mut self, data: TrainingData, mut work: SplitWorkspace, feature: Int) raises:
        self.categorical = True
        var nc = Int(data.n_categories.unsafe_get(feature * Int(data.n_categories.strides[0])))
        if nc < 0 or nc > 256:
            raise Error("ValueError: categorical best search requires at most 256 categories")
        self.counts = nm.zeros[DType.int64](nm.Shape(nc))
        self.sorted_categories = nm.zeros[DType.int64](nm.Shape(nc))
        var means = nm.zeros[DType.float64](nm.Shape(nc))
        var weights = nm.zeros[DType.float64](nm.Shape(nc))
        var end = work.end - 1
        var i = work.start
        work.n_missing = 0
        while i <= end:
            var x = data.X.unsafe_get(Int(work.samples.unsafe_get(end)) * Int(data.X.strides[0]) + feature * Int(data.X.strides[1]))
            if x != x:
                work.n_missing += 1
                end -= 1
                continue
            x = data.X.unsafe_get(Int(work.samples.unsafe_get(i)) * Int(data.X.strides[0]) + feature * Int(data.X.strides[1]))
            if x != x:
                swap(work.samples, i, end)
                work.n_missing += 1
                end -= 1
            var sample = Int(work.samples.unsafe_get(i))
            var c = Int(data.X.unsafe_get(sample * Int(data.X.strides[0]) + feature * Int(data.X.strides[1])))
            work.feature_values.unsafe_set(i, Float32(c))
            var w = Float64(1)
            if data.sample_weight:
                w = data.sample_weight.value().unsafe_get(sample * Int(data.sample_weight.value().strides[0]))
            self.counts.unsafe_set(c, self.counts.unsafe_get(c) + 1)
            weights.unsafe_set(c, weights.unsafe_get(c) + w)
            means.unsafe_set(c, means.unsafe_get(c) + w * data.y.unsafe_get(sample * Int(data.y.strides[0])))
            i += 1
        for c in range(nc):
            self.sorted_categories.unsafe_set(c, Int64(c))
            if weights.unsafe_get(c) > 0:
                means.unsafe_set(c, means.unsafe_get(c) / weights.unsafe_get(c))
        introsort3(means, self.sorted_categories, 0, nc, introsort_depth(nc))
        var offsets = nm.zeros[DType.int64](nm.Shape(nc))
        var offset = work.start - 1
        for r in range(nc):
            var c = Int(self.sorted_categories.unsafe_get(r))
            offset += Int(self.counts.unsafe_get(c))
            offsets.unsafe_set(c, Int64(offset))
        i = work.start
        while i <= end:
            var c = Int(work.feature_values.unsafe_get(i))
            var dest = Int(offsets.unsafe_get(c))
            if dest > i:
                swap(work.samples, i, dest)
                var value = work.feature_values.unsafe_get(dest)
                work.feature_values.unsafe_set(dest, work.feature_values.unsafe_get(i))
                work.feature_values.unsafe_set(i, value)
                offsets.unsafe_set(c, Int64(dest - 1))
            else:
                i += 1

def introsort_depth(n: Int) -> Int:
    """sklearn's maxd = 2 * floor(log2(n))."""
    var depth = 0
    var size = n
    while size > 1:
        depth += 2
        size //= 2
    return depth

# Reference tie order is observable through equal feature values and equal category means.
# simultaneous_sort(use_three_way_partition=True) from sklearn/utils/_sorting.pyx (BSD-3-Clause).
def introsort3[dtype: DType](mut means: nm.NDArray[dtype], mut ids: Indices, start: Int, end: Int, var depth: Int):
    var lo = start
    while end - lo > 15:
        if depth <= 0:
            for root in range((end - lo - 2) // 2, -1, -1):
                category_sift(means, ids, lo, root, end - lo)
            for count in range(end - lo - 1, 0, -1):
                category_swap(means, ids, lo, lo + count)
                category_sift(means, ids, lo, 0, count)
            return
        depth -= 1
        var a = means.unsafe_get(lo)
        var b = means.unsafe_get(lo + (end - lo) // 2)
        var c = means.unsafe_get(end - 1)
        var pivot = max(min(a, b), min(max(a, b), c))
        var i = lo
        var left = lo
        var right = end
        while i < right:
            var dest = i
            if means.unsafe_get(i) < pivot:
                dest = left
                left += 1
            elif means.unsafe_get(i) > pivot:
                right -= 1
                dest = right
            var saved = means.unsafe_get(i)
            means.unsafe_set(i, means.unsafe_get(dest))
            means.unsafe_set(dest, saved)
            swap(ids, i, dest)
            if dest != right:
                i += 1
        introsort3(means, ids, lo, left, depth)
        lo = right
    for i in range(lo + 1, end):
        var value = means.unsafe_get(i)
        var category = ids.unsafe_get(i)
        var j = i
        while j > lo and means.unsafe_get(j - 1) > value:
            means.unsafe_set(j, means.unsafe_get(j - 1))
            ids.unsafe_set(j, ids.unsafe_get(j - 1))
            j -= 1
        means.unsafe_set(j, value)
        ids.unsafe_set(j, category)

def category_swap[dtype: DType](mut means: nm.NDArray[dtype], mut ids: Indices, i: Int, j: Int):
    var saved = means.unsafe_get(i)
    means.unsafe_set(i, means.unsafe_get(j))
    means.unsafe_set(j, saved)
    swap(ids, i, j)

def category_sift[dtype: DType](mut means: nm.NDArray[dtype], mut ids: Indices, offset: Int, var root: Int, end: Int):
    while True:
        var child = root * 2 + 1
        var largest = root
        if child < end and means.unsafe_get(offset + largest) < means.unsafe_get(offset + child):
            largest = child
        if child + 1 < end and means.unsafe_get(offset + largest) < means.unsafe_get(offset + child + 1):
            largest = child + 1
        if largest == root:
            return
        category_swap(means, ids, offset + root, offset + largest)
        root = largest
