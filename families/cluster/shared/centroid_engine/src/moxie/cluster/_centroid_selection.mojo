"""Private relocation selection, matching NumPy's four-lane argselect order.

Portable scalar adaptation of Intel x86-simd-sort (BSD-3-Clause); see
LICENSES/x86-simd-sort.txt. No SIMD runtime or Python dependency is required.
"""
import numojo as nm
from std.utils.numerics import inf
from moxie.cluster._contracts import Labels


@always_inline
def _value[dtype: DType](values: nm.NDArray[dtype], index: Int64) raises -> SIMD[dtype, 1]:
    if index < 0:
        return inf[dtype]()
    return values.unsafe_get(Int(index) * Int(values.strides[0]))


@always_inline
def _swap(mut indices: Labels, a: Int, b: Int) raises:
    var temporary = indices.unsafe_get(a)
    indices.unsafe_set(a, indices.unsafe_get(b))
    indices.unsafe_set(b, temporary)


@always_inline
def _exchange[dtype: DType](values: nm.NDArray[dtype], mut indices: Labels, a: Int, b: Int) raises:
    # Equal keys keep their lanes. Reversing merge partners still reorders ties.
    if _value(values, indices.unsafe_get(b)) < _value(values, indices.unsafe_get(a)):
        _swap(indices, a, b)


def _network[dtype: DType](values: nm.NDArray[dtype], mut indices: Labels, left: Int, right: Int) raises:
    var n = right - left
    var size = 4
    while size < n:
        size *= 2
    var work = nm.full[DType.int64](nm.Shape(size), -1)
    for i in range(n):
        work.unsafe_set(i, indices.unsafe_get(left + i))
    # Three stages sort each four-lane register.
    for i in range(0, size, 4):
        _exchange(values, work, i, i + 1)
        _exchange(values, work, i + 2, i + 3)
        _exchange(values, work, i, i + 3)
        _exchange(values, work, i + 1, i + 2)
        _exchange(values, work, i, i + 1)
        _exchange(values, work, i + 2, i + 3)
    var width = 8
    while width <= size:
        for start in range(0, size, width):
            # Reverse upper registers for comparison, then restore their lanes.
            for i in range(width // 2):
                _exchange(values, work, start + i, start + width - i - 1)
            var step = width // 4
            while step >= 4:
                for block in range(start, start + width, 2 * step):
                    for i in range(step):
                        _exchange(values, work, block + i, block + i + step)
                step //= 2
            for block in range(start, start + width, 4):
                _exchange(values, work, block, block + 2)
                _exchange(values, work, block + 1, block + 3)
                _exchange(values, work, block, block + 1)
                _exchange(values, work, block + 2, block + 3)
        width *= 2
    for i in range(n):
        indices.unsafe_set(left + i, work.unsafe_get(i))


def _store_block[dtype: DType](values: nm.NDArray[dtype], mut indices: Labels, block: Labels, pivot: SIMD[dtype, 1], mut left: Int, mut right: Int) raises:
    # Four unrolled vectors. AVX2 compression keeps low lanes in order and
    # reverses high lanes; preserving that permutation is observable in ties.
    for start in range(0, 16, 4):
        var low_count = 0
        var high_count = 0
        for lane in range(4):
            var index = block.unsafe_get(start + lane)
            if _value(values, index) < pivot:
                indices.unsafe_set(left + low_count, index)
                low_count += 1
            else:
                indices.unsafe_set(right + 3 - high_count, index)
                high_count += 1
        left += low_count
        right -= high_count


def _load_block(indices: Labels, start: Int, mut block: Labels) raises:
    for i in range(16):
        block.unsafe_set(i, indices.unsafe_get(start + i))



def _heap_adjust[dtype: DType](values: nm.NDArray[dtype], mut indices: Labels, start: Int, length: Int, hole: Int, value: Int64) raises:
    var position = hole
    var child = 2 * (position + 1)
    while child < length:
        if _value(values, indices.unsafe_get(start + child)) < _value(values, indices.unsafe_get(start + child - 1)):
            child -= 1
        indices.unsafe_set(start + position, indices.unsafe_get(start + child))
        position = child
        child = 2 * (position + 1)
    if child == length:
        indices.unsafe_set(start + position, indices.unsafe_get(start + child - 1))
        position = child - 1
    while position > hole:
        var parent = (position - 1) // 2
        if not (_value(values, indices.unsafe_get(start + parent)) < _value(values, value)):
            break
        indices.unsafe_set(start + position, indices.unsafe_get(start + parent))
        position = parent
    indices.unsafe_set(start + position, value)


def _heap_sort[dtype: DType](values: nm.NDArray[dtype], mut indices: Labels, left: Int, right: Int) raises:
    var length = right - left
    var parent = (length - 2) // 2
    while parent >= 0:
        _heap_adjust(values, indices, left, length, parent, indices.unsafe_get(left + parent))
        parent -= 1
    while length > 1:
        length -= 1
        var value = indices.unsafe_get(left + length)
        indices.unsafe_set(left + length, indices.unsafe_get(left))
        _heap_adjust(values, indices, left, length, 0, value)


def _sort_loop[dtype: DType](values: nm.NDArray[dtype], mut indices: Labels, left: Int, right: Int, depth: Int) raises:
    # Conventional introsort with the GNU std::sort traversal/tie policy used
    # by the oracle's emergency fallback: recurse right, iterate left.
    var end = right
    var remaining = depth
    while end - left > 16:
        if remaining == 0:
            _heap_sort(values, indices, left, end)
            return
        remaining -= 1
        var a = left + 1
        var b = left + (end - left) // 2
        var c = end - 1
        var va = _value(values, indices.unsafe_get(a))
        var vb = _value(values, indices.unsafe_get(b))
        var vc = _value(values, indices.unsafe_get(c))
        var median = b
        if va < vb:
            if vb < vc:
                median = b
            elif va < vc:
                median = c
            else:
                median = a
        else:
            if va < vc:
                median = a
            elif vb < vc:
                median = c
        _swap(indices, left, median)
        var pivot = _value(values, indices.unsafe_get(left))
        var lo = left + 1
        var hi = end
        while True:
            while _value(values, indices.unsafe_get(lo)) < pivot:
                lo += 1
            hi -= 1
            while pivot < _value(values, indices.unsafe_get(hi)):
                hi -= 1
            if lo >= hi:
                break
            _swap(indices, lo, hi)
            lo += 1
        _sort_loop(values, indices, lo, end, remaining)
        end = lo


def _fallback_sort[dtype: DType](values: nm.NDArray[dtype], mut indices: Labels, left: Int, right: Int) raises:
    var bits = right - left
    var depth = 0
    while bits > 1:
        bits //= 2
        depth += 2
    _sort_loop(values, indices, left, right, depth)
    # Stable insertion cleanup of the small partitions left by introsort.
    for i in range(left + 1, right):
        var value = indices.unsafe_get(i)
        var j = i
        while j > left and _value(values, value) < _value(values, indices.unsafe_get(j - 1)):
            indices.unsafe_set(j, indices.unsafe_get(j - 1))
            j -= 1
        indices.unsafe_set(j, value)

def relocation_order[dtype: DType](values: nm.NDArray[dtype], count: Int) raises -> Labels:
    """Indices of relocated samples, in the oracle's reversed partition order.

    Input is the engine's finite/nonnegative squared-residual vector. Selection
    order follows NumPy 2.5.3's AVX2 backend, independently of the host CPU.
    NumPy does not guarantee identical partition order across other backends.
    """
    if values.ndim != 1 or count < 1 or count > values.size:
        raise Error("ValueError: invalid relocation selection dimensions")
    var n = values.size
    var indices = nm.zeros[DType.int64](nm.Shape(n))
    for i in range(n):
        indices.unsafe_set(i, Int64(i))
    var left = 0
    var right = n
    var kth = n - count
    var bits = n
    var depth = 0
    while bits > 1:
        bits //= 2
        depth += 2
    while right - left > 256:
        if depth == 0:
            _fallback_sort(values, indices, left, right)
            left = right
            break
        depth -= 1
        var store_left, pivot, smallest, biggest = _partition(values, indices, left, right)
        if pivot != smallest and kth < store_left:
            right = store_left
        elif pivot != biggest and kth >= store_left:
            left = store_left
        else:
            # All equal keys on the selected side: the reference stops here.
            left = right
            break
    if left < right:
        if depth == 0:
            _fallback_sort(values, indices, left, right)
        else:
            _network(values, indices, left, right)
    var result = nm.zeros[DType.int64](nm.Shape(count))
    for i in range(count):
        result.unsafe_set(i, indices.unsafe_get(n - 1 - i))
    return result^


def _argsort_range[dtype: DType](values: nm.NDArray[dtype], mut indices: Labels, left: Int, right: Int, depth: Int) raises:
    if depth == 0:
        _fallback_sort(values, indices, left, right)
        return
    if right - left <= 256:
        _network(values, indices, left, right)
        return
    var store_left, pivot, smallest, biggest = _partition(values, indices, left, right)
    if pivot != smallest:
        _argsort_range(values, indices, left, store_left, depth - 1)
    if pivot != biggest:
        _argsort_range(values, indices, store_left, right, depth - 1)


def count_order[dtype: DType](counts: nm.NDArray[dtype]) raises -> Labels:
    """Ascending finite cluster weights, NumPy 2.5.3 AVX2 argsort tie order."""
    if counts.ndim != 1:
        raise Error("ValueError: cluster counts must be a vector")
    var order = nm.zeros[DType.int64](nm.Shape(counts.size))
    for i in range(counts.size): order.unsafe_set(i, Int64(i))
    var bits = counts.size
    var depth = 0
    while bits > 1:
        bits //= 2
        depth += 2
    var sorted_input = True
    for i in range(1, counts.size):
        if _value(counts, Int64(i)) < _value(counts, Int64(i - 1)):
            sorted_input = False
            break
    if not sorted_input:
        _argsort_range(counts, order, 0, counts.size, depth)
    return order^


def _partition[dtype: DType](values: nm.NDArray[dtype], mut indices: Labels, left: Int, right: Int) raises -> Tuple[Int, SIMD[dtype, 1], SIMD[dtype, 1], SIMD[dtype, 1]]:
    var first = nm.zeros[DType.int64](nm.Shape(16))
    var last = nm.zeros[DType.int64](nm.Shape(16))
    var block = nm.zeros[DType.int64](nm.Shape(16))
    var probes = nm.zeros[DType.int64](nm.Shape(4))
    # Median of four equally spaced sample values, upper median on ties.
    var spacing = (right - left - 1) // 4
    for i in range(4):
        probes.unsafe_set(i, indices.unsafe_get(left + (i + 1) * spacing))
    _network(values, probes, 0, 4)
    var pivot = _value(values, probes.unsafe_get(2))
    var smallest = inf[dtype]()
    var biggest = -inf[dtype]()
    for i in range(left, right):
        var value = _value(values, indices.unsafe_get(i))
        if value < smallest:
            smallest = value
        if value > biggest:
            biggest = value
    var lo = left
    var hi = right
    for _ in range((hi - lo) % 16):
        if _value(values, indices.unsafe_get(lo)) >= pivot:
            hi -= 1
            _swap(indices, lo, hi)
        else:
            lo += 1
    _load_block(indices, lo, first)
    _load_block(indices, hi - 16, last)
    var store_left = lo
    var store_right = hi - 4
    lo += 16
    hi -= 16
    while lo != hi:
        if store_right + 4 - hi < lo - store_left:
            hi -= 16
            _load_block(indices, hi, block)
        else:
            _load_block(indices, lo, block)
            lo += 16
        _store_block(values, indices, block, pivot, store_left, store_right)
    _store_block(values, indices, first, pivot, store_left, store_right)
    _store_block(values, indices, last, pivot, store_left, store_right)
    return store_left, pivot, smallest, biggest
