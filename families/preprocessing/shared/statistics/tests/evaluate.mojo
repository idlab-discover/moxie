"""Native differential probe; host supplies data and checks pinned reference."""
from std.sys import argv
from std.testing import assert_true
from std.math import isnan
from numojo import zeros, Shape, NDArrayStrides
from moxie.preprocessing._moments import MomentsStateV1, update_moments, update_moments_weighted, update_moments_scalar, ExtremaStateV1, update_extrema, robust_quantiles

def evaluate[dtype: DType]() raises:
    var args = argv()
    var n = Int(args[2])
    var p = Int(args[3])
    var mode = Int(args[4])
    var tile = Int(args[5])
    var track = Int(args[6]) != 0
    var layout = Int(args[7])
    var qmin = Float64(args[8])
    var qmax = Float64(args[9])
    var scalar = Float64(args[10])
    var owner = zeros[dtype](Shape(n * p * 3 + 20))
    var row_stride = p
    var col_stride = 1
    var offset = 0
    if layout == 1:
        row_stride = 3 * p
        col_stride = 2
        offset = 3
    elif layout == 2:
        row_stride = -3 * p
        col_stride = -2
        offset = 3 + (n - 1) * 3 * p + (p - 1) * 2
    elif layout == 3:
        row_stride = 1
        col_stride = n
    var x = owner.view_with_layout(Shape(n, p), NDArrayStrides(row_stride, col_stride), offset)
    var cursor = 11
    for i in range(n):
        for j in range(p):
            x.unsafe_set(i * row_stride + j * col_stride, Float64(args[cursor]).cast[dtype]())
            cursor += 1
    var w_owner = zeros[dtype](Shape(2 * n + 2))
    var w = w_owner.view_with_layout(Shape(n), NDArrayStrides(-2), 2 * n)
    for i in range(n):
        w.unsafe_set(-2 * i, Float64(args[cursor]).cast[dtype]())
        cursor += 1
    var state = MomentsStateV1(p, track_variance=track)
    for j in range(p):
        state.count.unsafe_set(j, Float64(args[cursor]))
        state.mean.unsafe_set(j, Float64(args[cursor + p]))
        state.variance.unsafe_set(j, Float64(args[cursor + 2 * p]))
        cursor += 1
    if mode == 0:
        update_moments(x, state, tile_rows=tile)
    elif mode == 1:
        update_moments_weighted(x, w, state, tile_rows=tile)
    else:
        update_moments_scalar(x, scalar, state, tile_rows=tile)
    var extrema = ExtremaStateV1(p)
    update_extrema(x, extrema)
    var quantiles = robust_quantiles(x, q_min=qmin, q_max=qmax)
    print('scalar', Int(state.scalar_count()))
    for j in range(p):
        print('count', state.count.unsafe_get(j))
        print('mean', state.mean.unsafe_get(j))
        print('variance', state.variance.unsafe_get(j))
        print('sum', state.new_sum.unsafe_get(j))
        print('min', extrema.minimum.unsafe_get(j))
        print('max', extrema.maximum.unsafe_get(j))
        print('abs', extrema.max_abs.unsafe_get(j))
        print('lower', quantiles.lower.unsafe_get(j))
        print('median', quantiles.median.unsafe_get(j))
        print('upper', quantiles.upper.unsafe_get(j))
        if track:
            print('constant', Int(state.constant_at(j)))
    # Borrowed data and weights must survive bit-for-bit (NaNs compare by class).
    cursor = 11
    for i in range(n):
        for j in range(p):
            var expected = Float64(args[cursor]).cast[dtype]()
            var actual = x.unsafe_get(i * row_stride + j * col_stride)
            assert_true(actual == expected or (isnan(actual) and isnan(expected)))
            cursor += 1
    for i in range(n):
        assert_true(w.unsafe_get(-2 * i) == Float64(args[cursor]).cast[dtype]())
        cursor += 1

def main() raises:
    var args = argv()
    if args[1] == 'f32':
        evaluate[DType.float32]()
    else:
        evaluate[DType.float64]()
