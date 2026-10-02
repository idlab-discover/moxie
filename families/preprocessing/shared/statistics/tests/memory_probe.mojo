"""Isolated native workloads; memory controls use identically touched input."""
from std.sys import argv
from std.time import perf_counter_ns
from std.math import isnan, sqrt
from std.testing import assert_true
from numojo import zeros, Shape, NDArray, NDArrayStrides
from moxie.preprocessing._moments import MomentsStateV1, update_moments_weighted, ExtremaStateV1, update_extrema, robust_quantiles, nanmean_at, nanmean_std_at

@no_inline
def simple_loop[dtype: DType](x: NDArray[dtype], w: NDArray[dtype]) raises -> Float64:
    var checksum = Float64(0)
    for j in range(x.shape[1]):
        var total = Float64(0)
        var count = Float64(0)
        for i in range(x.shape[0]):
            var v = Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
            if not isnan(v):
                var weight = Float64(w.unsafe_get(i))
                total += weight * v
                count += weight
        var center = total / count
        var correction = Float64(0)
        var squares = Float64(0)
        for i in range(x.shape[0]):
            var v = Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
            if not isnan(v):
                var d = v - center
                var weight = Float64(w.unsafe_get(i))
                correction += weight * d
                squares += weight * (d * d)
        checksum += center + (squares - correction * correction / count) / count
    return checksum


def bench[dtype: DType](mode: String, n: Int, p: Int, tile: Int, layout: String, reps: Int) raises:
    var backing_size = n * p
    if layout == 'tiny':
        backing_size = 8388608
    var owner = zeros[dtype](Shape(backing_size))
    for k in range(backing_size):
        owner.unsafe_set(k, SIMD[dtype, 1]((k * 13) % 101 - 50) / 8)
    var offset = 0
    var row_stride = p
    var col_stride = 1
    if layout == 'tiny':
        offset = backing_size - 17
        row_stride = -11
        col_stride = -2
    elif layout == 'reverse':
        offset = n * p - 1
        row_stride = -p
        col_stride = -1
    var x = owner.view_with_layout(Shape(n, p), NDArrayStrides(row_stride, col_stride), offset)
    var nan = Float64('nan').cast[dtype]()
    for i in range(0, n, 17):
        x.unsafe_set(i * row_stride, nan)
    var w = zeros[dtype](Shape(n))
    for i in range(n):
        w.unsafe_set(i, SIMD[dtype, 1](-1 if i % 5 == 0 else 2))
    var checksum = Float64(0)
    var start = perf_counter_ns()
    for rep in range(reps):
        if mode == 'moments':
            var state = MomentsStateV1(p)
            update_moments_weighted(x, w, state, tile_rows=tile)
            for j in range(p):
                checksum += state.mean.unsafe_get(j) + state.variance.unsafe_get(j)
        elif mode == 'extrema':
            var state = ExtremaStateV1(p)
            update_extrema(x, state)
            checksum += state.minimum.unsafe_get(0) + state.maximum.unsafe_get(0)
        elif mode == 'quantiles':
            var q = robust_quantiles(x)
            checksum += q.median.unsafe_get(0) + q.lower.unsafe_get(0) + q.upper.unsafe_get(0)
        elif mode == 'precision0' or mode == 'precision1':
            var axis = 0 if mode == 'precision0' else 1
            var groups = p if axis == 0 else n
            for j in range(groups):
                var stats = nanmean_std_at(x, axis, j)
                checksum += Float64(stats.mean) + Float64(stats.stddev)
                checksum += Float64(nanmean_at(x, axis, j))
        elif mode == 'precision_loop':
            for j in range(p):
                var total = SIMD[dtype, 1](0)
                var count = 0
                for i in range(n):
                    var value = x.unsafe_get(i * row_stride + j * col_stride)
                    if not isnan(value):
                        total += value
                        count += 1
                var center = (Float64(total) / Float64(count)).cast[dtype]()
                var squares = SIMD[dtype, 1](0)
                for i in range(n):
                    var value = x.unsafe_get(i * row_stride + j * col_stride)
                    if not isnan(value):
                        var delta = value - center
                        squares += delta * delta
                checksum += Float64(center) + Float64(sqrt((Float64(squares) / Float64(count)).cast[dtype]()))
        elif mode == 'forbidden':
            # Deliberate prohibited buffer: touched and live until its scan ends.
            # Tiny-view case copies the whole backing just like NDArray.copy().
            var bad = zeros[DType.float64](Shape(backing_size))
            for k in range(backing_size):
                bad.unsafe_set(k, Float64(owner.unsafe_get(k)))
            for k in range(backing_size):
                if not isnan(bad.unsafe_get(k)):
                    checksum += bad.unsafe_get(k)
        elif mode == 'loop':
            checksum += simple_loop(x, w)
        elif mode != 'control':
            raise Error('unknown memory operation')
    var elapsed = Float64(perf_counter_ns() - start) / 1e9
    # No snapshots or checksums masking mutations: inspect every logical element.
    for i in range(n):
        for j in range(p):
            var v = x.unsafe_get(i * row_stride + j * col_stride)
            var k = offset + i * row_stride + j * col_stride
            if i % 17 == 0 and j == 0:
                assert_true(isnan(v))
            else:
                assert_true(v == SIMD[dtype, 1]((k * 13) % 101 - 50) / 8)
        assert_true(w.unsafe_get(i) == SIMD[dtype, 1](-1 if i % 5 == 0 else 2))
    print('elapsed', elapsed)
    print('checksum', checksum)


def main() raises:
    var args = argv()
    if args[1] == 'f32':
        bench[DType.float32](args[2], Int(args[3]), Int(args[4]), Int(args[5]), args[6], Int(args[7]))
    else:
        bench[DType.float64](args[2], Int(args[3]), Int(args[4]), Int(args[5]), args[6], Int(args[7]))
