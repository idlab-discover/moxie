"""Test-only composition of statistics and scale correction passes, no estimator."""
from std.sys import argv
from std.math import abs, isnan
from std.testing import assert_true
from numojo import zeros, Shape, NDArrayStrides
from moxie.preprocessing._moments import nanmean_at, nanmean_std_at

def evaluate[dtype: DType]() raises:
    var args = argv()
    var n = Int(args[2])
    var p = Int(args[3])
    var axis = Int(args[4])
    var layout = Int(args[5])
    var center = Int(args[6]) != 0
    var scaling = Int(args[7]) != 0
    var owner = zeros[dtype](Shape(3 * n * p + 20))
    var rs = p
    var cs = 1
    var offset = 0
    if layout == 1:
        rs = 3 * p
        cs = 2
        offset = 3
    elif layout == 2:
        rs = -3 * p
        cs = -2
        offset = 3 + (n - 1) * 3 * p + (p - 1) * 2
    elif layout >= 3:
        rs = 1
        cs = n
        if layout == 4:
            rs = -1
            cs = -n
            offset = n * p - 1
    var x = owner.view_with_layout(Shape(n, p), NDArrayStrides(rs, cs), offset)
    var cursor = 8
    for i in range(n):
        for j in range(p):
            x.unsafe_set(i * rs + j * cs, Float64(args[cursor]).cast[dtype]())
            cursor += 1
    var groups = p if axis == 0 else n
    var length = n if axis == 0 else p
    var step = rs if axis == 0 else cs
    var outer_step = cs if axis == 0 else rs
    var means = zeros[dtype](Shape(groups))
    var scales = zeros[dtype](Shape(groups))
    var corrections = zeros[dtype](Shape(groups))
    for j in range(groups):
        var stats = nanmean_std_at(x, axis, j)
        means.unsafe_set(j, stats.mean)
        scales.unsafe_set(j, stats.stddev)
        print('mean', Float64(stats.mean))
        print('stddev', Float64(stats.stddev))
        print('count', stats.count)
        var mean_only = nanmean_at(x, axis, j)
        assert_true(mean_only == stats.mean or (isnan(mean_only) and isnan(stats.mean)))
    # Reductions must borrow, without modifying even a signed-stride view.
    cursor = 8
    for i in range(n):
        for j in range(p):
            var expected = Float64(args[cursor]).cast[dtype]()
            var actual = x.unsafe_get(i * rs + j * cs)
            assert_true(actual == expected or (isnan(actual) and isnan(expected)))
            cursor += 1
    if center:
        for j in range(groups):
            for i in range(length):
                var at = j * outer_step + i * step
                x.unsafe_set(at, x.unsafe_get(at) - means.unsafe_get(j))
        var correct = False
        for j in range(groups):
            var residual_mean = nanmean_at(x, axis, j)
            corrections.unsafe_set(j, residual_mean)
            print('mean1', Float64(residual_mean))
            correct = correct or not (abs(residual_mean) <= 1e-8)
        print('correct1', Int(correct))
        if correct:
            for j in range(groups):
                for i in range(length):
                    var at = j * outer_step + i * step
                    x.unsafe_set(at, x.unsafe_get(at) - corrections.unsafe_get(j))
    if scaling:
        var eps = Float64(2.220446049250313e-16)
        comptime if dtype == DType.float32:
            eps = 1.1920928955078125e-7
        for j in range(groups):
            var divisor = scales.unsafe_get(j)
            if divisor < (10 * eps).cast[dtype]():
                divisor = 1
            for i in range(length):
                var at = j * outer_step + i * step
                x.unsafe_set(at, x.unsafe_get(at) / divisor)
        if center:
            var correct = False
            for j in range(groups):
                var residual_mean = nanmean_at(x, axis, j)
                corrections.unsafe_set(j, residual_mean)
                print('mean2', Float64(residual_mean))
                correct = correct or not (abs(residual_mean) <= 1e-8)
            print('correct2', Int(correct))
            if correct:
                for j in range(groups):
                    for i in range(length):
                        var at = j * outer_step + i * step
                        x.unsafe_set(at, x.unsafe_get(at) - corrections.unsafe_get(j))
    for i in range(n):
        for j in range(p):
            print('output', Float64(x.unsafe_get(i * rs + j * cs)))

def main() raises:
    if argv()[1] == 'f32':
        evaluate[DType.float32]()
    else:
        evaluate[DType.float64]()
