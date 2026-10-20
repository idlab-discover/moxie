"""Timing driver: median fit time of LinearRegression on synthetic data.

Usage: benchmark <n_samples> <n_features> <repeats> <positive 0/1>. Prints
`dtype n p seconds` lines (median over repeats) for Float64 and Float32.
"""
import numojo as nm
from numojo import Shape
from std.sys import argv
from std.time import perf_counter_ns
from moxie.linear_model import LinearRegression


def time_fit[dtype: DType](n: Int, p: Int, repeats: Int, positive: Bool) raises:
    var X = nm.zeros[dtype](Shape(n, p))
    var y = nm.zeros[dtype](Shape(n))
    var state: UInt64 = 88172645463325252
    for i in range(n * p):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        X.unsafe_set(i, (Float64(state % 1000000) / 500000.0 - 1.0).cast[dtype]())
    for i in range(n):
        var total = Float64(0)
        for j in range(p):
            total += Float64(X.unsafe_get(i * p + j)) * Float64(j % 5 - 2)
        y.unsafe_set(i, total.cast[dtype]())
    var times = List[Float64]()
    for _ in range(repeats):
        var model = LinearRegression[dtype](positive=positive)
        var start = perf_counter_ns()
        model.fit(X, y)
        times.append(Float64(perf_counter_ns() - start) / 1e9)
    sort(times)
    print(String(dtype), n, p, times[len(times) // 2])


def main() raises:
    var args = argv()
    var n = Int(args[1])
    var p = Int(args[2])
    var repeats = Int(args[3])
    var positive = Int(args[4]) == 1
    time_fit[DType.float64](n, p, repeats, positive)
    time_fit[DType.float32](n, p, repeats, positive)
