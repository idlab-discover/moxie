"""Native timing driver for tools/benchmark.py: prints `name|n|p|q|k|ms`."""
import numojo as nm
from std.sys import argv
from std.time import perf_counter_ns
from moxie.cross_decomposition import CCA, PLSCanonical, PLSRegression


def data(rows: Int, cols: Int, seed: Int) raises -> nm.NDArray[DType.float64]:
    var a = nm.zeros[DType.float64](nm.Shape(rows, cols))
    var state = UInt64(seed * 2654435761 + 12345)
    for i in range(rows * cols):
        state = state * 6364136223846793005 + 1442695040888963407
        a.unsafe_set(i, Float64(state >> 11) / Float64(UInt64(1) << 53) - 0.5)
    return a^


def main() raises:
    var args = argv()
    var n = Int(String(args[1]))
    var p = Int(String(args[2]))
    var q = Int(String(args[3]))
    var k = Int(String(args[4]))
    var repeats = Int(String(args[5]))
    var X = data(n, p, 1)
    var y = X @ data(p, q, 2)
    var noise = data(n, q, 3)
    for i in range(y.size):
        y.unsafe_set(i, y.unsafe_get(i) + noise.unsafe_get(i))
    var best = Float64(1e300)
    for _ in range(repeats):
        var t0 = perf_counter_ns()
        var e = PLSRegression(k)
        e.fit(X, y)
        best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
    print("PLSRegression|", n, "|", p, "|", q, "|", k, "|", best, sep="")
    best = 1e300
    for _ in range(repeats):
        var t0 = perf_counter_ns()
        var e = PLSCanonical(k, algorithm="svd")
        e.fit(X, y)
        best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
    print("PLSCanonical-svd|", n, "|", p, "|", q, "|", k, "|", best, sep="")
    best = 1e300
    for _ in range(repeats):
        var t0 = perf_counter_ns()
        var e = CCA(k)
        e.fit(X, y)
        best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
    print("CCA|", n, "|", p, "|", q, "|", k, "|", best, sep="")
