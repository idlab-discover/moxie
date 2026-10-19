"""Timing driver: fit GaussianMixture on a deterministic dataset.

Usage: benchmark <n> <d> <k> <covariance_type> <repeats>. Prints one elapsed
seconds value per repeat (fit only; data generation excluded).
"""
from numojo import Shape, zeros
from std.sys import argv
from std.time import perf_counter_ns
from moxie._random import RandomState
from moxie.mixture import GaussianMixture


def main() raises:
    var args = argv()
    var n = Int(args[1])
    var d = Int(args[2])
    var k = Int(args[3])
    var kind = String(args[4])
    var repeats = Int(args[5])
    var rng = RandomState(0)
    var X = zeros[DType.float64](Shape(n, d))
    for i in range(n):
        var center = Float64((i % k) * 5)
        for f in range(d):
            X.unsafe_set(i * d + f, center + rng.standard_normal())
    for _ in range(repeats):
        var model = GaussianMixture[DType.float64](k, covariance_type=kind, random_state=0, max_iter=100)
        var start = perf_counter_ns()
        _ = model.fit(X)
        var elapsed = Float64(perf_counter_ns() - start) * 1e-9
        print(elapsed, model.n_iter_)
