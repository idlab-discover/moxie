"""Native GPR fit/predict timing (driven by tools/benchmark.py). Args: n d repeats."""
from numojo import Shape, zeros
from std.math import sin
from std.sys import argv
from std.time import perf_counter_ns
from moxie.gaussian_process import GaussianProcessRegressor
from moxie.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


def main() raises:
    var args = argv()
    var n = Int(String(args[1]))
    var d = Int(String(args[2]))
    var repeats = Int(String(args[3]))
    var X = zeros[DType.float64](Shape(n, d))
    var y = zeros[DType.float64](Shape(n))
    var state = 12345
    for i in range(n * d):
        state = (state * 1103515245 + 12345) % 2147483648
        X.unsafe_set(i, Float64(state) / 2147483648.0 * 6.0 - 3.0)
    for i in range(n):
        y.unsafe_set(i, sin(X.unsafe_get(i * d)))
    var best_fit = Float64.MAX
    var best_predict = Float64.MAX
    for _ in range(repeats):
        var gpr = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-2))
        var start = perf_counter_ns()
        _ = gpr.fit(X, y)
        best_fit = min(best_fit, Float64(perf_counter_ns() - start) / 1e9)
        start = perf_counter_ns()
        var r = gpr.predict(X, return_std=True)
        best_predict = min(best_predict, Float64(perf_counter_ns() - start) / 1e9)
        _ = r.mean.size
    print("T|fit|" + String(best_fit))
    print("T|predict_std|" + String(best_predict))
