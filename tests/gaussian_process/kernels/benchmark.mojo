"""Native timing of kernel evaluation with gradients (driven by tools/benchmark.py)."""
from numojo import Shape, zeros
from std.sys import argv
from std.time import perf_counter_ns
from moxie.gaussian_process.kernels import ConstantKernel, Kernel, Matern, RBF, WhiteKernel


def main() raises:
    var args = argv()
    var n = Int(String(args[1]))
    var d = Int(String(args[2]))
    var repeats = Int(String(args[3]))
    var X = zeros[DType.float64](Shape(n, d))
    var state = 12345
    for i in range(n * d):
        state = (state * 1103515245 + 12345) % 2147483648
        X.unsafe_set(i, Float64(state) / 2147483648.0)
    var kernels = List[Kernel]()
    kernels.append(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1))
    kernels.append(Kernel(Matern(1.0, nu=2.5)))
    kernels.append(Kernel(Matern(1.0, nu=0.75)))
    var names: List[String] = ["C*RBF+White", "Matern2.5", "Matern0.75"]
    for k in range(len(kernels)):
        var best = Float64.MAX
        for _ in range(repeats):
            var start = perf_counter_ns()
            var e = kernels[k].evaluate(X, eval_gradient=True)
            var elapsed = Float64(perf_counter_ns() - start) / 1e9
            _ = e.covariance.size
            if elapsed < best:
                best = elapsed
        print("T|" + names[k] + "|" + String(best))
