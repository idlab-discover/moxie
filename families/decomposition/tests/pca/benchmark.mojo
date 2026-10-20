"""Native timed region for pyperf; compilation and startup are outside timing."""
from std.sys import argv
from std.time import perf_counter_ns
from numojo import Shape, zeros
from moxie.decomposition import PCA


def bench[dtype: DType](m: Int, n: Int, loops: Int) raises:
    var X = zeros[dtype](Shape(m, n))
    for i in range(m * n):
        X.unsafe_set(i, SIMD[dtype, 1](Float64((i * 17) % 101 - 50) / 101))
    var model = PCA[dtype](min(3, n), svd_solver="full")
    var checksum: Float64 = 0
    var start = perf_counter_ns()
    for iteration in range(loops):
        X.unsafe_set(0, SIMD[dtype, 1](Float64(iteration % 7) / 10))
        model.fit(X)
        checksum += Float64(model.explained_variance_.value().unsafe_get(0))
    print(Float64(perf_counter_ns() - start) / 1e9, checksum)


def main() raises:
    var args = argv()
    if args[1] == "f32": bench[DType.float32](Int(args[2]), Int(args[3]), Int(args[4]))
    else: bench[DType.float64](Int(args[2]), Int(args[3]), Int(args[4]))
