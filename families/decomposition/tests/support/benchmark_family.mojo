"""Native timed fit loops for pyperf: benchmark_family ESTIMATOR f32|f64 n p loops."""
from std.sys import argv
from std.math import sin
from std.time import perf_counter_ns
from numojo import NDArray, Shape, zeros
from moxie.decomposition import PCA, TruncatedSVD, FactorAnalysis, FastICA, IncrementalPCA


def data[dtype: DType](n: Int, p: Int) raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(n, p))
    for i in range(n):
        for j in range(p):
            x.unsafe_set(i * p + j, Scalar[dtype](sin(Float64(i * (j + 1)) * 0.37) + Float64((i * 13 + j * 7) % 17) / 17))
    return x^


def bench[dtype: DType](name: String, n: Int, p: Int, loops: Int) raises:
    var x = data[dtype](n, p)
    var checksum: Float64 = 0
    var start = perf_counter_ns()
    for _ in range(loops):
        if name == "pca":
            var m = PCA[dtype](3, svd_solver="full")
            m.fit(x)
            checksum += Float64(m.explained_variance_.value().unsafe_get(0))
        elif name == "incremental_pca":
            var m = IncrementalPCA[dtype](3, batch_size=max(3, n // 4))
            m.fit(x)
            checksum += Float64(m.explained_variance_.value().unsafe_get(0))
        elif name == "truncated_svd":
            var m = TruncatedSVD[dtype](3, random_state=0)
            m.fit(x)
            checksum += Float64(m.singular_values_.value().unsafe_get(0))
        elif name == "factor_analysis":
            var m = FactorAnalysis[dtype](3)
            m.fit(x)
            checksum += Float64(m.noise_variance_.value().unsafe_get(0))
        elif name == "fastica":
            var m = FastICA[dtype](3, random_state=0)
            m.fit(x)
            checksum += Float64(m.components_.value().unsafe_get(0))
        else:
            raise Error("ValueError: unknown estimator " + name)
    print(Float64(perf_counter_ns() - start) / 1e9, checksum)


def main() raises:
    var args = argv()
    if String(args[2]) == "f32": bench[DType.float32](String(args[1]), Int(args[3]), Int(args[4]), Int(args[5]))
    else: bench[DType.float64](String(args[1]), Int(args[3]), Int(args[4]), Int(args[5]))
