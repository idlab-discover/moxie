from std.sys import argv
from std.time import perf_counter_ns
from numojo import zeros, Shape
from moxie.decomposition import randomized_svd
from moxie.decomposition._svd_engine import dense_svd, golub_kahan_svd


def bench[dtype: DType](m: Int, n: Int, repeats: Int, mode: String) raises:
    var x = zeros[dtype](Shape(m, n))
    for i in range(m*n):
        x.unsafe_set(i, SIMD[dtype, 1](Float64((i*17)%101-50)/101))
    var checksum: Float64 = 0
    var start = perf_counter_ns()
    for rep in range(repeats):
        x.unsafe_set(0, SIMD[dtype, 1](Float64(rep%7)/10))
        if mode == "random":
            var result = randomized_svd(x, min(4,n), random_state=42, n_oversamples=3, n_iter=2)
            checksum += Float64(result.singular_values.unsafe_get(0))
        elif mode == "gesvd":
            var result = golub_kahan_svd(x)
            checksum += Float64(result.singular_values.unsafe_get(0))
        else:
            var result = dense_svd(x)
            checksum += Float64(result.singular_values.unsafe_get(0))
    print(Float64(perf_counter_ns()-start)/1e9, checksum)


def main() raises:
    var args = argv()
    if args[1] == "f32":
        bench[DType.float32](Int(args[2]),Int(args[3]),Int(args[4]),args[5])
    else:
        bench[DType.float64](Int(args[2]),Int(args[3]),Int(args[4]),args[5])
