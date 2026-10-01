"""Native timed regions; pyperf supplies calibration and process management."""
from std.sys import argv
from std.time import perf_counter_ns
from numojo import Shape, NDArray
import numojo as nm
from mojo_linalg import cholesky
from moxie._linalg import covariance_factor


def run[dtype: DType](mode: String, n: Int, rounds: Int, strided: Bool) raises:
    var backing = nm.zeros[dtype](Shape(n * n * 2 + 4))
    var a = backing.view_with_layout(Shape(n, n), nm.NDArrayStrides(n * 2, 2), 2) if strided else nm.zeros[dtype](Shape(n, n))
    var rhs = nm.zeros[dtype](Shape(n, 4))
    var r = Int(a.strides[0])
    var c = Int(a.strides[1])
    for i in range(n):
        for j in range(n):
            a.unsafe_set(i * r + j * c, Scalar[dtype](n + 1) if i == j else Scalar[dtype](0.25))
        for j in range(4):
            rhs.unsafe_set(i * 4 + j, Scalar[dtype](i + j + 1))
    var checksum = Float64(0)
    var start = perf_counter_ns()
    for k in range(rounds):
        # Prevent a constant-input workload; initialization is outside timing.
        a.unsafe_set(0, Scalar[dtype](n + 1) + Scalar[dtype](k % 2) * 0.125)
        if mode == "backend":
            try:
                var factor = cholesky(a)
                var x = factor.solve(rhs)
                checksum += Float64(x.unsafe_get(0))
            except error:
                raise Error(String(error))
        elif mode == "adapter":
            var factor = covariance_factor(a)
            var x = factor.solve(rhs)
            checksum += Float64(x.unsafe_get(0))
        else:
            raise Error("unknown benchmark mode")
    print(Float64(perf_counter_ns() - start) / 1e9)
    print(checksum)


def main() raises:
    var args = argv()
    if args[1] == "32":
        run[DType.float32](args[2], Int(args[3]), Int(args[4]), args[5] == "strided")
    else:
        run[DType.float64](args[2], Int(args[3]), Int(args[4]), args[5] == "strided")
