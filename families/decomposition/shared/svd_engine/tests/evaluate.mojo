"""Validation-only CLI; input values and output factors cross the process boundary."""
from std.sys import argv
from numojo import NDArray, Shape, zeros
from moxie._random import RandomState
from moxie.decomposition._svd_engine import dense_svd, symmetric_eigh, randomized_svd, randomized_range_finder, qr_basis, golub_kahan_svd


def emit[dtype: DType](name: String, x: NDArray[dtype]) raises:
    var flat = x.flatten()
    for i in range(flat.size):
        print(name, Float64(flat.unsafe_get(i)))


def evaluate[dtype: DType]() raises:
    var args = argv()
    var m = Int(args[3])
    var n = Int(args[4])
    var a = zeros[dtype](Shape(m, n))
    for i in range(m * n):
        a.unsafe_set(i, SIMD[dtype, 1](Float64(args[12 + i])))
    var mode = String(args[2])
    if mode == "eigh":
        var result = symmetric_eigh(a)
        emit("s", result.eigenvalues)
        emit("u", result.eigenvectors)
    elif mode == "qr":
        emit("u", qr_basis(a))
    elif mode == "range":
        var rng = RandomState(Int(args[8]))
        emit("u", randomized_range_finder(a, rng, size=Int(args[5]), n_iter=Int(args[7]), power_iteration_normalizer=args[9]))
        print("draw", rng.standard_normal())
    elif mode == "svd" or mode == "gesvd":
        var result = dense_svd(a) if mode == "svd" else golub_kahan_svd(a)
        emit("u", result.U)
        emit("s", result.singular_values)
        emit("v", result.Vt)
    else:
        var rng = RandomState(Int(args[8]))
        var result = randomized_svd(a, Int(args[5]), rng, n_oversamples=Int(args[6]), n_iter=Int(args[7]), power_iteration_normalizer=args[9], transpose=args[10], svd_lapack_driver=args[11])
        emit("u", result.U)
        emit("s", result.singular_values)
        emit("v", result.Vt)
        print("draw", rng.standard_normal())


def main() raises:
    var args = argv()
    if args[1] == "f32":
        evaluate[DType.float32]()
    else:
        evaluate[DType.float64]()
