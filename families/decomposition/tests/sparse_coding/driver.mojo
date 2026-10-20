"""Threshold code driver: dtype X D cov alpha positive split use_cov."""
from std.sys import argv
from moxie.decomposition import SparseCoder, sparse_encode
from numojo import NDArray
from decomposition_testing import read_matrix, emit


def evaluate[dtype: DType]() raises:
    var a = argv()
    var x = read_matrix[dtype](String(a[2]))
    var d = read_matrix[dtype](String(a[3]))
    var cov: Optional[NDArray[dtype]] = None
    if Int(a[8]) != 0: cov = read_matrix[dtype](String(a[4]))
    var alpha: Optional[Float64] = None
    if Float64(String(a[5])) >= 0: alpha = Float64(String(a[5]))
    var positive = Int(a[6]) != 0
    emit("functional", sparse_encode(x, d, cov=cov, algorithm="threshold", alpha=alpha, positive=positive))
    var m = SparseCoder[dtype](d, transform_algorithm="threshold", transform_alpha=alpha,
        positive_code=positive, split_sign=Int(a[7]) != 0)
    var z = m.transform(x)
    emit("code", z)
    emit("inverse", m.inverse_transform(z))
    emit("fit", m.fit_transform(x))


def main() raises:
    if argv()[1] == "f32": evaluate[DType.float32]()
    else: evaluate[DType.float64]()
