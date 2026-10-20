"""Compile known shapes for the common sklearn differential driver."""
from std.sys import argv
from moxie_numax import NumaxPCABackend
from evaluate import evaluate


def dispatch[dtype: DType]() raises:
    var args = argv()
    var n = Int(args[2])
    var p = Int(args[3])
    if n == 12 and p == 4:
        evaluate[dtype, NumaxPCABackend[12, 4]]()
        return
    if n == 4 and p == 12:
        evaluate[dtype, NumaxPCABackend[4, 12]]()
        return
    if n == 50 and p == 3:
        evaluate[dtype, NumaxPCABackend[50, 3]]()
        return
    if n == 7 and p == 1:
        evaluate[dtype, NumaxPCABackend[7, 1]]()
        return
    if n == 1 and p == 5:
        evaluate[dtype, NumaxPCABackend[1, 5]]()
        return
    if n == 24 and p == 5:
        evaluate[dtype, NumaxPCABackend[24, 5]]()
        return
    if n == 5 and p == 24:
        evaluate[dtype, NumaxPCABackend[5, 24]]()
        return
    if n == 24 and p == 3:
        evaluate[dtype, NumaxPCABackend[24, 3]]()
        return
    if n == 8 and p == 3:
        evaluate[dtype, NumaxPCABackend[8, 3]]()
        return
    raise Error("Validation driver has no specialization for this shape")


def main() raises:
    var args = argv()
    if args[1] == "f32": dispatch[DType.float32]()
    else: dispatch[DType.float64]()
