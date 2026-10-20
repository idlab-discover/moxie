"""Expose exact relocation indices and weighted updates to the Python oracle."""
import numojo as nm
from std.sys import argv
from moxie.cluster._centroid_selection import relocation_order, _fallback_sort
from moxie.cluster._centroid_engine import update


def check[dtype: DType](count: Int, fallback: Bool = False) raises:
    var args = argv()
    var n = len(args) - 3
    var residual = nm.zeros[dtype](nm.Shape(n))
    var x = nm.zeros[dtype](nm.Shape(n, 1))
    var weights = nm.zeros[dtype](nm.Shape(n))
    var labels = nm.zeros[DType.int64](nm.Shape(n))
    var centers = nm.zeros[dtype](nm.Shape(count + 1, 1))
    for i in range(n):
        var value = SIMD[dtype, 1](Float64(args[i + 3]))
        residual.unsafe_set(i, value * value)
        x.unsafe_set(i, value)
        weights.unsafe_set(i, SIMD[dtype, 1](i % 13 + 1) / 4)
    var selected = relocation_order(residual, count)
    if fallback:
        var indices = nm.zeros[DType.int64](nm.Shape(n))
        for i in range(n):
            indices.unsafe_set(i, Int64(i))
        _fallback_sort(residual, indices, 0, n)
        for i in range(count):
            selected.unsafe_set(i, indices.unsafe_get(n - 1 - i))
    for i in range(count):
        print("index", selected.item(i))
    if count < n:
        var result = update(x, labels, weights, centers)
        for i in range(count + 1):
            print("center", result.centers.item(i, 0))
            print("mass", result.masses.item(i))


def main() raises:
    var args = argv()
    if args[1] == "f32" or args[1] == "sort32":
        check[DType.float32](Int(args[2]), args[1] == "sort32")
    else:
        check[DType.float64](Int(args[2]), args[1] == "sort64")
