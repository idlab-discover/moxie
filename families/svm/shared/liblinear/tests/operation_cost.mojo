"""Compare solver's private TRON contraction with an eager NuMojo composition."""
from std.sys import argv
from std.time import perf_counter_ns
from std.testing import assert_true
from std.math import abs
import numojo as nm
from moxie.svm._liblinear import _inner

def main() raises:
    var args = argv()
    var p = Int(args[1])
    var reps = Int(args[2])
    var mode = args[3]
    var a = nm.ones[DType.float64](nm.Shape(p))
    var b = nm.ones[DType.float64](nm.Shape(p))
    for i in range(p):
        a.unsafe_set(i, Float64(i % 7 - 3) / 8)
        b.unsafe_set(i, Float64(i % 11 - 5) / 8)
    var expected = nm.sum(a * b)
    assert_true(abs(_inner(a, b) - expected) < 1e-10)
    var checksum = Float64(0)
    var start = perf_counter_ns()
    for i in range(reps):
        a.unsafe_set(0, Float64(i % 3) / 8)
        if mode == "native":
            checksum += _inner(a, b)
        else:
            checksum += nm.sum(a * b)
    print(Float64(perf_counter_ns() - start) * 1e-9, checksum)
