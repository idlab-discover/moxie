"""Native elapsed time for complete solves; process startup excluded."""
from std.sys import argv
from std.time import perf_counter_ns
import numojo as nm
from moxie.svm._contracts import LinearOptions
from moxie.svm._liblinear import LibLinear


def main() raises:
    var args = argv()
    var n = Int(args[1])
    var p = Int(args[2])
    var solver = Int(args[3])
    var reps = Int(args[4])
    var x = nm.zeros[DType.float64](nm.Shape(n, p))
    var y = nm.zeros[DType.float64](nm.Shape(n))
    var sw = nm.ones[DType.float64](nm.Shape(n))
    var cw = nm.ones[DType.float64](nm.Shape(2))
    if solver >= 11:
        cw = nm.zeros[DType.float64](nm.Shape(0))
    for i in range(n):
        for j in range(p):
            x.unsafe_set(i * p + j, Float64((i * 17 + j * 11) % 31 - 15) / 16)
        y.unsafe_set(i, Float64(i % 2))
    var engine = LibLinear()
    var options = LinearOptions(solver, 0.1, 0.1, 1e-4, True, 1, 1000, 42)
    var checksum = Float64(0)
    var start = perf_counter_ns()
    for rep in range(reps):
        x.unsafe_set(0, Float64(rep % 3) / 16)
        var model = engine.solve(x, y, sw, cw, options)
        checksum += model.coef.unsafe_get(0) + Float64(model.iterations.unsafe_get(0))
    var elapsed = Float64(perf_counter_ns() - start) * 1e-9
    print(elapsed, checksum)
