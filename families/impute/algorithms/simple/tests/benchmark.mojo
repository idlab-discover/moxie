from std.sys import argv
from std.time import perf_counter_ns
import numojo as nm
from moxie.impute import SimpleImputer


def main() raises:
    var args = argv()
    var rows = Int(args[1])
    var rounds = Int(args[2])
    var X = nm.zeros[DType.float64](nm.Shape(rows, 16))
    for i in range(rows * 16):
        X.unsafe_set(i, Float64("nan") if i % 7 == 0 else Float64(i % 101))
    var model = SimpleImputer(add_indicator=True)
    var checksum = Float64(0)
    var start = perf_counter_ns()
    for step in range(rounds):
        X.unsafe_set(1, Float64(step % 19))
        var output = model.fit_transform(X)
        checksum += output.unsafe_get(0)
    print(Float64(perf_counter_ns() - start) / 1e9)
    print(checksum)
