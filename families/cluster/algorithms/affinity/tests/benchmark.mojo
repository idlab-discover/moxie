"""Native timed region; input construction and executable startup excluded."""
from std.sys import argv
from std.time import perf_counter_ns
from moxie.cluster import AffinityPropagation
import numojo as nm


def run[dtype: DType](n: Int, rounds: Int) raises:
    var x = nm.zeros[dtype](nm.Shape(n, 4))
    for i in range(n):
        for j in range(4):
            x[nm.Item(i, j)] = SIMD[dtype, 1]((i * 7 + j * 11) % 41) / 13
    var checksum = Int(0)
    var start = perf_counter_ns()
    for _ in range(rounds):
        var model = AffinityPropagation[dtype](random_state=42, max_iter=40)
        _ = model.fit(x)
        checksum += model.n_iter_ + model.cluster_centers_indices_.size
    print(Float64(perf_counter_ns() - start) / 1e9)
    print(checksum)


def main() raises:
    var args = argv()
    if args[1] == "32": run[DType.float32](Int(args[2]), Int(args[3]))
    else: run[DType.float64](Int(args[2]), Int(args[3]))
