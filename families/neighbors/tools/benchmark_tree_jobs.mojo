"""Time fitted public tree queries for the pyperf scheduling comparison."""

import numojo as nm
from moxie.neighbors import NearestNeighbors
from std.sys import argv
from std.time import perf_counter_ns


def feature_rows(rows: Int, seed: Int) raises -> nm.NDArray[DType.float64]:
    var x = nm.zeros[DType.float64](nm.Shape(rows, 5))
    var state = seed
    for i in range(x.size):
        state = (1_103_515_245 * state + 12_345) % 2_147_483_648
        x.unsafe_set(i, Float64(state) / 2_147_483_648)
    return x^


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error("usage: benchmark_tree_jobs ALGORITHM N_JOBS LOOPS")
    var loops = Int(args[3])
    var training = feature_rows(8192, 42)
    var queries = feature_rows(512, 43)
    var model = NearestNeighbors(
        n_neighbors=10, algorithm=String(args[1]), n_jobs=Int(args[2])
    )
    _ = model.fit(training)
    var first = model.kneighbors(queries)
    if loops == 0:
        ref distances = first.distances.value()
        for i in range(len(first.indices)):
            print(first.indices[i], distances.unsafe_get(i))
        return

    var checksum = Int(0)
    var start = perf_counter_ns()
    for i in range(loops):
        var result = model.kneighbors(queries)
        checksum += result.indices[i % len(result.indices)]
    var elapsed = perf_counter_ns() - start
    print(Float64(elapsed) / 1e9, checksum)
