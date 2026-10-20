"""Native timed region; invoked by pyperf, not a performance assertion."""
from std.sys import argv
from std.time import perf_counter_ns
from numojo import zeros, Shape
from moxie.feature_extraction.image import extract_patches_2d, grid_to_graph

def main() raises:
    var args = argv()
    if len(args) < 4:
        return
    var loops = Int(args[1])
    var n = Int(args[2])
    var image = zeros[DType.float64](Shape(n,n))
    var total = Float64(0)
    var start = perf_counter_ns()
    for k in range(loops):
        image.unsafe_set(0,Float64(k))
        if args[3] == "patch":
            var p = extract_patches_2d(image,(2,2))
            total += p.unsafe_get(0)
        else:
            var g = grid_to_graph(n,n,return_as="dense")
            total += Float64(g.unsafe_get(0))
    var elapsed = perf_counter_ns()-start
    print(Float64(elapsed)/1e9)
    print(total)
