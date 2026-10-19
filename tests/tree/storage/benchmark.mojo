"""Native predict timing; construction and process startup are excluded."""
from std.sys import argv
from std.time import perf_counter_ns
from numojo import Shape, zeros
from moxie.tree._records import Node, CategoryBits
from moxie.tree._tree import Tree, new_tree

def main() raises:
    var args = argv()
    var rows = Int(args[1])
    var reps = Int(args[2])
    var engine = Tree()
    var tree = new_tree(1, [1], [-1])
    # Balanced depth-six tree, breadth-first allocation.
    for i in range(127):
        var kind = 0 if i < 63 else -1
        _ = engine.add_node(tree, -2 if i == 0 else (i - 1) // 2, i % 2 == 1,
            Node(-1, -1, 0, 0.5, CategoryBits(0), 0, 1, 1, 0, Int8(kind)))
        tree.value.unsafe_set(i, Float64(i))
    var X = zeros[DType.float32](Shape(rows, 1))
    for i in range(rows):
        X.unsafe_set(i, Float32(i % 2))
    var checksum = Float64(0)
    var start = perf_counter_ns()
    for rep in range(reps):
        X.unsafe_set(0, Float32(rep % 2))
        var result = engine.predict(tree, X)
        checksum += result.unsafe_get(0) + result.unsafe_get(rows - 1)
    var elapsed = perf_counter_ns() - start
    print(Float64(elapsed) / 1e9, checksum)
