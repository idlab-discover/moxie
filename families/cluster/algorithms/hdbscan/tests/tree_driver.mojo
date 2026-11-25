import numojo as nm
from std.sys import argv
from moxie.cluster._contracts import MergeTree
from moxie.cluster._hdbscan_linkage import dense_brute_hierarchy
from moxie.cluster._hdbscan_tree import condense_tree, tree_to_labels


def _precomputed[dtype: DType](n: Int) raises -> MergeTree:
    var args = argv()
    var matrix = nm.zeros[dtype](nm.Shape(n, 2 * n))
    for i in range(n):
        for j in range(n):
            matrix[nm.Item(n - 1 - i, 2 * j)] = Float64(
                args[10 + i * n + j]
            ).cast[dtype]()
    var distances = matrix.view_with_layout(
        nm.Shape(n, n), nm.Strides(-2 * n, 2), 2 * n * (n - 1)
    )
    return dense_brute_hierarchy(distances, Int(args[8]), Float64(args[9]))


def _linkage(n: Int, layout: String) raises -> MergeTree:
    var args = argv()
    var children = nm.zeros[DType.int64](nm.Shape(n - 1, 4))
    var heights = nm.zeros[DType.float64](nm.Shape(2 * (n - 1)))
    for row in range(n - 1):
        var arg = 8 + row * 3
        children[nm.Item(n - 2 - row, 0)] = Int64(Int(args[arg]))
        children[nm.Item(n - 2 - row, 2)] = Int64(Int(args[arg + 1]))
        heights.unsafe_set(2 * row, Float64(args[arg + 2]))
    var child_view = children.view_with_layout(
        nm.Shape(n - 1, 2), nm.Strides(-4, 2), 4 * (n - 2)
    )
    var height_view = heights.view_with_layout(
        nm.Shape(n - 1), nm.Strides(2), 0
    )
    var components = 1
    if layout == "forest":
        components = 2
    var distances = Optional[nm.NDArray[DType.float64]](height_view^)
    if layout == "missing":
        distances = None
    return MergeTree(child_view^, n, components, None, distances^)


def main() raises:
    var args = argv()
    var n = Int(args[1])
    var min_size = Int(args[2])
    var method = String(args[3])
    var allow_root = args[4] == "1"
    var epsilon = Float64(args[5])
    var max_size = Optional[Int]()
    if args[6] != "None":
        max_size = Int(args[6])
    var layout = String(args[7])
    var tree: MergeTree
    if layout == "precomputed32":
        tree = _precomputed[DType.float32](n)
    elif layout == "precomputed64":
        tree = _precomputed[DType.float64](n)
    else:
        tree = _linkage(n, layout)
    var result = tree_to_labels(
        tree,
        min_size,
        cluster_selection_method=method,
        allow_single_cluster=allow_root,
        cluster_selection_epsilon=epsilon,
        max_cluster_size=max_size,
    )
    for i in range(n):
        print(result.labels.item(i), result.probabilities.item(i))

    var condensed = condense_tree(tree, min_size)
    print("edges")
    for edge in condensed.edges:
        print(edge.parent, edge.child, edge.lambda_value, edge.size)
