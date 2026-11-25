import numojo as nm
from std.sys import argv
from std.testing import assert_equal
from moxie.cluster._hdbscan_linkage import dense_feature_hierarchy
from moxie.cluster._hdbscan_tree import tree_to_labels
from moxie.neighbors._distance import Metric


def run[dtype: DType]() raises:
    var args = argv()
    var rows = Int(args[2])
    var columns = Int(args[3])
    var min_samples = Int(args[4])
    var alpha = Float64(args[5])
    var algorithm = String(args[6])
    var metric = String(args[7])
    var power = Float64(args[8])
    var leaf_size = Int(args[9])
    var layout = args[10]
    var method = String(args[11])
    var input = nm.zeros[dtype](nm.Shape(rows, columns))
    for i in range(rows * columns):
        input.unsafe_set(i, Float64(args[12 + i]).cast[dtype]())

    var features = input.view()
    if layout == "strided":
        var backing = nm.zeros[dtype](nm.Shape(rows, 2 * columns))
        for i in range(rows):
            for j in range(columns):
                backing[nm.Item(rows - 1 - i, 2 * j)] = input.item(i, j)
        features = backing.view_with_layout(
            nm.Shape(rows, columns),
            nm.Strides(-2 * columns, 2),
            (rows - 1) * 2 * columns,
        )
    elif layout == "flat":
        features = input.reshape(nm.Shape(rows * columns))
    elif layout == "offset":
        var backing = nm.zeros[dtype](nm.Shape(rows + 1, columns))
        for i in range(rows):
            for j in range(columns):
                backing[nm.Item(i + 1, j)] = input.item(i, j)
        features = backing.view_with_layout(
            nm.Shape(rows, columns), nm.Strides(columns, 1), columns
        )

    var tree = dense_feature_hierarchy(
        features,
        min_samples,
        Metric[DType.float64](metric, p=power),
        alpha=alpha,
        algorithm=algorithm,
        leaf_size=leaf_size,
    )
    tree.validate()
    assert_equal(tree.n_leaves, rows)
    assert_equal(tree.n_connected_components, 1)
    assert_equal(tree.children.shape[0], rows - 1)
    ref heights = tree.distances.value()
    for i in range(rows - 1):
        print(
            tree.children.item(i, 0), tree.children.item(i, 1), heights.item(i)
        )

    var flat = tree_to_labels(tree, 3, cluster_selection_method=method)
    print("points")
    for i in range(rows):
        print(flat.labels.item(i), flat.probabilities.item(i))
        for j in range(columns):
            assert_equal(features.item(i, j), input.item(i, j))


def main() raises:
    if argv()[1] == "32":
        run[DType.float32]()
    else:
        run[DType.float64]()
