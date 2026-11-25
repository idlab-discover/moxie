import numojo as nm
from std.sys import argv
from std.testing import assert_equal
from moxie.cluster._hdbscan_linkage import dense_brute_hierarchy


def run[dtype: DType]() raises:
    var args = argv()
    var rows = Int(args[2])
    var columns = Int(args[3])
    var layout = args[6]
    var input = nm.zeros[dtype](nm.Shape(rows, columns))
    for i in range(rows * columns):
        input.unsafe_set(i, Float64(args[7 + i]).cast[dtype]())

    var distances = input.view()
    if layout == "strided":
        var backing = nm.zeros[dtype](nm.Shape(rows, 2 * columns))
        for i in range(rows):
            for j in range(columns):
                backing[nm.Item(rows - 1 - i, 2 * j)] = input.item(i, j)
        distances = backing.view_with_layout(
            nm.Shape(rows, columns),
            nm.Strides(-2 * columns, 2),
            (rows - 1) * 2 * columns,
        )
    elif layout == "flat":
        distances = input.reshape(nm.Shape(rows * columns))

    var tree = dense_brute_hierarchy(distances, Int(args[4]), Float64(args[5]))
    tree.validate()
    assert_equal(tree.n_leaves, rows)
    assert_equal(tree.n_connected_components, 1)
    assert_equal(tree.children.shape[0], rows - 1)
    ref heights = tree.distances.value()
    for i in range(rows - 1):
        print(
            tree.children.item(i, 0), tree.children.item(i, 1), heights.item(i)
        )
    for i in range(rows):
        for j in range(columns):
            assert_equal(
                distances.item(i, j),
                Float64(args[7 + i * columns + j]).cast[dtype](),
            )


def main() raises:
    if argv()[1] == "32":
        run[DType.float32]()
    else:
        run[DType.float64]()
