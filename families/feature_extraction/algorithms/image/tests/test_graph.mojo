from std.testing import assert_equal, assert_raises
from numojo import zeros, Shape
from moxie.feature_extraction.image import grid_to_graph, img_to_graph

def main() raises:
    var grid = grid_to_graph(2, 3, return_as="dense")
    assert_equal(grid.shape[0], 6)
    assert_equal(grid.unsafe_get(0), 1)
    assert_equal(grid.unsafe_get(1), 1)
    assert_equal(grid.unsafe_get(2), 0)
    var x = zeros[DType.float64](Shape(2, 3))
    for i in range(6):
        x.unsafe_set(i, Float64(i)-3)
    var mask = zeros[DType.bool](Shape(2, 3))
    mask.unsafe_set(0, True)
    mask.unsafe_set(2, True)
    mask.unsafe_set(5, True)
    var graph = img_to_graph(x, mask=mask, return_as="dense")
    assert_equal(graph.shape[0], 3)
    assert_equal(graph.unsafe_get(0), -3)
    assert_equal(graph.unsafe_get(5), 3)
    with assert_raises():
        _ = grid_to_graph(1, 1)
    with assert_raises():
        _ = img_to_graph(x)
    print("PASS graph")
