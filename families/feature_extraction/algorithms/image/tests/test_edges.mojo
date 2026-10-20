from std.testing import assert_equal, assert_raises, assert_true
from std.math import isnan
from numojo import zeros, Shape
from moxie.feature_extraction.image import img_to_graph, grid_to_graph, extract_patches_2d, reconstruct_from_patches_2d, PatchLimit

def main() raises:
    var image = zeros[DType.uint64](Shape(1,2))
    image.unsafe_set(0,UInt64(9223372036854775809))
    image.unsafe_set(1,UInt64(9223372036854775810))
    var patches = extract_patches_2d(image,(1,1))
    assert_equal(patches.unsafe_get(0),UInt64(9223372036854775809))
    var graph = img_to_graph(image,return_as="dense")
    assert_equal(graph.unsafe_get(1),UInt64.MAX)
    var mask = zeros[DType.bool](Shape(2))
    var none = img_to_graph(image,mask=mask,return_as="dense")
    assert_equal(none.shape[0],0)
    var boolean = zeros[DType.bool](Shape(2,2))
    with assert_raises():
        _ = img_to_graph(boolean,return_as="dense")
    with assert_raises():
        _ = grid_to_graph(Int.MAX,2,return_as="dense")
    with assert_raises():
        _ = grid_to_graph(Int.MAX,1,return_as="dense")
    with assert_raises():
        _ = extract_patches_2d(image,(0,1))
    with assert_raises():
        _ = extract_patches_2d(image,(1,1),max_patches=PatchLimit(1,0))
    var real = zeros[DType.float64](Shape(2,2))
    real.unsafe_set(0,Float64("nan"))
    var nan_graph = img_to_graph(real,return_as="dense")
    assert_true(isnan(nan_graph.unsafe_get(0)))
    assert_true(isnan(nan_graph.unsafe_get(1)))
    for i in range(4):
        real.unsafe_set(i,Float64(i)/7.0)
    var p = extract_patches_2d(real,(1,2))
    var recovered = reconstruct_from_patches_2d(p,Shape(2,2))
    for i in range(4):
        assert_equal(real.unsafe_get(i),recovered.unsafe_get(i))
    print("PASS overflow, empty masks, bool rejection and nonfinite graphs")
