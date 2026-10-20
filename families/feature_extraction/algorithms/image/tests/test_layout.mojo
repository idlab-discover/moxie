from std.testing import assert_equal, assert_raises
from numojo import zeros, Shape, NDArrayStrides
from moxie.feature_extraction.image import extract_patches_2d, img_to_graph, grid_to_graph

def check[dtype: DType]() raises:
    var owner = zeros[dtype](Shape(4,8))
    for i in range(owner.size):
        owner.unsafe_set(i,SIMD[dtype,1](i))
    var view = owner.view_with_layout(Shape(3,4),NDArrayStrides(8,-2),7)
    var patches = extract_patches_2d(view,(2,2))
    assert_equal(patches.unsafe_get(0),SIMD[dtype,1](7))
    assert_equal(patches.unsafe_get(1),SIMD[dtype,1](5))
    assert_equal(patches.unsafe_get(2),SIMD[dtype,1](15))
    patches.unsafe_set(0,0)
    assert_equal(owner.unsafe_get(7),SIMD[dtype,1](7))
    var graph = img_to_graph(view,return_as="dense")
    assert_equal(graph.unsafe_get(0),SIMD[dtype,1](7))
    assert_equal(graph.unsafe_get(1),SIMD[dtype,1](2))
    var wide = img_to_graph[output_dtype=DType.float64](view,return_as="dense")
    assert_equal(wide.unsafe_get(1),2.0)

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    check[DType.int16]()
    check[DType.uint8]()
    var masks = zeros[DType.bool](Shape(8))
    masks.unsafe_set(7,True)
    masks.unsafe_set(1,True)
    var mask = masks.view_with_layout(Shape(2,2),NDArrayStrides(-4,-2),7)
    var g = grid_to_graph(2,2,mask=mask,return_as="dense")
    assert_equal(g.shape[0],2)
    assert_equal(g.unsafe_get(1),0)
    var bad = zeros[DType.float64](Shape(2,2))
    bad.unsafe_set(1,Float64("nan"))
    with assert_raises():
        _ = extract_patches_2d(bad,(1,1))
    print("PASS strided ownership and finite validation")
