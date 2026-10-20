from std.testing import assert_equal, assert_raises
from numojo import zeros, Shape
from moxie.base import clone
from moxie._random import RandomState
from moxie.feature_extraction.image import PatchExtractor, PatchLimit, PatchParams, extract_patches_2d, grid_to_graph, img_to_graph

def main() raises:
    var x = zeros[DType.float64](Shape(2,3,4))
    for i in range(x.size):
        x.unsafe_set(i, Float64(i))
    var model = PatchExtractor(patch_size=(2,2),max_patches=PatchLimit(1,2),random_state=42)
    var a = model.transform(x)
    model.fit(x)
    var b = model.fit_transform(x)
    for i in range(a.size):
        assert_equal(a.unsafe_get(i),b.unsafe_get(i))
    var other = clone(model)
    var params = other.get_params()
    params.patch_size = (1,1)
    other.set_params(params)
    var small = other.transform(x)
    assert_equal(small.shape[1],1)
    assert_equal(model.patch_size[0],2)
    var rng = RandomState(42)
    var borrowed = model.transform(x,random_state=rng)
    for i in range(a.size):
        assert_equal(a.unsafe_get(i),borrowed.unsafe_get(i))
    var fit_rng = RandomState(42)
    var fit_borrowed = model.fit_transform(x,random_state=fit_rng)
    for i in range(a.size):
        assert_equal(a.unsafe_get(i),fit_borrowed.unsafe_get(i))
    var fit_after = fit_rng.copy()
    var transform_after = rng.copy()
    assert_equal(fit_after.next_uint32(),transform_after.next_uint32())
    other.set_params(patch_size=(2,1))
    assert_equal(other.max_patches.count,2)
    assert_equal(other.random_state,42)
    other.set_params(random_state=0,max_patches=PatchLimit())
    other.set_params()
    var all_patches = other.set_params(patch_size=(2,1)).fit(x).transform(x)
    assert_equal(all_patches.shape[0],16)
    assert_equal(all_patches.shape[1],2)
    assert_equal(all_patches.shape[2],1)
    other.set_params(patch_size=(-1,-1),random_state=-1)
    assert_equal(other.patch_size[0],-1)
    assert_equal(other.random_state,-1)
    var rng_after = rng.copy()
    _ = model.transform(x,random_state=rng)
    # The caller's stream advances and copying it is independent.
    var word = rng.next_uint32()
    if word == rng_after.next_uint32():
        raise Error("RNG stream unexpectedly reset")
    var bad = PatchExtractor(patch_size=(2,2),max_patches=PatchLimit(2,0,1.0))
    with assert_raises():
        bad.fit(x)
    bad.set_params(max_patches=PatchLimit(1,0))
    var untouched = RandomState(42)
    var expected_untouched = untouched.copy()
    with assert_raises():
        _ = bad.fit_transform(x,random_state=untouched)
    assert_equal(untouched.next_uint32(),expected_untouched.next_uint32())
    var empty = zeros[DType.float64](Shape(0,3,4))
    with assert_raises():
        _ = model.transform(empty)
    var image = zeros[DType.float64](Shape(3,4))
    with assert_raises():
        _ = extract_patches_2d(image,(1,1),max_patches=PatchLimit(1,2))
    with assert_raises():
        _ = extract_patches_2d(image,(1,1),random_state=-2)
    with assert_raises():
        _ = grid_to_graph(0,2,return_as="dense")
    with assert_raises():
        _ = grid_to_graph(2,2,return_as="unknown")
    var mask = zeros[DType.bool](Shape(1))
    with assert_raises():
        _ = img_to_graph(image,mask=mask,return_as="dense")
    print("PASS lifecycle and invalid inputs")
