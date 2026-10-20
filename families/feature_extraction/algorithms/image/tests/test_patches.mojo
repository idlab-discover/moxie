from std.testing import assert_equal, assert_raises
from numojo import zeros, Shape
from moxie.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d, PatchLimit
from moxie._random import RandomState

def main() raises:
    var x = zeros[DType.uint8](Shape(3, 4))
    for i in range(12):
        x.unsafe_set(i, UInt8(i))
    var p = extract_patches_2d(x, (2,2))
    assert_equal(p.shape[0],6)
    assert_equal(p.unsafe_get(4),1)
    var r = reconstruct_from_patches_2d(p, Shape(3,4))
    for i in range(12):
        assert_equal(r.unsafe_get(i), Float64(i))
    var rng = RandomState(0)
    var sample = extract_patches_2d(x,(2,2),max_patches=PatchLimit(1,2),random_state=rng)
    assert_equal(sample.shape[0],2)
    with assert_raises():
        _ = extract_patches_2d(x,(4,2))
    print("PASS patches")
