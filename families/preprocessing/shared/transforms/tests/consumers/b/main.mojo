from numojo import NDArray, Shape
from moxie.preprocessing._dense_transforms import transform, transform_inplace
from std.testing import assert_equal

def main() raises:
    var x=NDArray[DType.float32](Shape(1,2))
    var shift=NDArray[DType.float32](Shape(2))
    var scale=NDArray[DType.float32](Shape(2))
    for c in range(2):
        x.unsafe_set(c,Float32(c+3))
        shift.unsafe_set(c,1)
        scale.unsafe_set(c,2)
    var y=transform(x,shift,scale)
    assert_equal(y.unsafe_get(0),1)
    assert_equal(x.unsafe_get(0),3)
    transform_inplace(y,shift,scale,operation=1)
    assert_equal(y.unsafe_get(1),4)
    print("PASS consumer B: affine forward/inverse shared engine")
