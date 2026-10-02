from numojo import NDArray, Shape
from moxie.preprocessing._dense_transforms import transform, normalize
from std.testing import assert_equal

def main() raises:
    var x = NDArray[DType.float64](Shape(1, 2))
    x.unsafe_set(0, 3)
    x.unsafe_set(1, 4)
    var n = normalize(x)
    assert_equal(n.unsafe_get(0), 0.6)
    print("PASS basic")
