from numojo import full, Shape
from std.testing import assert_equal, assert_raises
from moxie._random import RandomState
from moxie.probe_a import Probe as A
from moxie.probe_b import Probe as B

def main() raises:
    var rng = RandomState(42)
    var cloned_rng = rng.copy()
    assert_equal(rng.next_uint32(), cloned_rng.next_uint32())
    var x = full[DType.float64](Shape(2, 1), 3)
    var a = A()
    var b = B()
    a.fit(x)
    b.fit(x)
    assert_equal(a.total(), b.total())
    assert_equal(a.total(), 6)
    var view = x.view()
    view.unsafe_set(0, 9)
    assert_equal(a.total(), 6)
    b.fit(x)
    assert_equal(b.total(), 12)
    var unfitted = a.clone()
    with assert_raises():
        _ = unfitted.total()
    print("PASS isolated composed consumer")
